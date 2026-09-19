# INTERFACE.md — Kokoro-Shogi データ契約 (正本)

> **Schema Version: 1.1**
> AI側 (Python) と Unity側の唯一の接点。この文書にない形のデータを
> 送ってはならず、この文書にないフィールドを期待してはならない。
> 変更手順は§7。正本は本ファイル (リポジトリ `docs/INTERFACE.md`)。

---

## 1. 通信の全体像

- **リプレイ**: AIが対局を JSONL ファイルに書き出す (1行 = 1メッセージ)。
  Unityはファイルを読んで再生する。置き場所:
  `unity/KokoroShogi/Assets/StreamingAssets/sample_data/*.jsonl`
- **ライブ**: AIのWebSocketサーバ `ws://localhost:8765` にUnityが接続。
  1通のテキストメッセージ = JSONLの1行と**完全に同形**。
- 文字コード UTF-8。数値は全てAI側で正規化して送る (Unity側で統計処理しない)。
- 全メッセージ共通フィールド: `"schema": "1.1"` と `"type": "<種別>"`。
  Unityは schema 不一致時に警告表示 (処理は続行してよい)。

## 2. メッセージ種別一覧

| type | 方向 | 用途 | 頻度 |
|---|---|---|---|
| `state_update` | AI→Unity | 1手ごとの盤面+内面データ (§3) | 毎手 |
| `legal_moves` | AI→Unity | 人間手番の合法手リスト (§4) | 人間手番開始時 |
| `move_request` | Unity→AI | 人間の指し手 (§4) | 人間が指した時 |
| `game_control` | 双方向 | start / resign / reset (§4) | 随時 |
| `career` | AI→Unity | 駒の通算戦績 (§5) | 対局終了時 or 起動時 |

## 3. state_update (主メッセージ)

```jsonc
{
  "schema": "1.1",
  "type": "state_update",
  "ply": 42,                      // 手数 (1始まり)。初期局面は ply=0, last_move=null
  "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
  "last_move": {                  // 直前の1手。ply=0ではnull
    "from": "77",                 // 移動元 "筋段"。打ちの場合は "00"
    "to": "76",
    "piece_id": "P77_gen0_0003",
    "capture": false,
    "promote": false,
    "drop": false
  },
  "eval": 0.12,                   // 形勢 V ∈ [-1,+1]。+が手番側有利…ではなく常に先手有利
  "pieces": [                     // 盤上+持ち駒の全駒 (最大40)
    {
      "piece_id": "P77_gen0_0003",// 恒久ID: <初期位置><_gen世代><_連番>
      "species": "FU",            // SFEN準拠: FU KY KE GI KI KA HI OU / 成: TO NY NK NG UM RY
      "owner": 0,                 // 0=先手, 1=後手 (現在の所有者)
      "square": "76",             // 盤上のマス。持ち駒は "hand"
      "mood":   { "fear": 0.7, "aggression": 0.2, "valence": -0.3 },
                                  // fear,aggression ∈ [0,1] / valence ∈ [-1,+1]
      "desire": { "survive": 0.8, "attack": 0.1, "promote": 0.3,
                  "defend": 0.5, "advance": 0.6, "redeploy": 0.0 },  // 各 [0,1]
      "alpha": 0.05,              // 発言力 (mixing係数を正規化) ∈ [0,1]
      "relations": [              // 関係の強い相手のみ (r ≥ 0.3、最大5件/駒)
        { "to": "K59_gen0_0001", "r": 0.9 }
      ]
    }
  ],
  "council": [                    // 会議ログ。会議機能OFF時は空配列 []
    { "round": 1, "proposals": [
        { "piece_id": "B88_gen0_0010", "move": "2d2b+", "bid": 2.3 } ] }
  ],
  "narration": "3ラウンド目、角が2二への成り込みを強く主張した。"
                                  // 実況文。なければ空文字 ""
}
```

**Unity側の消費者マップ** (誰がどのフィールドを読むか):
- U1: schema, ply, sfen, last_move
- U2: pieces[].mood / alpha / relations, council, narration, eval
- 未知フィールドは無視する (前方互換のため)

## 4. 人間対局 (週8)

```jsonc
// AI→Unity: 人間手番の開始と同時に配信
{ "schema": "1.1", "type": "legal_moves",
  "moves": [ { "from": "77", "to": "76", "promote": false },
             { "from": "00", "to": "55", "drop_species": "FU" } ] }

// Unity→AI: 人間の指し手 (legal_movesに含まれるものだけ送る)
{ "schema": "1.1", "type": "move_request",
  "move": { "from": "77", "to": "76", "promote": false } }

// 双方向: 対局制御
{ "schema": "1.1", "type": "game_control", "command": "start" }
// command: "start" | "resign" | "reset"
```

状態機械 (これ以外の遷移を実装しない):
```
[待機] --start--> [人間手番] --move_request--> [AI思考中]
                     ↑                              |
                     └──── state_update 受信 ───────┘
任意の状態で resign / reset → [待機]
```
- 合法手の判定はAI側の責務。Unityは legal_moves を表示・選択させるだけ
- 不正な move_request が来た場合、AIは legal_moves を再送する (エラー扱いにしない)

## 5. career (キャリアUI用、週9)

```jsonc
{ "schema": "1.1", "type": "career",
  "pieces": [
    { "piece_id": "P77_gen0_0003", "species": "FU",
      "games": 120, "survival_rate": 0.42,
      "promotions": 18, "mvp_count": 3 } ],
  "last_game_mvp": { "piece_id": "R28_gen0_0011", "contribution": 2.3 },

  // 終局時だけ付く (§5.1)。起動時の成績配信ではキーごと省く
  "result": { "winner": "white", "human": 0, "reason": "checkmate" } }
```
ライブでは対局終了時に配信。リプレイ用には同形の `career.json` を
StreamingAssets に置く。

### 5.1 result (終局通知)

**終局時に「誰が勝ったか・なぜ終わったか」を伝えるのはこのフィールドだけ**。
`career` は起動時の成績配信にも使うので、**`result` の無い `career` を終局扱いにしない**こと。
終局でないときは `result: null` ではなく**キー自体を出さない**。

| フィールド | 値 | 意味 |
|---|---|---|
| `winner` | `black` / `white` / `draw` | 盤の先後で表す。**`black` は常に先手で、人間とは限らない** |
| `human` | `0` / `1` | 人間の手番 (0=先手 / 1=後手)。Unity が「あなたの勝ち」を出すのに使う |
| `reason` | 下表 | 終了理由 |

| `reason` | 意味 | winner |
|---|---|---|
| `checkmate` | 王手されていて合法手がない | 手番の反対側 |
| `no_legal_moves` | 王手ではないが合法手がない | 手番の反対側 |
| `repetition` | 千日手 (現在の実装で引き分け判定) | `draw` |
| `max_plies` | 手数上限 (既定 320 手) に到達 | `draw` |
| `resign` | 人間が投了 | AI 側 |
| `engine_no_move` | 合法手があるのにエンジンが手を返せない | `draw` (**異常終了**。詰みと誤表示しないための区別で、成績上の引き分けではない) |

**送出順**:

```
最後の state_update → result 付き career → セッションは待機へ (以後の着手を受け付けない)
```

終局時に `legal_moves` は送らない。投了は盤面が変わらないので追加の `state_update` は無い。
`board.turn` は「これから指す側」なので、最後の着手後に詰んでいればその**反対側**が勝者。
詰みと手数上限が同じ手で成立したときは詰みを優先する。

## 6. SFEN仕様 (Unity実装者向け最小知識)

スペース区切り4要素: `盤面 手番 持ち駒 手数`
- 盤面: 9段を `/` 区切り、**上(一段目)から**、各段は**左(9筋)から**
- 大文字=先手 / 小文字=後手。P歩 L香 N桂 S銀 G金 B角 R飛 K玉。`+P`=と金
- 数字 = 連続する空マスの数
- 持ち駒: 例 `2Pb` = 先手歩2・後手角1。なし = `-`
- Unityは盤面部分のみパースすればよい (手番・手数は state_update の他フィールドで足りる)

座標: square "76" = 7筋6段。Unityワールド座標変換は
`x = (5 - 筋) * 0.036` / `z = (段 - 5) * 0.033` を基準に盤モデル実寸で確定 (U1↔B2)。

## 7. スキーマ変更手順 (儀式)

1. 変更したい人 → U1に相談 → U1がIssue化
2. Taka承認 → 本ファイル更新 + バージョン番号を上げる (後方互換の追加は
   1.0→1.1、破壊的変更は2.0)
3. Taka が `scripts/gen_sample_jsonl.py` を更新 → `sample_data/` 再生成
4. U1がUnity側のデシリアライズクラスを同期
5. Discordで全員に「schema x.y になりました」通知

**禁止事項**: 本文書にないフィールドの無断追加 / フィールド名のリネーム /
値域の変更。困ったら足す前にIssueへ。

## 9. piece_id の対応表 (恒久ID)

`piece_id` は**対局開始時に 40 枚へ一度だけ割り当て、対局が終わるまで変わらない**追跡キー。
形式は `<駒種のSFEN文字><初期マス>_gen<世代>_<連番4桁>`:

```
P77_gen0_0029
│ │  │     └─ 連番 (初期局面のマス順に 1〜40)
│ │  └─ 世代 (血統用。現在は常に gen0)
│ └─ 初期マス (筋段。77 = 7七)
└─ 生駒のSFEN文字 (P歩 L香 N桂 S銀 G金 B角 R飛 K玉)
```

### id をパースして駒種・手番を決めないこと

**id は「最初に何だったか」であり「今どうか」ではない。** 実測した挙動 (先手の角 `B88_gen0_0035`):

| 場面 | `species` | `owner` | `square` |
|---|---|---|---|
| 初期 | KA | 0 (先手) | 88 |
| 8八→2二 と成った | **UM (馬)** | 0 | 22 |
| 後手に取られた | KA | **1 (後手)** | `"hand"` |
| 後手が 5五に打った | KA | 1 | 55 |

id は `B88_gen0_0035` のまま一貫する。成っても・取られて相手の持ち駒になっても・打ち直されても
変わらない (「駒が敵に転生する」を表現するための設計)。したがって描画・エフェクトでは:

| 知りたいこと | 見るフィールド |
|---|---|
| どの駒か (追跡キー) | `piece_id` |
| 今の駒種 | `species` |
| 今どちら側か | `owner` (0=先手 / 1=後手) |
| 今どこにいるか | `square` (持ち駒は `"hand"`) |

### 初期配置の全 40 枚

`state_update.pieces` は piece_id の辞書順に並ぶ。

| piece_id | species | owner | 初期マス | 駒 |
|---|---|---|---|---|
| `B22_gen0_0006` | KA | 1 | 22 | 後手の角 |
| `B88_gen0_0035` | KA | 0 | 88 | 先手の角 |
| `G41_gen0_0015` | KI | 1 | 41 | 後手の金 |
| `G49_gen0_0018` | KI | 0 | 49 | 先手の金 |
| `G61_gen0_0023` | KI | 1 | 61 | 後手の金 |
| `G69_gen0_0026` | KI | 0 | 69 | 先手の金 |
| `K51_gen0_0019` | OU | 1 | 51 | 後手の玉 |
| `K59_gen0_0022` | OU | 0 | 59 | 先手の玉 |
| `L11_gen0_0001` | KY | 1 | 11 | 後手の香 |
| `L19_gen0_0004` | KY | 0 | 19 | 先手の香 |
| `L91_gen0_0037` | KY | 1 | 91 | 後手の香 |
| `L99_gen0_0040` | KY | 0 | 99 | 先手の香 |
| `N21_gen0_0005` | KE | 1 | 21 | 後手の桂 |
| `N29_gen0_0010` | KE | 0 | 29 | 先手の桂 |
| `N81_gen0_0031` | KE | 1 | 81 | 後手の桂 |
| `N89_gen0_0036` | KE | 0 | 89 | 先手の桂 |
| `P13_gen0_0002` | FU | 1 | 13 | 後手の歩 |
| `P17_gen0_0003` | FU | 0 | 17 | 先手の歩 |
| `P23_gen0_0007` | FU | 1 | 23 | 後手の歩 |
| `P27_gen0_0008` | FU | 0 | 27 | 先手の歩 |
| `P33_gen0_0012` | FU | 1 | 33 | 後手の歩 |
| `P37_gen0_0013` | FU | 0 | 37 | 先手の歩 |
| `P43_gen0_0016` | FU | 1 | 43 | 後手の歩 |
| `P47_gen0_0017` | FU | 0 | 47 | 先手の歩 |
| `P53_gen0_0020` | FU | 1 | 53 | 後手の歩 |
| `P57_gen0_0021` | FU | 0 | 57 | 先手の歩 |
| `P63_gen0_0024` | FU | 1 | 63 | 後手の歩 |
| `P67_gen0_0025` | FU | 0 | 67 | 先手の歩 |
| `P73_gen0_0028` | FU | 1 | 73 | 後手の歩 |
| `P77_gen0_0029` | FU | 0 | 77 | 先手の歩 |
| `P83_gen0_0033` | FU | 1 | 83 | 後手の歩 |
| `P87_gen0_0034` | FU | 0 | 87 | 先手の歩 |
| `P93_gen0_0038` | FU | 1 | 93 | 後手の歩 |
| `P97_gen0_0039` | FU | 0 | 97 | 先手の歩 |
| `R28_gen0_0009` | HI | 0 | 28 | 先手の飛 |
| `R82_gen0_0032` | HI | 1 | 82 | 後手の飛 |
| `S31_gen0_0011` | GI | 1 | 31 | 後手の銀 |
| `S39_gen0_0014` | GI | 0 | 39 | 先手の銀 |
| `S71_gen0_0027` | GI | 1 | 71 | 後手の銀 |
| `S79_gen0_0030` | GI | 0 | 79 | 先手の銀 |

## 8. 変更履歴

| Version | 日付 | 内容 |
|---|---|---|
| 1.0 | (キックオフ日) | 初版: state_update / legal_moves / move_request / game_control / career |
| 1.1 | 2026-09-19 | `career.result` (終局の勝敗と理由) を追加。後方互換の追加なので §7 に従い 1.1 へ。`result` は終局時のみでキーごと省略されるため、1.0 の受け手も壊れない。§9 piece_id 対応表を追記。依頼元: `docs/AI_RESULT_HANDOFF.md` |
