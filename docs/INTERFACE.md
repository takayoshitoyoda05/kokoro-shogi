# INTERFACE.md — Kokoro-Shogi データ契約 (正本)

> **Schema Version: 1.0**
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
- 全メッセージ共通フィールド: `"schema": "1.0"` と `"type": "<種別>"`。
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
  "schema": "1.0",
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
{ "schema": "1.0", "type": "legal_moves",
  "moves": [ { "from": "77", "to": "76", "promote": false },
             { "from": "00", "to": "55", "drop_species": "FU" } ] }

// Unity→AI: 人間の指し手 (legal_movesに含まれるものだけ送る)
{ "schema": "1.0", "type": "move_request",
  "move": { "from": "77", "to": "76", "promote": false } }

// 双方向: 対局制御
{ "schema": "1.0", "type": "game_control", "command": "start" }
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
{ "schema": "1.0", "type": "career",
  "pieces": [
    { "piece_id": "P77_gen0_0003", "species": "FU",
      "games": 120, "survival_rate": 0.42,
      "promotions": 18, "mvp_count": 3 } ],
  "last_game_mvp": { "piece_id": "R28_gen0_0011", "contribution": 2.3 } }
```
ライブでは対局終了時に配信。リプレイ用には同形の `career.json` を
StreamingAssets に置く。

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

## 8. 変更履歴

| Version | 日付 | 内容 |
|---|---|---|
| 1.0 | (キックオフ日) | 初版: state_update / legal_moves / move_request / game_control / career |
