# 個人計画書 U1: Unity-Python連携担当 (完全詳細版 / 10週)

> 親文書: `TEAM_PLAN.md` / データ仕様の正本: `docs/INTERFACE.md`
> この文書だけで作業を始められるよう、コード・設定値・コマンドまで全て記載する。

## 役割

AIエンジン (Python) が出力するデータをUnityに取り込む配管のすべて:
JSONLリプレイ再生、WebSocketライブ受信、盤面状態の管理、C#イベントの発火、
リプレイ操作UI。加えてチーム進行役 (週次MTG司会・タスク再割当て)。

**あなたの成果物の上で U2/B1/B2 全員が作業する。最速で「動く土台」を出すこと。**

---

## 1. 環境構築 (週1にやること・全コマンド)

### 1.1 Unityのインストール
1. Unity Hub をインストール → Installs → **6000.3.8f1 (LTS)** を追加
   - チーム標準はこのLTS版に**固定** (6000.5.1f1 等の非LTS版は入れていても
     使わない。バージョン混在はシーン/Prefabの互換性事故の元)
   - モジュール: 「Windows Build Support (IL2CPP)」にチェック
2. プロジェクト作成: テンプレート **Universal 3D** (Unity 6のURP標準
   テンプレート)、名前 `KokoroShogi`、場所はリポジトリの `unity/` 直下

### 1.2 Gitまわり (リポジトリ管理者としての初期設定)
```bash
# 各自が clone より前に1回だけ (全員に周知)
git lfs install

# .gitattributes (リポジトリルート) — LFS対象
*.fbx  filter=lfs diff=lfs merge=lfs -text
*.png  filter=lfs diff=lfs merge=lfs -text
*.psd  filter=lfs diff=lfs merge=lfs -text
*.wav  filter=lfs diff=lfs merge=lfs -text
*.mp3  filter=lfs diff=lfs merge=lfs -text
*.blend filter=lfs diff=lfs merge=lfs -text

# .gitignore に Unity 標準を追記 (github/gitignore の Unity.gitignore をコピーし
# パスを unity/KokoroShogi/ 配下に合わせる)
unity/KokoroShogi/Library/
unity/KokoroShogi/Temp/
unity/KokoroShogi/Logs/
unity/KokoroShogi/UserSettings/
unity/KokoroShogi/obj/
```

### 1.3 Unityプロジェクト設定 (必ず初週に)
- Edit → Project Settings → Editor →
  - Asset Serialization: **Force Text** (マージ事故対策)
  - Version Control: Visible Meta Files
- Edit → Project Settings → Player → Other Settings →
  - Api Compatibility Level: **.NET Standard 2.1**

### 1.4 パッケージ導入
`unity/KokoroShogi/Packages/manifest.json` の `dependencies` に追記:
```json
"com.unity.nuget.newtonsoft-json": "3.2.1",
"com.cysharp.unitask": "https://github.com/Cysharp/UniTask.git?path=src/UniTask/Assets/Plugins/UniTask",
"com.endel.nativewebsocket": "https://github.com/endel/NativeWebSocket.git#upm"
```
DOTween は Asset Store 版 (無料) を U2 と共有インポート
(Window → Package Manager → My Assets)。

### 1.5 フォルダ構成の作成 (担当割当てをフォルダで固定)
```
Assets/_Project/
├── Scenes/        Replay.unity, Live.unity   (U1)
├── Scripts/Core/  盤面状態・SFEN              (U1)
├── Scripts/Net/   WebSocket・JSONモデル       (U1)
├── Scripts/Replay/ JSONL読込・再生制御        (U1)
├── Scripts/Fx/                                (U2)
├── Scripts/UI/    操作UI=U1, 演出UI=U2
├── Prefabs/ Models/ Materials/ VFX/ UI/ Audio/ (U2/B1/B2)
└── StreamingAssets/sample_data/  サンプルJSONL置き場
```
最後に全員向け `docs/CONTRIBUTING.md` (環境構築手順・ブランチ手順) を書く。

---

## 2. データ仕様 (INTERFACE.md の要点を実装者向けに)

### 2.1 メッセージ = JSONLの1行 = WebSocketの1通 (完全に同形)
```jsonc
{
  "schema": "1.0",
  "type": "state_update",
  "ply": 42,
  "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
  "last_move": {"from": "77", "to": "76", "piece_id": "P77_gen0_0003",
                 "capture": false, "promote": false, "drop": false},
  "eval": 0.12,
  "pieces": [ { "piece_id": "P77_gen0_0003", "species": "FU", "owner": 0,
    "square": "76",
    "mood":   {"fear": 0.7, "aggression": 0.2, "valence": -0.3},
    "desire": {"survive": 0.8, "attack": 0.1, "promote": 0.3,
               "defend": 0.5, "advance": 0.6, "redeploy": 0.0},
    "alpha": 0.05,
    "relations": [{"to": "K59_gen0_0001", "r": 0.9}] } ],
  "council": [{"round": 1, "proposals":
      [{"piece_id": "...", "move": "2d2b+", "bid": 2.3}]}],
  "narration": "3ラウンド目、角が2二への成り込みを強く主張した。"
}
```

### 2.2 SFEN の読み方 (盤面部分だけ実装すればよい)
- スペース区切りで4要素: `盤面 手番 持ち駒 手数`
- 盤面: 9段を `/` 区切り。**上から1段目、左から9筋** の順
- 1文字 = 駒: 大文字=先手 (P歩L香N桂S銀G金B角R飛K玉)、小文字=後手
- 数字 = その数だけ空マス。`+P` = と金 (成り)
- 持ち駒: `2Pb` = 先手歩2枚・後手角1枚。なしは `-`
- **合法手判定・王手判定は実装しない** (AI側が正しい手だけ送る契約)

### 2.3 座標系
- square "76" = 7筋6段。Unityワールド座標への変換は
  `x = (5 - 筋) * マス幅` / `z = (段 - 5) * マス幅` を目安に盤モデルに合わせる
  (B2の盤ができたら実寸で係数を合わせる)

---

## 3. 実装ガイド (コード骨子つき)

### 3.1 JSONデシリアライズ用クラス (Scripts/Net/Messages.cs)
```csharp
using Newtonsoft.Json;
using System.Collections.Generic;

public class StateUpdate {
    public string schema; public string type; public int ply;
    public string sfen; public LastMove last_move; public float eval;
    public List<PieceInfo> pieces; public List<CouncilRound> council;
    public string narration;
}
public class LastMove { public string from; public string to;
    public string piece_id; public bool capture; public bool promote; public bool drop; }
public class Mood { public float fear; public float aggression; public float valence; }
public class Desire { public float survive; public float attack; public float promote;
    public float defend; public float advance; public float redeploy; }
public class Relation { public string to; public float r; }
public class PieceInfo { public string piece_id; public string species;
    public int owner; public string square; public Mood mood; public Desire desire;
    public float alpha; public List<Relation> relations; }
public class Proposal { public string piece_id; public string move; public float bid; }
public class CouncilRound { public int round; public List<Proposal> proposals; }

// 使い方: var su = JsonConvert.DeserializeObject<StateUpdate>(line);
```

### 3.2 イベントハブ (Scripts/Core/GameEvents.cs) — U2との境界
```csharp
using System;
public static class GameEvents {
    // U2はこれらを購読するだけ。JSONやSFENには触れない
    public static event Action<StateUpdate> OnStateUpdated;   // 毎手 (全内面データ)
    public static event Action<LastMove>    OnMovePlayed;     // 駒移動
    public static event Action<string>      OnPieceCaptured;  // piece_id
    public static event Action<string>      OnPiecePromoted;
    public static event Action<string>      OnPieceDropped;
    public static event Action<string>      OnNarration;      // 実況文

    public static void Raise(StateUpdate su) {
        OnStateUpdated?.Invoke(su);
        if (su.last_move != null) {
            OnMovePlayed?.Invoke(su.last_move);
            if (su.last_move.capture) OnPieceCaptured?.Invoke(su.last_move.piece_id);
            if (su.last_move.promote) OnPiecePromoted?.Invoke(su.last_move.piece_id);
            if (su.last_move.drop)    OnPieceDropped?.Invoke(su.last_move.piece_id);
        }
        if (!string.IsNullOrEmpty(su.narration)) OnNarration?.Invoke(su.narration);
    }
}
```

### 3.3 リプレイヤー (Scripts/Replay/ReplayPlayer.cs) の設計
- `StreamingAssets/sample_data/game01.jsonl` を `File.ReadAllLines` で読み込み
- 1行 → `StateUpdate` → 盤面適用 → `GameEvents.Raise(su)`
- 再生制御: 状態は {停止/再生/一時停止}、速度倍率 {0.5/1/2/4}、
  1手送り/戻し (戻しは「最初から ply-1 まで早回し再適用」で実装が簡単)
- **演出との同期**: U2の演出が終わるまで次を待つか、`skipFx` フラグで即送りか。
  実装: `GameEvents` にコールバック完了待ちを入れるより、
  「手と手の間隔 (秒) を速度倍率で決め打ち」の方が単純で事故らない (推奨)

### 3.4 盤面状態 (Scripts/Core/BoardState.cs)
- `Dictionary<string /*piece_id*/, GameObject>` で駒オブジェクトを管理
- 毎手: sfenをパースして「あるべき配置」を作り、差分だけ動かす…は複雑なので、
  **last_move だけ適用する差分方式** を基本にし、`ply` 飛び (シーク時) だけ
  sfenから全再構築する二段構え

### 3.5 WebSocketクライアント (Scripts/Net/WsClient.cs) — 週5
```csharp
using NativeWebSocket;
public class WsClient : MonoBehaviour {
    WebSocket ws;
    async void Start() {
        ws = new WebSocket("ws://localhost:8765");
        ws.OnMessage += bytes => {
            var line = System.Text.Encoding.UTF8.GetString(bytes);
            var su = JsonConvert.DeserializeObject<StateUpdate>(line);
            if (su.schema != "1.0") Debug.LogWarning($"schema不一致: {su.schema}");
            MainThreadQueue.Enqueue(() => GameEvents.Raise(su)); // メインスレッドで
        };
        ws.OnClose += _ => Invoke(nameof(Reconnect), 3f); // 3秒後に再接続
        await ws.Connect();
    }
    void Update() { ws?.DispatchMessageQueue(); }  // NativeWebSocket必須のお作法
}
```
注意: Unityの描画はメインスレッドのみ。OnMessage内で直接GameObjectを
触らない (簡単なQueueクラスを1つ作る)。

### 3.6 リプレイ操作UI (あなたの担当分のUI)
- UGUIのボタン: ⏮ ⏪ ⏯ ⏩ ⏭ + 速度ドロップダウン + シークスライダー
- ReplayPlayer のpublicメソッドを `onClick` に割り当てるだけ。見た目はU2が後で整える

---

### 3.7 人間対局モード (週8・必須)
メッセージ (INTERFACE.md):
```jsonc
// Unity→AI: 人間の指し手
{"type": "move_request", "move": {"from": "77", "to": "76", "promote": false}}
// AI→Unity: 合法手リスト (人間の手番開始時に配信)
{"type": "legal_moves", "moves": [{"from": "77", "to": "76", "promote": false}, ...]}
// 双方向: 対局制御
{"type": "game_control", "command": "start" | "resign" | "reset"}
```
状態機械 (これ以外の遷移を作らない):
```
[待機] --start--> [人間手番] --move_request送信--> [AI思考中]
                     ↑                                  |
                     └------- state_update受信 ---------┘
どの状態でも resign/reset → [待機]
```
実装ポイント:
- 駒クリック: `Physics.Raycast` → 駒のColliderにpiece_idを持たせておく
- 合法手ハイライト: `legal_moves` の to マスに半透明Quadを置く (自前の
  合法手計算は**しない**。AIから受信したリストを表示するだけ)
- 人間手番中はReplayPlayer系の自動進行を止める (状態機械のガード)

## 4. 週次計画 (10週)

| 週 | やること | 完了条件 |
|---|---|---|
| 1 | §1の環境構築すべて + CONTRIBUTING.md + 空シーン2つ | 全員がcloneして開ける |
| 2 | Messages.cs / GameEvents.cs / SFENパーサ / ReplayPlayer v0 (Cube駒で再生) | サンプルJSONL 1局が最後まで流れる |
| 3 | ReplayPlayer v1: 操作UI・シーク・速度 / B1の駒Prefab差し替え口 | 10局全てエラーなし・操作可能 |
| 4 | ★統合点①主担当: 全員の成果を統合、Issue消化 | デモ可能なリプレイ |
| 5 | WsClient + MainThreadQueue + 再接続 + schema検査 | ローカルAIサーバと接続確認 (Takaと) |
| 6 | ライブ観戦モード (Live.unity)、リプレイとイベント共通化の確認 | 実データ受信で盤面が動く |
| 7 | ★統合点②主担当 + ユーザーテスト運営 | ライブ観戦デモ |
| 8 | **人間対局モード (必須)**: 駒クリック→合法手ハイライト→move_request送信→AI応手。game_control状態機械 (下記§3.7) | 人間vs AIが1局完走 |
| 9 | 全体統合・エッジケース (切断・不正データ・シーク連打) | 統合リハーサル通過 |
| 10 | ★統合点③ + Windowsビルド作成 + デモ撮影運営 | 発表PCで動くexe |

## 5. 進行役の仕事
- 週次MTG司会 (30-60分): 先週/今週/ブロッカー。**ブロッカーは48時間ルール**
  (2日止まったら即再割当て)
- スキーマ変更窓口: 要望を取りまとめ → Aに提案 → 承認後に全員通知
- 統合週 (4/7/10) は自分の新規開発を止めて統合に専念する

## 6. つまずき対策 (先回り)
- LFS未installでcloneした人が出る → CONTRIBUTING.md冒頭に太字、MTGで確認
- `.meta` ファイル差分が荒れる → Force Text設定 + 「Unityを閉じてからpull」を周知
- JSONのフィールド名ミスマッチ → C#クラスのフィールド名はJSONと**完全一致**
  (last_move のようにスネークケースのまま。リネームしない)
- WebSocketが繋がらない → まずブラウザやwscatで `ws://localhost:8765` を叩いて
  サーバ側の生死を切り分けてからUnityを疑う
