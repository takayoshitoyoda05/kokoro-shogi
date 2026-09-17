# Kokoro-Shogi (こころ将棋)

**駒が心を持つ将棋AI** — 各駒が性格・感情・関係性を持ち、議論して一手を決め、
その内面をUnityの3D盤面で観戦できるプロジェクト。

> 歩が怯えて青ざめ、金と玉が絆の光で結ばれ、駒たちの会議が吹き出しで交わされ、
> AIの実況が流れる。それでいて、中身は本気で強さを目指した深層学習将棋エンジン。

---

## はじめての人へ (3ステップ)

1. この README を最後まで読む (5分)
2. [`TEAM_PLAN.md`](./TEAM_PLAN.md) で全体の流れと自分の役割を確認する
3. `docs/plans/` の**自分の計画書**を開く — 週1のタスクと環境構築手順が
   全部書いてある。困ったらまず自分の計画書の「困ったときは」節へ

## これは何?

従来の将棋AI (AlphaZero系) は盤面全体を1つの評価関数で見ます。
Kokoro-Shogiは発想を変え、**盤上の40枚の駒それぞれをエージェントとして扱います**。

- 各駒は **欲求** (生存・攻撃・成り・守備…) と **性格** (どの欲求を重視するか) を持つ
- 各駒は対局中の出来事で変化する **感情** を持つ (仲間が取られると怯える)
- 駒同士は **関係性** を築く (何局も共闘した金と玉の絆)
- 一手は駒たちの **会議** (bid合戦) で決まり、その議事録がLLMで実況される
- 取られた駒は相手の持ち駒として **転生** する — 将棋にしかない、エージェントの
  所有権が敵に移る現象
- 対局を重ねると駒に **キャリア** (生存率・MVP) が蓄積し、優秀な駒の性格は
  **血統** として次世代に受け継がれる

これらの「内面」はすべて補助構造として設計されており、**外しても強い将棋AIが
残る**よう、強さの経路 (Transformer + 蒸留 + 自己対戦RL) と分離されています。

## 技術ハイライト

> この表はAI側の技術要約です (対外説明・発表用)。Unity/Blender担当は
> 読み飛ばしてOK — 作業に必要な知識は各自の計画書にすべて書いてあります。

| 領域 | 中身 |
|---|---|
| アーキテクチャ | 駒トークン化Transformer (Chessformer/Leela流) + 利き関係attentionバイアス |
| 方策 | 手スコア = ⟨性格, 欲求⟩ + 自由項 → 全駒softmax (微分可能な「調停」) |
| 学習 | floodgate棋譜からの蒸留 → MAPPO / Gumbel AlphaZero自己対戦RL |
| 功績配分 | QMIX流の単調mixing + COMA反実仮想advantageで勝敗を駒ごとに分解 |
| 解釈可能性 | 「棋力 vs 欲求説明率」をλ_g掃引で定量化。\|自由項\|が捨て駒検出器になる |
| 進化 | 個体性格の永続化・血統 (進化戦略)・文化リーグ (PBT) |
| 実況 | 会議ログ → ローカルLLM (Ollama + Qwen2.5 7B) で自然言語実況。コストゼロ |
| 可視化 | Unity (URP): 感情シェーダ・関係線VFX・会議吹き出し・リプレイ/ライブ観戦/対局 |

新規性の核は (1) ターン制完全情報ゲームでの駒粒度マルチエージェント学習、
(2) 持ち駒 = エージェント転生問題、(3) 解釈可能性と棋力のトレードオフの構造化。
詳細は [`DESIGN.md`](./DESIGN.md) を参照。

## システム構成

```
┌────── src/kokoro_shogi (Python) ──────────┐   ┌── unity/KokoroShogi (C#) ───┐
│                                            │   │                              │
│  学習パイプライン                           │   │  3D盤面・駒モデル             │
│   蒸留 → 自己対戦RL → 進化リーグ            │   │  感情シェーダ・関係線VFX      │
│           │                                │   │  会議吹き出し・実況字幕       │
│           ▼                                │   │  リプレイ/ライブ/対局モード   │
│  推論エンジン ──► JSONLログ ────────────────┼──►│  (JSONLリプレイ再生)         │
│           │                                │   │                              │
│           └────► WebSocketサーバ ◄─────────┼──►│  (リアルタイム観戦・対局)     │
│                                            │   │                              │
│  ローカルLLM実況 (Ollama)                   │   │                              │
└────────────────────────────────────────────┘   └──────────────────────────────┘
                     両者の契約 = docs/INTERFACE.md (JSONスキーマ)
```

AI側とUnity側は **JSONログとWebSocketでしか会話しない** 疎結合設計。
片方の遅延がもう片方を止めません。同様に、BlenderとUnityは
**FBXと命名契約 (naming.md) でしか会話しない**。

## リポジトリ

**モノレポ1つ** (`kokoro-shogi`): Python (A管理)・Unity (unity/ 配下, U1管理)・
Blender素材 (blender/, B1/B2管理) が同居。
構成の詳細・ブランチ運用・Git LFS: [`docs/REPO_STRUCTURE.md`](./docs/REPO_STRUCTURE.md)

## ドキュメント一覧

| 文書 | 内容 | 対象 |
|---|---|---|
| [`DESIGN.md`](./DESIGN.md) | AIアルゴリズム設計書 v2 (数式仕様・出典・学習フロー) | A |
| [`TEAM_PLAN.md`](./TEAM_PLAN.md) | チーム全体計画 (v4: 役割・10週マイルストーン・運用ルール) | 全員 |
| [`docs/INTERFACE.md`](./docs/INTERFACE.md) | データ契約の正本 (JSONスキーマ, バージョン管理) | 全員 |
| [`docs/REPO_STRUCTURE.md`](./docs/REPO_STRUCTURE.md) | リポジトリ構成 (フォルダの意味・置いてよいもの・担当区分) | 全員 |
| [`docs/GIT_GUIDE.md`](./docs/GIT_GUIDE.md) | Gitの操作手順 (初学者向け: 環境構築・毎日の手順・トラブル対処) | 全員 |
| [`docs/plans/PLAN_A_algorithm.md`](./docs/plans/PLAN_A_algorithm.md) | 個人計画書: AIアルゴリズム | Taka |
| [`docs/plans/PLAN_U1_lead_network.md`](./docs/plans/PLAN_U1_lead_network.md) | 個人計画書: Unity-Python連携 (初学者向け) | U1 |
| [`docs/plans/PLAN_U2_unity_main.md`](./docs/plans/PLAN_U2_unity_main.md) | 個人計画書: Unity本体 (初学者向け) | U2 |
| [`docs/plans/PLAN_B1_blender_pieces.md`](./docs/plans/PLAN_B1_blender_pieces.md) | 個人計画書: Blender 駒 (初学者向け) | B1 |
| [`docs/plans/PLAN_B2_blender_stage.md`](./docs/plans/PLAN_B2_blender_stage.md) | 個人計画書: Blender 舞台 (初学者向け) | B2 |

## チーム

| 役割 | 担当 | 一言 |
|---|---|---|
| A: AIアルゴリズム | Taka | AIエンジン全体・スキーマ策定・データ納品 |
| U1: Unity-Python連携 | メンバー1 | リプレイ/WebSocket・盤面ロジック・進行役 |
| U2: Unity本体 | メンバー2 | シーン・演出 (シェーダ/VFX/アニメ)・UI・音 = 画面のすべて |
| B1: Blender (駒) | メンバー3 | 感情の仕掛けを埋め込んだ駒14種 |
| B2: Blender (舞台) | メンバー4 | 盤・駒台・環境・ライティング素材 = 世界のすべて |

## ロードマップ (10週・全機能 + 継続)

```
週 1     キックオフ: スキーマ策定・サンプルJSONL納品・アート方向性決定
週 2-3   AI: 前処理→蒸留 ┃ Unity: リプレイヤー ┃ Blender: 先行駒・盤
週 4     ★統合点①: サンプルログがUnityで完全再生
週 5-6   AI: 感情/関係→会議+LLM実況→実データ納品 ┃ Unity: WebSocket・VFX
週 7     ★統合点②: ライブ観戦 ┃ AI: 功績配分・キャリア・SQLite
週 8     AI: 自己対戦RL+忠誠 (5五将棋) ┃ Unity: 人間対局モード・キャリアUI着手
週 9     AI: 文化リーグ (5五将棋) ┃ Unity: キャリアUI完成・最適化
週 10    ★統合点③: フルデモ → デモ動画・発表資料
以降     RL/リーグ/血統の本将棋フルスケール化・floodgate参戦 (AI側継続)
```

重い機能 (RL・文化リーグ・忠誠実験) は10週内は **5五将棋 + 小型モデル** で
「動く・兆候が見える」まで実証し、本将棋フルスケールは継続タスクとする戦略。

## セットアップ

### 全員共通 (最初に1回)
```bash
git lfs install              # ★cloneより先に必ず1回 (忘れると3Dモデルが開けない)
git clone <this-repo>
```

### AI側 (担当: A)
```bash
# uv未導入なら: https://docs.astral.sh/uv/ (Windows: winget install astral-sh.uv)
cd kokoro-shogi
uv sync                      # Python 3.12 + 依存一式
uv run pytest                # ※初期リポジトリは全テストskipでグリーンになれば正常
                             #   (テストは各Phaseの実装と同時に有効化していく)
```

#### 学習済みモデル
対局サーバが読むモデルは通常 `checkpoints/` ごと Git 管理外ですが、**共有用の 2 つだけは
リポジトリに入っています**。clone すればそのまま対局できます。

| ファイル | 中身 |
|---|---|
| `checkpoints/league_E2b_grace/league.pt` | **デモはこちら**。ppo2 相手に文化平均 0.633 で最も強い |
| `checkpoints/league_E7_ema/league.pt` | 最新 (2026-09-12)。適応度 EMA の実験条件で、強さは 0.540 |

```bash
uv sync --group train        # torch が要る (既定の sync からは外してある)
uv run python scripts/play_server.py \
  --checkpoint checkpoints/league_E2b_grace/league.pt --host 0.0.0.0
```
起動時に `mood: 感情GRU / relations: r_ij状態 / council: ON` と出れば正常です。
Unity 側は **Python を起動してから** Play してください (接続は Unity からの一方向で、
Unity 側が Python を起動することはできません)。詳細は `docs/INTERFACE.md` §4。

### Unity側 (担当: U1/U2)
```
Unity Hub → Installs → 6000.3.8f1 (LTS) を追加 (バージョン固定・他は使わない)
Unity Hub → unity/KokoroShogi/ を開く
```
- ※ `unity/KokoroShogi/` プロジェクト本体は**週1にU1が作成**します。
  それまでは unity/README.md の案内だけが置いてあります
- ※ `Replay.unity` でのサンプル再生確認は**週2以降** (U1のリプレイヤー完成後)
  の手順です。それより前は何も動かないのが正常です

### Blender側 (担当: B1/B2)
Blender 4.x を blender.org からインストール → 各自の計画書 §1 へ。

うまくいかないときは Issue へ。**READMEの手順通りで動かなければ、それは
READMEのバグです** (あなたのせいではありません)。

## ライセンス / クレジット

- 棋譜データ: floodgate (wdoor) — 利用条件に従う
- 実況LLM: Qwen2.5 (Apache 2.0) via Ollama
- 効果音・BGM・フォント: `unity/KokoroShogi/Assets/_Project/Audio/LICENSES.md` に一覧 (U2管理)
- テクスチャ素材: `unity/KokoroShogi/Assets/_Project/Textures/LICENSES_textures.md` に一覧 (B2管理)
- 本体ライセンス: チームで決定後に記載 (公開するなら MIT 推奨)
