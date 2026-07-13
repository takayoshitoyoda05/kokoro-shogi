# Claude Code プロンプト: kokoro-shogi 初期共有リポジトリ生成 (実装ゼロ版)

用途: チームに最初に共有する「おおもとリポジトリ」を作る。
コードの実装は一切含まず、**構造 + 説明コメントだけのファイル群**を生成する。
空リポジトリのルートで `claude` を起動し、以下を貼り付ける。

なお docs 類 (README.md, DESIGN.md, TEAM_PLAN.md, INTERFACE.md,
REPO_STRUCTURE.md, 個人計画書5通) は Taka が別途手元のファイルを配置するので、
Claude Code には作らせない (プレースホルダも不要)。

---

## プロンプト本文 (ここから下をコピー)

チーム5人で開発する「kokoro-shogi」の初期リポジトリを作ってほしい。
これはキックオフでチームに共有する骨組みで、**プログラムの実装は1行も
含めてはならない**。まず作成する全ファイルの一覧を提示して私の承認を得てから
作業を開始すること。

### 絶対ルール (最優先)

1. **実装コード禁止**: 関数定義・クラス定義・変数代入・import文を含む
   Pythonコードを書いてはならない。例外は `__init__.py` (完全に空) のみ。
2. 各 `.py` ファイルの中身は**モジュールdocstring1つだけ**とする。
   docstringには次を日本語で書く:
   - このファイルが将来担うもの (1-3行)
   - 実装予定のクラス/関数の名前と役割の箇条書き (シグネチャは書かない)
   - 参照すべき設計文書の場所 (例: 「詳細は DESIGN.md §3(3) 欲求ヘッド」)
   - 実装予定週 (例: 「実装: 週4 / 担当: A」)
3. yaml / md / .gitignore / .gitattributes は「設定・文書」なので完全に
   書いてよい。ただし yaml は値の意味をコメントで説明すること。
4. テストファイルも同様に docstring のみ (「ここで検証する予定の性質」を
   箇条書き)。pytest の import も書かない。
5. CI・ビルドスクリプトは作らない (実装が無いのでCIは実装開始時に足す)。
6. **どんなに自明でも「ついでに」コードを書かない。** 迷ったらコメントにする。

### プロジェクト文脈 (docstringの中身の材料)

駒エージェント将棋AI。各駒に性格 (θ)・欲求 (6軸: 生存/攻撃/成り/守備/前進/
再登場)・感情 (GRU状態)・関係性を持たせ、手スコア = ⟨性格, 欲求⟩ + 自由項 を
全駒softmaxで調停して指す。学習は floodgate 棋譜からの蒸留 → 5五将棋での
MAPPO自己対戦RL。功績配分 (COMA) で駒のキャリア・血統・文化リーグへ発展。
会議ログをローカルLLM (Ollama) で実況。Unity側とは JSONL / WebSocket
(docs/INTERFACE.md, schema 1.0) でのみ通信。10週開発、担当は
A=AI全部, U1=Unity連携, U2=Unity本体, B1=Blender駒, B2=Blender舞台。

### 作成するツリー (これ以外を作らない)

```
kokoro-shogi/
├── .gitignore              # Python + Unity + Blender向け (下記仕様)
├── .gitattributes          # Git LFS: *.fbx *.png *.psd *.wav *.mp3 *.blend
├── pyproject.toml          # プロジェクト名・Python 3.12・依存の予定リストは
│                           #   コメントで記載 (dependencies自体は空でよい)
├── docs/
│   ├── plans/.gitkeep      # 個人計画書はTakaが配置
│   └── decisions/.gitkeep  # ADR置き場
├── src/kokoro_shogi/
│   ├── __init__.py         # 空
│   ├── core/
│   │   ├── __init__.py
│   │   ├── piece_state.py  # 予定: PieceState (恒久ID/性格/感情/忠誠/関係/キャリア/血統)
│   │   ├── tokenizer.py    # 予定: cshogi盤面→駒トークン特徴列 (DESIGN.md §3(1))
│   │   └── effects.py      # 予定: 利き関係の抽出→attentionバイアス素材 (§3(2))
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trunk.py        # 予定: 共有Transformer+利き/関係バイアス (§3(2))
│   │   ├── heads.py        # 予定: 欲求/性格重み/自由項/critic/単調mixing (§3(3)-(8))
│   │   ├── mood.py         # 予定: 感情GRU+イベント特徴 (§3(9))
│   │   ├── council.py      # 予定: 会議調停ループ (§3(6))
│   │   └── policy.py       # 予定: 全体を束ねるKokoroPolicy
│   ├── train/
│   │   ├── __init__.py
│   │   ├── distill.py      # 予定: 蒸留 (週3) / selfplay_ppo.py: 5五将棋MAPPO (週8)
│   │   ├── selfplay_ppo.py
│   │   └── league.py       # 予定: 文化リーグPBT (週9)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── floodgate.py    # 予定: CSA棋譜パース (週2)
│   │   └── labels.py       # 予定: 欲求ラベル事前計算 (週2)
│   ├── persist/
│   │   ├── __init__.py
│   │   └── store.py        # 予定: SQLite (θ_ind/career/lineage) (週7)
│   ├── logging/
│   │   ├── __init__.py
│   │   └── jsonl.py        # 予定: INTERFACE.md準拠のstate_update書き出し
│   ├── viz/
│   │   ├── __init__.py
│   │   └── narrator.py     # 予定: Template/Ollama実況 (週6)
│   └── server/
│       ├── __init__.py
│       └── server.py       # 予定: WebSocketサーバ ws://localhost:8765 (週5)
├── scripts/
│   ├── download_floodgate.py  # 予定: 棋譜取得 (週2)
│   ├── make_labels.py         # 予定: ラベル生成 (週2)
│   └── gen_sample_jsonl.py    # 予定: ダミー対局10局のJSONL生成 (週1・最優先)
├── configs/
│   ├── base.yaml           # モデル/損失ハイパラ (全キーにコメント)
│   └── features.yaml       # 機能フラグ: mood/relations/council/individual/
│                           #   loyalty/league すべて false (各フラグにコメント)
├── tests/
│   ├── test_tokenizer.py   # 検証予定: 初期局面40トークン/SFEN往復
│   ├── test_policy.py      # 検証予定: Σπ=1/非合法手0/合法手1つなら≈1
│   ├── test_heads.py       # 検証予定: w_i>0 / α_i≥0 (単調性)
│   ├── test_features.py    # 検証予定: フラグ全falseでもforwardが通る
│   └── test_jsonl.py       # 検証予定: 出力がschema 1.0に準拠
├── sample_data/.gitkeep    # 週1にgen_sample_jsonl.pyの出力を置く
├── blender/.gitkeep        # B1/B2の.blend原本置き場
└── unity/
    ├── README.md           # 「ここにU1がUnity Hubで KokoroShogi プロジェクト
    │                       #   (6000.3.8f1 LTS / Universal 3D=URP) を作成する。
    │                       #   clone前に git lfs install 必須」の案内のみ
    └── .gitkeep
```

### .gitignore の内容 (完全に書いてよい)

Python: `__pycache__/ .venv/ *.egg-info/ .ruff_cache/`
データ系: `data/ checkpoints/ runs/ *.db *.pth *.pt .env`
Unity (unity/KokoroShogi/ 配下): `Library/ Temp/ Logs/ UserSettings/ obj/ Build/`
Blender: `*.blend1 *.blend2`
OS: `.DS_Store Thumbs.db`

### configs の内容 (完全に書いてよい・全キーにコメント)

base.yaml: d_model 256 / n_layers 6 / n_heads 8 / d_theta 16 / d_mood 32 /
lambda_g 0.01 / tau 1.0 / c1 1.0 / c2 0.5 / c3 0.01 / seed 42
features.yaml: mood, relations, council, individual, loyalty, league = false
(コメント例: `council: false  # 会議調停ループ [D] 週6に実装後trueへ`)

### 完了条件

- ツリーが上記と完全一致 (過不足なし)
- `grep -rn "def \|class \|import " src/ scripts/ tests/` が**1件もヒットしない**
- 全 .py にモジュールdocstringがあり、担当週・参照文書が書かれている
- git init → 初回コミット `chore: initial skeleton (no implementation)` まで実施

## (コピーここまで)

---

## 使い方メモ (Taka向け、プロンプトには含めない)

1. GitHub でリポジトリ `kokoro-shogi` を作成 (README無しの空で)
2. ローカルで clone → `claude` 起動 → 上記を貼付 → ファイル一覧を承認
3. 完了後、手元の文書を配置:
   ルートに README.md / DESIGN.md / TEAM_PLAN.md、
   docs/ に INTERFACE.md / REPO_STRUCTURE.md、docs/plans/ に計画書5通
4. レビュー: 完了条件の grep を自分でも実行 (「気を利かせた実装」の検出)
5. push → チームに共有 → キックオフ
6. 週1の実作業はここから: 「scripts/gen_sample_jsonl.py の docstring に
   従って実装して。出力は docs/INTERFACE.md の schema 1.0 に準拠」と
   Claude Code に依頼するところが実装フェーズの開始点
