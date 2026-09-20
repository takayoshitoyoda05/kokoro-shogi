# 記録: 木を作らずに強くする方向の最新研究サーベイ (2022-2026)

- 日付: 2026-09-19
- 記録者: Taka (A)
- 関連: [`2026-09-19-strength-roadmap-after-presentation.md`](./2026-09-19-strength-roadmap-after-presentation.md) /
  [`2026-09-11-adaptive-council-rounds.md`](./2026-09-11-adaptive-council-rounds.md) /
  [`2026-08-31-ppo-entropy-c3.md`](./2026-08-31-ppo-entropy-c3.md) / `DESIGN.md` §3(6) §4b §4c
- 状態: 調査のみ。着手判断は未定 (E4 完走後に 1 番から試す想定)
- **2026-09-19 決定 (Taka)**: 「木を作らない」は**推論時**の制約。**学習・評価時の外部エンジン利用は可** (費用ゼロ・CPU で完結・GPU と並走可を確認のうえ)。
  → 系統 2 (エンジン教師蒸留: value 教師・multipv soft target)、L0-b 固定外部基準、L7-a |g| 定量化が解禁
- **2026-09-19 決定 (Taka)**: 目的関数は**強さ**。解釈可能性 (説明率・|g|) は制約にせず、常に並走して記録する指標とする (L0-c の 2 軸ダッシュボード)
- **2026-09-19 決定 (Taka)**: Phase E (駒の内面: θ_ind 更新則・血統・感情の教師・敵対関係) は **Phase D (自己対戦の安定化) の後**に着手する。順序は A 土台 → B 会議ループ → C 学習信号 → D 自己対戦 → E 内面 → F アーキテクチャ
- 実装メモ: 現行 やねうら王 (2026-09-14 時点の main) には **gensfen/学習部が無い** (ソース参照ゼロ)。教師局面は USI (`go depth/nodes` + MultiPV) を自作ドライバで叩いて作る。最新の水匠評価関数はスポンサー限定なので、無償公開の評価関数 (水匠5 / tanuki 系) を使う

## 前提の変更

ユーザー方針: **今後は木 (探索) を作らない方向で強くする**。
→ 2026-09-19 の強さロードマップで «1位: 推論時探索 (Gumbel AlphaZero)» としていた項目は**取り下げ**。
代わりに本サーベイの系統 1 (ループ・再帰深度) を最有力とする。

## 結論

**会議ループの深化が本命。** `policy.py:_council` が trunk 最終2層を重み共有で再適用する構造は、
2025-2026 に整備された **looped transformer / recurrent depth** (潜在空間での test-time compute
スケーリング) そのもの。木を作らずに推論時計算量を増やす唯一の確立した方法がこれ。

観測済みの現象が、この分野の中心問題と一致している:

| 本プロジェクトの観測 (ADR 2026-09-11) | 分野での呼び名 |
|---|---|
| R=2 で 46.16% がピーク、R=3 43.74%、R=4 40.26% に劣化 | test-time scaling の不安定性 (ピーク後の崩壊) |
| オラクル上限 +13.8pt、手作り停止則は全敗 | 学習型 halting の必要性 |
| 会議は収束しない (R=3→4 でも 22% が変化) | 潜在動力学が漸近安定固定点に収束していない |

## 系統 1: ループ・再帰深度 ★最有力

- **Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"**
  (arXiv:2502.05171, NeurIPS 2025)。再帰ブロック反復でテスト時に任意深度へ展開。3.5B/800B トークン (Huginn)。
  **「前段ブロック → 4ブロックの反復スタック → 最終ブロック」の3部構成**を採る (UT の単純な同一ブロック再利用ではない)。
  CoT と違い特別な訓練データも大きな文脈窓も不要、言葉にしにくい推論を捉えられる。
  → 本実装の `reapply_last_layers(depth=2)` の構造見直しの根拠。
- **Yang et al., "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped
  Language Models"** (arXiv:2605.26733, ICML 2026)。**本サーベイ最大の収穫**。
  「ある反復深度でピーク後に性能が崩壊する」問題を正面から扱い、安定性と有効性のトレードオフが
  既存アーキテクチャに内在すると特定。**STARS (STAbility-driven Recurrent Scaling)** = 推論を不確実性削減と捉え、
  潜在状態を漸近安定な固定点に収束させる訓練枠組み。実装は
  **Jacobian Spectral Radius Regularization + random loop sampling**。
  → **R=3,4 の劣化はこの論文が扱う現象そのもの**。random loop sampling (学習時に R を 1〜4 から
  ランダムに引く) は数十行で入る。
- **Banino et al., "PonderNet: Learning to Ponder"** (arXiv:2107.05407)。停止を幾何分布事前を持つ
  潜在変数モデルとして定式化、KL 正則化。「まだ止まっていない条件下で今止まる確率」を予測。
  **完全微分可能** (REINFORCE より低分散、ACT と違い不偏)。
  → 停止ヘッドの正解。手作り閾値が全敗したのは停止を後付けにしたから。教師データ
  (7,846 局面 × R=0..4) は `eval_adaptive_council.py` が既に保持。
- ⚠ 未確認: "Loop, Think, & Generalize" (Kohli et al., 2026、推論時反復増で深さ外挿)、
  "AdaPonderLM: Gated Pondering LM with Token-Wise Adaptive Depth" (arXiv:2603.01914、
  トークンごとに深さを変える = 駒ごとに会議ラウンド数を変える発想に対応)

## 系統 2: 探索を重みに畳み込む (蒸留)

| 論文 | 要点 | 適用 |
|---|---|---|
| Ruoss et al. (arXiv:2402.04494) | Stockfish 16 の行動価値で 1,000万局→150億データ点、270M Transformer、Lichess 2895 | 教師局面生成の手本 |
| Amortized Planning (NeurIPS 2024) | 探索は償却できるが**差は完全には埋まらない** | 期待値設定 |
| Stop Regressing (arXiv:2403.03950, ICML 2024) | 価値を回帰でなく**分類** (HL-Gauss) で学習。検証ドメインに "chess without search" を含む | `c1·(V−z)²` → HL-Gauss、数十行 |
| Anthony et al., "Policy Gradient Search" (arXiv:1904.03646) | ExIt の発展形、**明示的な探索木なしのオンライン計画** | 「木なし計画」の原典 |

## 系統 3: 自己対戦の循環を止める

観測済みの **t25→t29→t33→t25 のジャンケン (非推移的循環、2026-09-03)** に対応する系統。

- **R-NaD** (Perolat et al., Science 2022 / arXiv:2206.15378)。均衡周りの「サイクル」を学習ダイナミクス
  自体を変えて収束させる。探索なしで Stratego 人間エキスパート級
- **Magnetic Mirror Descent (MMD) / NeuRD**。マグネットへの近接正則化で循環を減衰。IMMD は前ラウンドの
  収束戦略を次の正則化中心に使う
- ⚠ "Reevaluating Policy Gradient Methods for Imperfect-Information Games" (ICLR 2026)。PPO 自己対戦に
  一様マグネット正則化、線形アニーリング (Rudolph+ 2026) と冪則 (Sokota+ 2025) を比較
- ⚠ **EMAgnet** (arXiv:2606.23995)。パラメータ空間の EMA をマグネットに使う。
  → **重み平均 (soup) の正しい使い方の可能性**。2026-09-19 の実験で事後 soup は失敗した
  (t35-39 で 0.225 に崩壊) が、EMAgnet は学習中の正則化中心として使う真逆の発想

## 系統 4: 駒ごとの功績配分 (COMA/QMIX の後継)

`DESIGN.md` §4d の COMA と `MonotonicValueMixing` の QMIX 単調性は 2017-2018 の手法。以下いずれも ⚠ 未確認:

- "Multi-level Advantage Credit Assignment for Cooperative MARL" (arXiv:2508.06836)
- "COSAC: Counterfactual Credit Assignment in Sequential Cooperative Teams" (arXiv:2604.17693)
  — **逐次的**協調チーム。将棋は手番が逐次なので設定が近い
- "Cooperative Game-Theoretic Credit Assignment via the Core" (arXiv:2506.04265)、nucleolus 版 (AAMAS 2025)
- "Assigning Credit with Partial Reward Decoupling in MAPPO" (arXiv:2408.04295)
  — **MAPPO が全エージェントで同じ global advantage を共有するのは準最適**という指摘。
  `selfplay_ppo.py` の現状そのもの

→ 強さより「駒の内面 (MVP・θ_ind) の質」に効く系統。テーマとの相性は最高。

## 系統 5: アーキテクチャ

- **Chessformer** (arXiv:2605.19091, 2026-05)。Daniel Monroe (LC0 チーム) + Zhenwei Tang・Ashton Anderson
  (Maia-2 の CSSLab)。エンコーダのみ Transformer、マスをトークン化、**Geometric Attention Bias (GAB)**、
  source-destination 方策ヘッド。人間着手予測 57.1%、**LC0 統合で +100 Elo 超**。
  **強さ・人間予測・解釈可能性の3点同時**という問題設定が本プロジェクトと同一 = 最良の参照点かつ競合。
  前身は "Mastering chess with a transformer model" (arXiv:2409.12272)。査読状況は未確認 (プレプリント)
- ⚠ AlphaViT (arXiv:2408.13871)。複数ゲーム・可変盤サイズ。本将棋と5五将棋を1モデルで扱う話に関係しうる
- 解釈可能性の対比相手: "Tracing the Thought of a Grandmaster-level Chess-Playing Transformer"
  (arXiv:2604.10158, 2026-04)。LC0 を疎な置換層で分解し MLP と attention の両面から解釈。
  → 本プロジェクトの |g| は「構造に埋め込む」、こちらは「事後に解剖する」。発表での対比に使える
- 人間らしさ: Maia-2 (NeurIPS 2024, arXiv:2409.20553)。skill-aware attention、perplexity 4.67→4.07 bits。
  後継 Maia-3 あり (⚠ 未確認)。Matilda (arXiv:2606.25176, 2026-06) が Maia-3 + Stockfish の残差リランキング

## 参考: 探索なしモデルフリー RL の到達点 (既知の上限の目安)

- DeepNash (Science 2022): 探索なしで Stratego 人間エキスパート級
- GT Sophy (Nature 2022): QR-SAC (SAC の期待値を分布に置換)、モデルフリーで世界トップドライバーに勝利
- AlphaStar / OpenAI Five も木探索なし (2019、知見は出尽くし)

## 空白

**将棋における「探索なし」の学術研究が見当たらない。** 日英で検索したが、出てくるのは dlshogi の実装解説・
WCSC35 アピール文書・コンピュータ将棋 Wiki といった実務/競技資料で、査読論文の形の searchless shogi は
発見できなかった (⚠ 検索範囲での話)。チェスで2024年に確立した路線が将棋では埋まっていない
= 本プロジェクトが主張できる位置。

## 【2026-09-19 夜 追記】順序を修正した

外部基準の実測と構造分析 (trunk 94.68% / 内面 0.09%) を受けて、**Phase F (アーキテクチャ) を最後尾から前倒し**し、
**Phase A (測定) を完成させてから次に進む**ことにした。詳細と理由は
[`2026-09-19-plan-revision.md`](./2026-09-19-plan-revision.md)。下の優先順位表は「木を作らない」観点での
素の順位で、そこに上記 2 点が被さる。

## 着手の優先順位 (木を作らない前提)

| 順位 | やること | 根拠 | コスト | 期待 |
|---|---|---|---|---|
| **1** | **会議の random loop sampling** (学習時に R を 1〜4 からランダムに引く) | STARS | **数十行** | R=3,4 の劣化を消し、推論時に R を増やせるようにする |
| 2 | **PonderNet 型の停止ヘッド** | PonderNet | 中 (教師データは既存) | オラクル上限 +13.8pt |
| 3 | **value を HL-Gauss (分類) に** | Stop Regressing | 数十行 | 価値の質 → advantage の質 |
| 4 | Jacobian スペクトル半径正則化 | STARS | 中 | ループ安定化 (1 で足りなければ) |
| 5 | 自己対戦にマグネット正則化 | MMD / EMAgnet | 中 | 循環モードの抑制 |
| 6 | 会議ループを3部構成に見直し | Geiping et al. | 大 (再学習) | 深さスケーリングの素地 |
| 7 | COMA を後継手法に差し替え | COSAC 等 | 大 | 内面の質 (強さではない) |

**1 が突出**: 数十行で、測定済みの弱点を直接叩き、成功すれば「会議を深くするほど強くなる」という
発表で最も映える性質が手に入る。E4 完走後の最初の一手にする。

## 確認の状況 (再現用)

- 本文まで確認: STARS / Chessformer / Tracing the Thought / How Reasoning Evolves (arXiv:2604.05134, ICML 2026) / Matilda
- 一次情報源 (会議録・公式ページ) で確認: Geiping / PonderNet / Ruoss / Stop Regressing / Amortized Planning /
  DeepNash / GT Sophy / Maia-2
- **タイトルと arXiv ID のみ (本文未確認、実装前に要読)**: COSAC / EMAgnet / AdaPonderLM / Multi-level Advantage /
  Core-based credit / PRD-MAPPO / AlphaViT / ICLR 2026 Reevaluating / Loop Think & Generalize / Maia-3
- 2026年7〜9月 (arXiv 2607〜2609) の新着は検索に出てこなかった (⚠ 拾えていないだけの可能性)

## 実施記録: 外部エンジン環境 (2026-09-19)

- **エンジン**: やねうら王 main (commit c1b80ea, 2026-09-14) を `data/engines/YaneuraOu` に clone し、
  `make normal COMPILER=g++ TARGET_CPU=AVX2 YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE` でビルド
  (clang 無し、AVX512 無しのため)。バイナリは `data/engines/bin/YaneuraOu-by-gcc` (id: YaneuraOu NNUE 9.80git 64AVX2)
- **評価関数**: tanuki- 系 **Háo** (halfkp_256x2-32-32、64MB、**GPLv3**、配布物に gpl-3.0.txt 同梱)。
  `data/engines/eval_hao/eval/nn.bin`。配布ページ推奨の **FV_SCALE=20** で使う (既定 16 ではない)。
  最新の水匠はスポンサー限定 (有料) のため不採用。水匠5 は無償公開最終版だが今回は Háo で開始
- **gensfen は現行版に無い** → `scripts/engine_teacher.py` (USI を直接叩く MultiPV 収集ドライバ、途中再開可)
- **スループット実測** (i7-12700KF 20 スレッド、16 ワーカー × Threads=1、E4 学習と並走中):
  depth 10 / MultiPV 4 で **113 局面/秒** (中央値 64 ms、5.5 万ノード/局面)。
  → 2M 局面 ≈ 5 時間、既存蒸留データと同規模の 22M ≈ 2.3 日。GPU 使用ゼロ
- 出力形式: 1 行 1 JSON `{sfen, depth, best, pv:[{move,cp,mate,depth}...] (MultiPV 順), nodes, time_ms}`。
  **cp は手番側視点** (USI 慣例)
- 次: floodgate CSA (data/floodgate/csa、360,868 局) から局面リストを作る → depth 10 でラベル付け →
  蒸留側に value 教師 (HL-Gauss) と multipv soft target を実装

## 実施記録: 絶対的な物差しの初回測定 (2026-09-19 12:02、`scripts/eval_vs_engine.py`)

モデル側 = 最強構成 (league_E2b_grace culture1、argmax、1手詰 ON)。相手 = やねうら王 + Háo、Threads 1、定跡なし。
先後交互、各 40 局、最大 256 手。

| 相手 | 勝率 | 手数 (min/med/max) | 終局 |
|---|---|---|---|
| Háo depth 1 | **0.075** | 67 / 78 / 98 | 全局詰み |
| Háo depth 2 | 0.100 | 35 / 83 / 144 | 全局詰み |
| Háo depth 3 | 0.200 | 45 / 79 / 125 | 全局詰み |

2 局を手で再生して手順が正常なことを確認 (バグではない)。深さと勝率の逆相関は 40 局のノイズ範囲。
**結論: 対 ppo2 で 0.72〜0.83 の最強構成は、NNUE + 1 手読み相手にほぼ勝てない。** 自己相対の +450〜550 Elo は
絶対水準ではこの位置。以後は **「対 Háo depth 1 勝率 (40 局以上)」を標準の物差し**として固定する。
結果 JSON: `checkpoints/vs_engine_hao_d{1,2,3}.json`。

運用メモ: 教師ラベル付けを 12 ワーカーで回すと E4 (GPU 学習) の iter 時間が 25s→55s に倍増した
(CPU 餓死、2026-09-04 の教訓の再現)。8 ワーカー (65 局面/秒、残り ≈ 7.5h) に落として並走。
`pkill -f engine_teacher.py` は自分のシェルと監視まで巻き込むので `pgrep -f "[e]ngine_teacher"` 形式を使うこと。

## 実施記録: Phase C (学習信号) の実装 (2026-09-19 12:30)

- `scripts/ops/join_teacher.py`: 教師 JSONL をシャードへ **SFEN (盤面+手番+持ち駒) で結合**し、
  シャードと同名の `*.teacher.npz` (teacher_value / teacher_cp / teacher_actions (N,4) / teacher_cps (N,4)) を書く。
  シャード本体は無変更。cp→value は 2σ(cp/600)−1、mate は ±1。MultiPV の手は action index に写す
  (打ちは piece_id 最小トークン、`legal_move_mask` と同じ規則)。1 シャード 10 万局面 ≈ 6 秒
- `data/dataset.py`: 側面ファイルがあれば読み、無ければ NaN/-1 で埋める (`ShardDataset.teacher_shards`)。
  `Position` / `collate` に teacher_value / teacher_actions / teacher_cps を追加 (既定値付きなので既存コードは無変更で動く)
- `train/distill.py`: `value_target` (教師があれば `teacher_value_weight` で z と混合、無ければ z) と
  `teacher_soft_loss` ($-\sum_k q_k\log\pi(a_k)$、$q=\mathrm{softmax}(\mathrm{cp}_k/T)$、教師のある行だけで平均)。
  `compute_loss_tensors` に `c_soft * soft_loss` を追加。`Metrics.soft_loss`
- `config.py` / `configs/base.yaml`: `c_soft=1.0`、`teacher_temp=200` (cp)、`teacher_value_weight=1.0`
- テスト: `tests/test_teacher.py` 5 本 (cp→value の単調性、USI→action の合法性、側面ファイルの読み込みと既定値、
  value 教師の混合、soft loss が有限で教師なしなら 0)。全体 311 passed
- 未実施: value を HL-Gauss (分類) にするのは価値ヘッドの構造変更を伴うので、まず密な教師 + MSE で効果を見てから
- 注意: 蒸留の入口は 2 つある。`distill.py` (局面単位、mood なし) と `mood_distill.py` (系列、感情 GRU 込み)。
  現行の最強系列 (phase37→ppo2→E2b) は mood/relations/council 込みなので、教師蒸留も **mood_distill 側**で回す必要がある
  → `data/sequence.py` / `collate_sequences` / `mood_distill._STEP_KEYS` に教師列を通した (12:40)。詰め物は NaN/-1
  (0 を入れると偽の教師になる)。`tests/test_sequence.py` に系列経由の教師テストを追加
- **第1走を予約** (`checkpoints/ops/c_chain.sh`、ログ `checkpoints/c_chain.log`): 教師ラベル完了 (≈19:45) と E4 終了 (≈20:30) を
  待って、join (全シャード) → mood_distill 1 epoch (warm start ppo2、relations/council、amp、20k 局 ≈ 1.7h、lr 5e-5)
  → 対 Háo depth 1 で 40 局。比較基準として ppo2 自身の対 Háo depth 1 も測っておく (`vs_engine_hao_d1_ppo2.json`)

## 実施記録: 物差し一括測定 第1弾 (2026-09-19 12:22〜12:45、対 Háo depth 1、各 100 局、argmax + 1手詰)

| モデル | 勝率 | 平均手数 |
|---|---|---|
| phase36_best | 0.00 | 51 |
| phase37 | 0.01 | 115 |
| ppo2_t13 | 0.00 | 82 |
| ppo2_t25 | 0.03 | 102 |
| ppo2 | 0.48 | 67 |
| ppo2_tau01 | 0.51 | 67 |
| ppo2_nomate | 0.48 | 71 |
| E2b_culture0 | 0.02 | 84 |
| E2b_culture1 | 0.03 | 74 |
| E2b_culture2 | 0.02 | 84 |
| E2b_culture3 | 0.03 | 68 |
| E2b_culture4 | 0.03 | 103 |
| E2b_culture5 | 0.02 | 74 |
| E7_default | 0.50 | 87 |
| E2_default | 0.00 | 82 |
| E1_default | 0.00 | 75 |

**読み (仮説、第2弾で検証中):**
1. **PPO は外部相手にも効いた**が、立ち上がりは最後の区間に集中 (t13 0.00 → t25 0.03 → t39=ppo2 0.48)
2. **E2 系リーグ (E2 / E2b) は ppo2 特化で外部強さが崩壊** (全 6 文化 0.02〜0.03、文化差なし = 共有 trunk の問題)。
   自己相対 (対 ppo2 0.633) と外部 (0.03) が真逆 = カウンターフィット。E1 (乱択) も 0.00
3. **E7 (適応度 EMA、9/12) は 0.50 で ppo2 と同等**。E2b (9/7) との差が「EMA」か「9/11 以降のコード差」かは
   第2弾 (E6 9/11、A-drift 9/12、run2 9/4) で切り分ける
4. argmax / 1手詰 / τ=0.1 の差は外部相手では出ない (0.48〜0.51)。今日の「argmax +14pt」は対 ppo2 限定
5. **今日の play_server 既定 (E2b culture1) は外部基準では最弱クラス** → 第2弾の結果で既定を差し替える

結果 JSON: `checkpoints/yardstick/*.json`、スクリプト `checkpoints/ops/yardstick_batch.sh`

## 訂正: 第1弾・第2弾の物差しは標本が縮退していた (2026-09-19 13:00)

先後別・手数別に分解したところ、argmax 方策 × 決定論的エンジン (depth 1、Threads 1、定跡なし) で
**同じ手番の 50 局がほぼ同一棋譜** (ppo2 先手: 49/50 が 66 手) だった。100 局 = 実質 2 局。
さらに全モデルが先手で負け、ppo2/E7 は後手で 0.96〜0.98 という強い先後非対称 (エンジン同士 depth1 でも先手 4/4)。
→ 第1弾の「ppo2 0.48 / E7 0.50 / E2b 0.03」の**差は 2〜4 局分で無意味**。「E2b がカウンターフィットで崩壊」は撤回。
ADR 2026-08-31 追記3 の教訓 (決定論的方策の対局は標本が縮退する) をそのまま再現した。

対処: `scripts/ops/make_openings.py` で floodgate 実戦の序盤 (10〜24 手) 500 通りを切り出し、
`eval_vs_engine.py --openings` で局ごとに違う序盤から (先後で同じ序盤を 1 回ずつ)。
ppo2 6 局の予備測定は 1/6 で手数もすべて異なった (縮退解消)。第3弾 (`yardstick_batch3.sh`、12 候補 × 100 局) で測り直す。
**以後、物差しの数字には必ず「異なる棋譜の数」と先後別の勝率を添える。**
また評価スクリプトの torch を 2 スレッドに制限 (全コア使用で並走中の E4 が 25s→51s/iter に餓死していた)。

## 実施記録: 物差し第3弾 = 最初の信頼できる絶対基準 (2026-09-19 13:02〜13:15)

対 Háo depth 1、**序盤ブック付き** (floodgate 実戦 10〜24 手、局ごとに別、先後で同じ序盤を 1 回ずつ)、各 100 局、
argmax + 1手詰 (τ=0.1 の行を除く)。SE ≈ 0.04。

| モデル | 勝率 | 先手 / 後手 | 異なる手数 (/100) |
|---|---|---|---|
| ppo2 | **0.23** | 0.24 / 0.22 | 68 |
| phase37 | **0.22** | 0.18 / 0.26 | 65 |
| ppo2_tau01 | **0.16** | 0.12 / 0.20 | 64 |
| E7_default | **0.16** | 0.14 / 0.18 | 63 |
| E2_default | **0.15** | 0.20 / 0.10 | 66 |
| E6_default | **0.14** | 0.16 / 0.12 | 65 |
| E1_default | **0.14** | 0.16 / 0.12 | 63 |
| E2b_culture4 | **0.13** | 0.16 / 0.10 | 63 |
| ppo2_t25 | **0.12** | 0.06 / 0.17 | 59 |
| A_drift_default | **0.11** | 0.13 / 0.09 | 62 |
| E2b_culture1 | **0.10** | 0.04 / 0.16 | 65 |
| run2_default | **0.10** | 0.12 / 0.08 | 62 |

**結論:**
1. **PPO も文化リーグも、外部エンジン相手の強さを上げていない。** 蒸留だけの phase37 (0.22) と PPO 9,200 iter 後の ppo2 (0.23) が同じ。
   途中の t25 は 0.115 と一度下がっている。自己相対の +450〜550 Elo は自己対戦の内側でしか通用しなかった
2. **リーグ系は全部 0.10〜0.16 で、蒸留/ppo2 より下** (差は SE の 2 倍前後)。E2b が特別に悪いわけではない (第1弾の「崩壊」は撤回済み)
3. argmax は τ=0.1 より +7pt (0.23 vs 0.16)。1手詰チェックの効果は第1弾でも第3弾でも出ていない (害もない)
4. 先後差は消えた (0.24/0.22)。第1弾の非対称は固定序盤の産物
5. **既定モデルは ppo2.pt** (外部基準で最強、蒸留と同等)。文化 (棋風) を見せたいときは E7 (0.16)。
   E2b (0.10〜0.13) は既定から外す

**含意:** 強さの経路として自己対戦 (PPO / リーグ) に投資しても外部強さは伸びない、という実測。
Phase C (エンジン教師蒸留) が正攻法であることの裏付け。E4 (GRU 解凍 PPO) も外部強さには効かない可能性が高い。
結果 JSON: `checkpoints/yardstick_open/*.json`。

## 実施記録: E4 (感情 GRU 解凍 PPO) を iter 253 で停止 (2026-09-19 13:30、ユーザー承認)

Δgru の推移 (‖gru₀‖ = 20.64):

| iter | 1 | 50 | 100 | 200 | 253 |
|---|---|---|---|---|---|
| Δgru | 0.005 (0.02%) | 0.042 (0.20%) | 0.063 (0.30%) | 0.082 (0.39%) | **0.091 (0.44%)** |

anchor 損失 0.000015、対開始時勝率 0.45 → 0.50 → 0.525 (40 局、横ばい)、H 0.81 → 0.71。

**停止の理由:** (1) Δgru の傾きが最初の 100 iter の 0.27 倍まで減速しており、iter 1200 まで回しても
外挿で 0.22 (1.1%) 止まり = 「解凍した」と言える状態にならない。(2) 報告指標の「対開始時勝率」は
自己相対で、同日の物差しでそれが外部強さと無関係と実測済み。(3) PPO 系は外部強さを上げないことも同日実測
(phase37 0.22 ≈ ppo2 0.23)。(4) 原因は anchor 1.0 が強すぎるか gru-lr 1e-5 が小さすぎるかで、
**間違った設定の完走 (残り 10.3h) では診断が進まない**。正しい検証は `--gru-lr 1e-4 --anchor 0.1` の
ゲート再走 (45 分) で、これは Phase E (内面、D の後) で行う。

**停止直前の反証と決着:** iter 250 の評価で対開始時勝率が 0.525 → **0.75** に跳ねた (40 局、SE 0.079 の 2.8 倍)。
ただし Δgru は 0.092 のままで GRU は動いていない = 上がったのは方策側 (普通の PPO の進歩)。
元の ppo2 キャンペーンでも「対アンカー勝率は iter 200 で 0.75 前後に早期飽和」と記録済みで、既知パターンの再現。
**外部基準で決着**: `ppo_e4_gru.pt` (iter 250) の対 Háo depth1 (序盤ブック、100 局) は **0.175** で、
出発点の ppo2 (0.23) より下 (差 1.4 SE)。自己相対の +0.225 は外部に伝わらなかった (同日 3 例目)。
結果: `checkpoints/yardstick_open/e4_iter250.json`。

保存物: `checkpoints/ppo_e4_gru.pt` (iter 250 時点)、
`checkpoints/ppo_e4_gru_history.json` (= `_stopped_iter249.json`)、ログ `checkpoints/e4_stage.log`。

## 実施記録: Phase C パイロット — 教師の効果は出なかった (2026-09-19 13:33〜14:29)

E4 停止で空いた GPU で、教師データが 1/3 (2023 の 6 シャード、カバー率 20〜33%) の段階で予行演習。
**対照を置いた**: `--no-teacher` (c_soft=0, teacher_value_weight=0) で完全に同条件を回し、教師の効果だけを分離する。

条件: ppo2.pt から warm start、2023、3,936 局、1 epoch、batch 24、tbptt 16、lr 5e-5、relations+council、amp。各 25.5 分。

| | val 一致率 | 説明率 | value 損失 | **対 Háo depth1 (100局)** | 先手/後手 |
|---|---|---|---|---|---|
| ppo2 (出発点) | — | — | — | **0.230** | 0.24 / 0.22 |
| pilot_teacher | 41.59% | 80.5% | 0.605 | **0.205** | 0.14 / 0.27 |
| pilot_control | 42.32% | 80.9% | 0.679 | **0.205** | 0.16 / 0.25 |

**結論: 教師あり/なしで外部強さは完全に同じ (0.205 vs 0.205)。**
同じ序盤・同じ手番で対応をとった符号検定でも teacher 15 勝 / control 14 勝 / 同点 71、**p ≈ 1.00**。
対応ありの SE は 0.054 なので、|差| > 0.11 程度の効果があれば検出できたはずで、それが無い。

副作用として、両者とも 1 epoch の蒸留で ppo2 の 0.230 → 0.205 と**わずかに下がった** (1 SE 以内)。
一致率は control の方が 0.7pt 高い (教師の soft target が one-hot と競合)。value 損失は教師が違うので直接比較不可。

**留保 (この結果で否定できないこと):** 教師データが最終量の 1/3、カバー率 20〜33%、1 epoch、3,936 局。
Ruoss et al. は 1,000 万局・150 億データ点・270M パラメータで grandmaster に届いており、
**規模が 3〜4 桁足りない**。「エンジン教師蒸留は効かない」ではなく「この規模では検出できない」が正しい読み。

**本走 (c_chain.sh、教師完了後) の判断:** 全シャード (7.6M 局面 × カバー率) × 20k 局で回して、
同じ対照つきで測る。そこでも差が出なければ、規模を上げるか (数日〜) 経路自体を見直すかの判断に入る。
結果 JSON: `checkpoints/yardstick_open/pilot_{teacher,control}.json`、ログ `checkpoints/c_pilot.log`。

## 実施記録: E4b — 感情 GRU の解凍は「無効」ではなく「有害」だった (2026-09-19 14:55〜15:58)

E4 の診断 (anchor 1.0 / gru-lr 1e-5 では GRU が動かない) を受けた再走。**anchor 0.1 / gru-lr 1e-4** で 100 iter。

| | Δgru @iter100 | ‖gru₀‖=20.64 比 | 対開始時勝率 (40局) | **対 Háo depth1 (100局)** |
|---|---|---|---|---|
| ppo2 (出発点) | — | — | — | **0.230** |
| E4 (lr 1e-5 / anchor 1.0) | 0.063 | 0.30% | 0.50 | 0.175 (iter250 時点、Δgru 0.091) |
| **E4b (lr 1e-4 / anchor 0.1)** | **0.500** | **2.42%** | 0.5625 | **0.110** |

Δgru は √t 則によく乗る (係数 E4b 0.0498 / E4 0.0058、iter100 の予測 0.498 に対し実測 0.5001)。
√t = 拡散 (ランダムウォーク) 的な動きで、E0 の θ_ind (log-log 傾き 0.46) と同じ形。

**対応あり符号検定 (同じ序盤・同じ手番で ppo2 と比較): E4b 8 勝 / ppo2 20 勝 / 同点 72、p = 0.036。**
100 局の独立標本で**有意な悪化**。

**結論: 感情 GRU を PPO で解凍すると外部強さが下がる。** しかも Δgru が大きいほど悪いという用量反応が見える
(0.44% → 0.175 / 2.42% → 0.110)。ゲート (WR 0.5625、Δgru 相対 2.4%、H10 0.726) は全項目 PASS したが、
**ゲートは崩壊の検知用で有効性の検知用ではない**ため通ってしまった。ゲート設計の限界として記録する。

**推定原因:** 感情 GRU は系列蒸留 (mood_distill) で意味のある感情を学習済み。PPO の疎報酬 (1局1ビット) で
√t 的に拡散させるとその構造が壊れる。anchor はそれを防ぐ項だが、1.0 では GRU が動かず 0.1 では拡散を止められない。
**「動く」と「壊れない」を両立する anchor が存在するかは未確認** (E4c で anchor 1.0 + lr 1e-4 の中間点を測る)。

**Phase E への含意:** 「感情を PPO で育てる」路線はこの形では筋が悪い。代わりに**感情に密な補助損失をつけた蒸留**
(勾配の出どころを疎報酬から教師に替える) を検討する。E4c の役割も「律速の切り分け」から
「拡散量と外部強さの負相関を 3 点で確認する」に変わった。
結果 JSON: `checkpoints/yardstick_open/e4b_iter100.json`。

## 実施記録: Phase B 先行実装 — 会議の random loop sampling (2026-09-19 夕、実行は明日)

サーベイで期待値 1 位とした項目を、GPU が塞がっている間に CPU 側で実装した。**明日は測定から始められる。**

**実装:**
- `model/policy.py`: `KokoroPolicy.council_round_choices`（既定は空タプル）。`_council` は
  (1) `rounds` 明示があればそれ、(2) `self.training` かつ choices があればそこから一様抽選、
  (3) それ以外は既定 `council_rounds` (=2)、の順で解決する。
  **評価時は必ず既定に戻る**ので物差しの再現性は壊れない
- `train/mood_distill.py`: `--council-round-choices 1 2 3 4`。`--council` 無しで指定すると停止。
  チェックポイントに `council_round_choices` を保存する
- `scripts/eval_vs_engine.py`: `--council-rounds R` で推論時のラウンド数を上書き（外部基準での R 掃引用）。
  `export_model_jsonl.ModelRunner.forward` にも `rounds` を通した
- `tests/test_relations_council.py`: 学習時に R が散らばり、`eval()` で既定に戻り、
  `rounds` 明示が学習中でも勝つ、の 3 点を議事録の長さで検証。**全 314 passed**
- 疎通確認: ppo2 を R=1/2/4 で 6 局ずつ走らせ、結果が R で変わることを確認（実際にラウンド数が効いている）

**明日の実験 (`checkpoints/ops/b_chain.sh`、GPU 解放後に自動起動):**
対照つきで 2 本、各 20k 局・1 epoch・教師なし（会議だけの効果を見る）:

| | 学習時のラウンド |
|---|---|
| `b_rls` | 1〜4 から一様抽選 |
| `b_fix` | 既定 R=2 固定（対照） |

その後、両方を **R = 1, 2, 3, 4, 6** で外部基準（対 Háo depth 1、序盤ブック、各 100 局）にかける。計 10 本 × 100 局。

**成功の形:** `b_rls` が R を増やすほど強くなる、または少なくとも R=3,4 で落ちない。
`b_fix` は既存の一致率の形（R=2 ピーク → R=3,4 で低下）を外部基準でも再現するはず。
**これが通れば、木を作らずに推論時計算量で強さを買えることになる** — 今日の「trunk を動かせるのは蒸留だけ」
という結論の、数少ない例外候補。
