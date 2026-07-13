# Kokoro-Shogi: 駒エージェント将棋AI 統合設計書 v2

> 各駒が性格・感情・記憶・関係性を持ち、議論して指し、進化する将棋AI。
> 強さの経路(共有Transformer + 蒸留 + 自己対戦RL)と面白さの経路(駒の内面構造)を
> 分離して両立させる。
>
> v2: 全コンポーネントの数式仕様、ベースアルゴリズムの出典、学習フローを統合。

---

## 0. 設計原則

1. **強さの経路を壊さない**: すべての「面白さ機能」は補助構造(追加トークン特徴・
   補助損失・attentionバイアス・後処理)として実装し、外しても素の強いpolicy netが残る。
2. **単一のPieceStateに全機能を集約**: 全機能は `PieceState` のフィールドとして表現され、
   configの機能フラグで個別にON/OFFできる。
3. **段階的検証 (Gate制)**: 各Phaseは前Phaseの検証を通過してから着手。
4. **すべてログに残す**: 欲求・感情・bid・attention・議事録は構造化ログ(JSONL)に吐き、
   可視化・LLM実況・論文の図はログから生成。
5. **新規アルゴリズムは発明しない**: 全部品は実績ある手法(§9 出典表)の組み合わせ。
   新規性は「駒粒度への適用」「解釈可能な効用分解の構造化」「持ち駒=転生の扱い」に置く。

---

## 1. 記号の定義

| 記号 | 意味 |
|---|---|
| $t$ | 手数 (時刻) |
| $s_t$ | 局面 |
| $i, j \in \{1,\dots,N\}$ | 駒のインデックス ($N \le 40$、盤上+持ち駒) |
| $c(i)$, $p(i)$ | 駒 $i$ の駒種・位置 (持ち駒は専用位置ID) |
| $\mathcal{A}_i(s_t)$ | 駒 $i$ の合法手集合、$a_t$ は実際に指された手 |
| $z \in \{+1,-1\}$ | 対局の最終勝敗 |
| $\theta^{sp}_{c}$ | 駒種性格 (16次元、学習対象) |
| $\theta^{ind}_i$ | 個体性格 (16次元、対局後ループで進化) [B] |
| $m_i^{(t)}$ | 感情状態 (32次元、GRU hidden) [A] |
| $r_{ij}$ | 関係性バイアス [C] |
| $\text{loyalty}_i$, $\text{origin}_i$ | 忠誠・出生所有者 [E] |
| $\Phi_k = \{\theta^{sp}_c\}_c$ | 文化 $k$ の性格セット [F] |

---

## 2. コアデータモデル

```python
@dataclass
class PieceState:
    piece_id: str            # 例 "P27_gen3_0012" (初期位置+世代+連番)
    species: PieceSpecies
    # 性格
    theta_species: Tensor    # 駒種共有性格 (共有埋め込み参照)
    theta_individual: Tensor # 個体性格 [B]
    # 対局内の動的状態
    mood: Tensor             # 感情 m_i [A]
    loyalty: float           # [E]
    owner: Player
    origin_owner: Player     # [E]
    # 社会
    relations: dict[str, float]  # r_ij [C]
    # 対局を跨ぐ [B]
    career: CareerStats      # 生存率, 成り回数, 王手関与, MVP回数, 対局数
    lineage: list[str]       # 血統 (親piece_id) [F]
```

---

## 3. 推論パイプライン (局面 → 1手) — 完全数式仕様

### (1) トークン化

$$x_i^{(0)} = E_{sp}[c(i)] + E_{pos}[p(i)] + E_{flag}[\text{成}_i, \text{所有}_i]
           + W_{ind}\,\theta^{ind}_i + W_m\, m_i^{(t)}$$

### (2) 共有Transformer Trunk ($L$層、既定 $L=6$、$d=256$)

self-attention (マルチヘッド、1ヘッド分を表記):

$$e_{ij} = \frac{(x_i W_Q)(x_j W_K)^\top}{\sqrt{d}}
        + \underbrace{B_{\text{利き}}(i,j)}_{\text{利き関係バイアス (Lc0/GAB流)}}
        + \underbrace{r_{ij}}_{\text{関係性 [C]}}$$

$$\alpha_{ij} = \mathrm{softmax}_j(e_{ij}), \qquad
  x_i^{(\ell+1)} = \mathrm{FFN}\Big(x_i^{(\ell)} + \sum_j \alpha_{ij}\,(x_j W_V)\Big)$$

出力: $h_i = x_i^{(L)} \in \mathbb{R}^{256}$

### (3) 欲求ヘッド — 手 $a$ ごと (移動先 $\text{dst}(a)$、成り $\rho(a)$)

$$u_{i,a} = [\,h_i \,\|\, E_{pos}[\text{dst}(a)] \,\|\, \rho(a)\,] \in \mathbb{R}^{321}$$

$$d_i(a) = \sigma\big(W_2\,\mathrm{ReLU}(W_1 u_{i,a})\big) \in [0,1]^6$$

欲求6軸と実測教師:

| 軸 $k$ | 定義 | 教師 $y_i^{(k)}$ (対局ログから自動生成) |
|---|---|---|
| 生存 | $k{=}4$手で取られない | $\mathbb{1}[\text{駒}i\text{が}t{+}4\text{で盤上}]$ |
| 攻撃 | 捕獲/王手に関与 | 実際の捕獲・王手 |
| 成り | 4手以内に成る | 実際に成ったか |
| 守備 | 味方(特に玉)への利き貢献 | 利き数の実測 |
| 前進 | 敵陣への接近 | 位置変化量 |
| 再登場 | (持ち駒専用) 打たれる | 実際に打たれたか |

### (4) 性格重み — 正制約が解釈可能性の要 (QMIX単調性と同思想)

$$w_i = \mathrm{softplus}\big(W_w\,[\,\theta^{sp}_{c(i)} \,\|\, \theta^{ind}_i \,\|\, m_i^{(t)}\,]\big)
      \in \mathbb{R}_{>0}^{6}$$

$m_i$ 依存により同一個体でも対局中に性格が動く [A]。

### (5) 自由項 (大局観の受け皿、L2罰則付き)

$$g_i(a) = W_4\,\mathrm{ReLU}(W_3 u_{i,a}) \in \mathbb{R}$$

$|g_i(a)|$ の大きい手 = 欲求(本能)に反した大局的判断 = **捨て駒検出器**。

### (6) 手スコアと会議調停 [D]

初期スコア: $s^{(0)}_{i,a} = \langle w_i, d_i(a)\rangle + g_i(a)$

会議ラウンド $r=1..R$ (既定 $R=3$、trunk最終2層を重み共有で再適用):

$$P^{(r)} = \{\mathrm{Embed}(i,a,s^{(r-1)}_{i,a}) : (i,a)\in\text{top-}k(s^{(r-1)})\}$$
$$\{h^{(r)}_i\} = \mathrm{Trunk}_{L-1:L}\big([\{h^{(r-1)}_i\};P^{(r)}]\big)
  \;\Rightarrow\; (3)(5)\text{再計算} \;\Rightarrow\; s^{(r)}_{i,a}$$

各ラウンドの $\Delta s$ をJSONLログ → 議事録 → (後処理) LLM実況。

### (7) 方策 = マスク付き温度softmax (微分可能な調停)

$$\pi(a\mid s_t) = \frac{\exp(s^{(R)}_{i,a}/\tau)\cdot\mathbb{1}[a\in\mathcal{A}_i]}
                       {\sum_j\sum_{a'\in\mathcal{A}_j}\exp(s^{(R)}_{j,a'}/\tau)}$$

非合法手は $-10^9$ に置換してからsoftmax。学習時 $\tau{=}1$、対局時 $\tau{\approx}0.1$。

### (8) 価値と単調Mixing (QMIX/Qatten流)

$$V_i = W_v h_i, \quad \bar h = \tfrac1N\sum_i h_i, \quad
  \alpha_i = |W_{hyp}\bar h|_i \ge 0$$

$$V(s_t) = \sum_i \alpha_i V_i + \mathrm{MLP}_b(\bar h), \qquad
  \frac{\partial V}{\partial V_i} = \alpha_i \ge 0 \;(\text{IGM/単調性})$$

$\alpha_i$ = 「駒 $i$ の価値判断が今の形勢にどれだけ効いているか」(解釈出力)。

### (9) 手の決定・感情更新 [A]・忠誠 [E]

$$a_t \sim \pi(\cdot\mid s_t)\;(\text{学習時}), \qquad a_t = \arg\max \pi\;(\text{対局時})$$

$$m_i^{(t+1)} = \mathrm{GRU}\big(u^{ev}_i(s_t,a_t),\, m_i^{(t)}\big), \quad
  \text{例: } u^{ev}_{i,1} = \sum_{j\in\text{被取}} e^{-\|p(i)-p(j)\|_1/\beta}$$

捕獲後 ($\text{owner}_i \ne \text{origin}_i$) の忠誠ハンデ [E, バリアント限定]:

$$d_i(a) \leftarrow d_i(a)\cdot(1-\kappa_E\cdot\text{loyalty}_i)$$

---

## 4. 学習パイプライン — 完全数式仕様

### 統一損失 (全フェーズ共通の骨格)

$$\boxed{\,L = L_{\text{policy}} + c_1 L_{\text{value}} + c_2 L_{\text{desire}}
        + \lambda_g L_g - c_3 H(\pi)\,}$$

$$L_{\text{desire}} = \sum_{i,k}\mathrm{BCE}\big(d^{(k)}_i(a_t), y^{(k)}_i\big), \qquad
  L_g = \tfrac{1}{|\mathcal{A}|}\sum_{i,a} g_i(a)^2, \qquad
  H(\pi) = -\sum_a \pi\log\pi$$

$L_{\text{policy}}$ だけがフェーズで入れ替わる。他は全フェーズ並走。

### 4a. Phase 1-2: 蒸留 (教師あり多タスク学習、dlshogi式)

データ: floodgate/dlshogi棋譜 $\mathcal{D}=\{(s, a^\ast, z, \{y^{(k)}_i\})\}$ (ラベルは前処理で自動生成)

$$L_{\text{policy}} = -\log\pi(a^\ast\mid s), \qquad L_{\text{value}} = (V(s)-z)^2$$

$$\Theta \leftarrow \Theta - \eta\nabla_\Theta\,\mathbb{E}_\mathcal{D}[L] \quad (\text{AdamW})$$

評価 ($\lambda_g$ 掃引でトレードオフ曲線 = 論文の主図):

$$\text{一致率} = \mathbb{E}\,\mathbb{1}[\arg\max\pi = a^\ast], \qquad
  \text{欲求説明率} = \frac{\mathbb{E}\langle w_i,d_i\rangle^2}{\mathbb{E}\,s_{i,a}^2}$$

### 4b. Phase 7-α: 自己対戦RL (MAPPO、第一候補)

疎報酬: $r_t = 0\,(t<T),\; r_T = z$

GAE: $\;\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
      \hat A_t = \sum_{l\ge0}(\gamma\lambda)^l \delta_{t+l} \quad (\gamma{=}1, \lambda{=}0.95)$

PPOクリップ ($\epsilon{=}0.2$):

$$\rho_t = \frac{\pi_\Theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}, \qquad
  L_{\text{policy}} = -\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\,
    \mathrm{clip}(\rho_t,1{\pm}\epsilon)\hat A_t\big)\Big]$$

$$L_{\text{value}} = (V-\hat R_t)^2, \quad \hat R_t = \hat A_t + V(s_t)$$

ループ: [128並列自己対戦で軌跡収集] → [GAE計算] → [数epochのPPO更新] → 繰返し。

### 4c. Phase 7-β: Gumbel AlphaZero (探索あり、切替オプション)

候補選択 (Gumbel-Top-k、$G(a)\sim\mathrm{Gumbel}(0)$ i.i.d.):

$$\mathcal{C} = \text{top-}k_a\big[G(a) + \mathrm{logits}(a)\big], \quad
  \mathrm{logits}(a) = s_{i,a}/\tau$$

Sequential Halving (予算 $n$ を $\log_2 k$ 段に分割):

$$\mathcal{C} \leftarrow \text{top-}\tfrac{|\mathcal{C}|}{2}
  \big[G(a)+\mathrm{logits}(a)+\sigma(\hat q(a))\big], \quad
  \sigma(q) = (c_{visit}+\max_b N(b))\,c_{scale}\,q$$

方策教師 (completed Q: 未訪問手は $V$ で補完):

$$\pi'(a) = \mathrm{softmax}\big(\mathrm{logits}(a)+\sigma(\hat q_{\text{comp}}(a))\big), \qquad
  L_{\text{policy}} = \mathrm{KL}(\pi'\,\|\,\pi_\Theta)$$

少数シミュレーション(16〜64/手)で成立するのが採用理由 (Gumbel MuZeroは2simでも学習可)。

### 4d. Phase 5-6: 対局後ループ (勾配ループの外側、trunk凍結)

**駒別advantage (COMA counterfactual):**

$$A_i(s_t,a_t) = Q(s_t,a_t) - \sum_{a'\in\mathcal{A}_i}
  \frac{\pi(a'|s_t)}{\sum_{a''\in\mathcal{A}_i}\pi(a''|s_t)}\,Q(s_t,a')$$

($Q$ は $\hat q$ または $r+\gamma V(s_{t+1})$ で近似)

**個体性格の更新 [B]** ($\eta_{ind}\ll\eta$、normクリップ $\kappa$):

$$\theta^{ind}_i \leftarrow \mathrm{clip}_{\|\cdot\|\le\kappa}\Big(
  \theta^{ind}_i + \eta_{ind}\sum_{t:\,a_t\in\mathcal{A}_i}
  A_i(s_t,a_t)\,\nabla_{\theta^{ind}_i}\log\pi(a_t|s_t)\Big)$$

**キャリア [B]:** $\;\mathrm{MVP}_g = \arg\max_i\sum_t A_i, \quad
  \text{生存率}_i \leftarrow \mathrm{EMA}(\mathbb{1}[\text{終局時盤上}])$

**血統 (ES交叉+変異) [B/F]:**

$$\theta^{ind}_c = \beta\theta^{ind}_p + (1{-}\beta)\theta^{ind}_q + \varepsilon, \quad
  \varepsilon\sim\mathcal{N}(0,\sigma^2_{mut}I),\; \beta\sim U(0,1)$$

**文化リーグ (PBT) [F]** (リーグ節ごと、下位淘汰・上位複製変異):

$$\Phi_k \leftarrow \Phi_{k^\ast} + \mathcal{N}(0,\sigma^2_{PBT}), \qquad
  (\tau,\lambda_g)_k \leftarrow (\tau,\lambda_g)_{k^\ast}\cdot e^{\mathcal{N}(0,0.2^2)}, \qquad
  k^\ast = \arg\max_k \mathrm{WR}_k$$

### 依存関係マップ

$$\underbrace{(1)(2)}_{\text{表現}} \to \underbrace{(3)\text{-}(6)}_{\text{スコア}}
  \to \underbrace{(7)}_{\pi} \to
  \begin{cases}\text{蒸留: } -\log\pi(a^\ast)\\ \text{RL: } L_{\text{PPO}} \text{ or } \mathrm{KL}(\pi'\|\pi)\end{cases}$$

$$\underbrace{(8)}_{V,V_i,\alpha_i} \to
  \begin{cases}\hat A_t\ (\text{GAE, 学習の燃料})\\
  A_i\ (\text{COMA, 駒の功績}) \to \theta^{ind}\text{更新・キャリア・血統・文化}\end{cases}$$

二重ループ構造: 内側 = 勾配 (trunk+ヘッド+$\theta^{sp}$)、
外側 = 進化 ($\theta^{ind}$, $\Phi_k$, 血統)。混ぜない。

---

## 5. ベースアルゴリズム出典表

| 設計要素 | ベース | 借りているもの |
|---|---|---|
| trunk・トークン化 | Chessformer / Lc0 transformer | マス(→駒)トークン化, 利きattentionバイアス, src-dst方策ヘッド |
| 方策(調停) | softmax方策 + Gumbel-Softmax | argmax調停の微分可能緩和, 温度制御 |
| 蒸留 | dlshogi式教師あり学習 | floodgate棋譜前処理, policy/value同時学習 |
| 自己対戦RL(α) | MAPPO (Yu+ 2022) | パラメータ共有actor+中央critic, PPOクリップ, GAE |
| 自己対戦RL(β) | Gumbel AlphaZero (Danihelka+ ICLR22) / MA版 (AAAI24) | Gumbel-Top-k, sequential halving, completed Q |
| 価値分解 | VDN→QMIX→Qatten | 単調mixing (hypernetwork+非負制約), IGM |
| 駒別advantage | COMA (Foerster+ 2018) | counterfactual baseline (離散行動) |
| 感情 [A] | リカレント方策 (DRQN/R2D2系) | GRU state。独自要素はイベント特徴設計と可視化のみ |
| 関係性 [C] | Lc0 学習可能attentionバイアス | 同じ挿入位置に r_ij を加算, 永続化が拡張 |
| 会議 [D] | Universal Transformer (反復適用) + CNP/オークション | 提案条件付きdeliberation, bidログ |
| 欲求補助損失 | UNREAL系 auxiliary tasks | 自動ラベルの副問題で表現学習を助ける |
| 血統・文化 [B/F] | ES(交叉+変異) + PBT + AlphaStarリーグ | 集団維持, 淘汰複製変異, メタゲーム観測 |
| 実装基盤候補 | MiniZero, cshogi/python-shogi | AZ/Gumbel両対応フレームワーク, 将棋環境 |

**単体で新規のアルゴリズムはゼロ。** 新規性は (a) ターン制完全情報ゲームの駒粒度MARL検証,
(b) 持ち駒=エージェント転生という将棋固有問題, (c) 構造的解釈可能性と棋力の
トレードオフ定量化, の3点。

---

## 6. 実装フェーズとマイルストーン (Gate制)

> **注**: 「目安」は研究として各Phaseを丁寧にやる場合の工数。チーム開発
> (TEAM_PLAN v4 / 10週) では縮退版を毎週1 Phase消化する: Phase 0=週1-2,
> 1=週3, 2=週4, 3=週5, 4=週6, 5=週7, 7=週8 (5五将棋MAPPO), 6=週9 (4文化)。
> Gateの完了条件も10週中は縮退 (λ_g掃引3点、RLは「兆候が見える」まで)。
> フル条件での再実行は10週後の研究フェーズ (GPW投稿準備) で行う。

| Phase | 内容 | 完了条件 (Gate, フル版) | 目安 (フル版) |
|---|---|---|---|
| 0 | 環境: cshogi, floodgate棋譜DL・前処理, トークン化, 欲求ラベル事前計算 | 単体テスト, 1M局面変換ベンチ | 1-2週 |
| 1 | ベースライン: 駒トークンtrunk + 通常policyヘッドで蒸留 §4a | 一致率がCNNベースライン比 -2%以内 | 2-3週 |
| 2 | 基盤: 欲求(3)+性格(4)+自由項(5), λ_g掃引 | トレードオフ曲線, 捨て駒検出の定性確認 | 2-3週 |
| 3 | A感情(9) + C関係性(2) | アブレーション(棋力±), 感情ヒートマップデモ | 2-3週 |
| 4 | D会議(6) + 議事録 + LLM実況 | R=1..4棋力比較, 実況デモ | 2-3週 |
| 5 | B永続化(4d) + E忠誠(9) ※5五将棋 | キャリアUI, E三案(転生/リセット/ハンデ)比較 | 3-4週 |
| 6 | F文化リーグ(4d) ※5五将棋→本将棋 | 文化分化の観測 (性格分布距離の世代推移) | 4週〜 |
| 7 | 自己対戦RL §4b→§4c | 技巧2/やねうら王(ノード制限)勝率 → floodgate | 継続 |

Phase 4 終了時点で「感情を持つ駒が議論して指し実況される将棋」としてデモ・発表可能。

---

## 7. リポジトリ構成

チーム開発ではモノレポ `kokoro-shogi` に統合された (正式な構成・Git運用は
`docs/REPO_STRUCTURE.md` が正本)。本設計書に対応するAI側モジュールの
式番号マッピングのみここに記す:

```
src/kokoro_shogi/
├── core/            # PieceState, トークン化(1), 利き関係抽出(2のB_effect素材)
├── model/
│   ├── trunk.py     # (2) Transformer + 利き/関係バイアス [C]
│   ├── heads.py     # (3)(4)(5)(8) 欲求/性格/自由項/critic/単調mixing
│   ├── mood.py      # (9) 感情GRU [A]
│   ├── council.py   # (6) 会議ループ [D]
│   └── policy.py    # (7) 全体を束ねる方策
├── train/           # distill(§4a) / selfplay_ppo(§4b) / league(§4d PBT [F])
├── persist/         # SQLite: θ_ind, career, lineage [B] (§4d)
├── logging/         # docs/INTERFACE.md schema準拠のJSONL出力
├── viz/             # narrator (Template/Ollama実況), 可視化
└── server/          # WebSocketサーバ (INTERFACE.md §1)
```

忠誠 [E] は `variant/` 相当の機能フラグ (configs/features.yaml: loyalty) で
本線と分離し、棋力評価には使わない。実験結果は `experiments/` に置く。

---

## 8. 計測と評価

- **棋力**: 蒸留一致率 → ノード制限エンジン対局勝率 → floodgate/レーティングサイト
- **面白さの定量化**:
  - 欲求説明率 $\mathbb{E}\langle w,d\rangle^2/\mathbb{E}s^2$ と一致率のトレードオフ曲線
  - 捨て駒検出 precision/recall ($|g|$ 上位手 vs プロ棋譜アノテーション)
  - 感情 $m_i$ と形勢 $V$ の相関 / 関係グラフと囲い進行の一致度
  - 文化分化: $\|\Phi_k - \Phi_{k'}\|$ の世代推移, 戦型統計との相関
- **アブレーション**: 機能 {A,C,D} ON/OFF × 棋力 × 説明率 の全組合せ表 (論文の柱)

## 9. リスクと対策

| リスク | 対策 |
|---|---|
| Phase 1で一致率が出ない | 利きバイアス強化 / マス+駒ハイブリッドトークンに後退 |
| 欲求制約で棋力低下 | λ_g↓ (自由項に逃がす)。欲求は解釈用と割り切る |
| θ_ind更新が学習を不安定化 | η_ind微小化, normクリップκ, trunk凍結の厳守 |
| 会議で推論が遅い | R=2 / bid分散が大きい局面のみR増 |
| COMAのQ近似が粗い | まず Σ_t A_i の相対比較(MVP等)にのみ使用, 絶対値を信用しない |
| Fの計算資源不足 | 5五将棋で全実験 → 本将棋は上位2文化のみ |
| 全部盛りで未完 | Gate制。Phase 4で一度「完成形」を作る |
