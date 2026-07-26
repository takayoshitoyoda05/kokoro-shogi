"""駒表現から手スコアと状態価値を出す各ヘッド (DESIGN.md §3(3)-(8))。

**Phase 1 (素のベースライン)**

- `PolicyHead`: 素の src-dst 方策ヘッド (DESIGN.md §5 出典表 Chessformer/Lc0)
- `ValueHead`: 状態価値 $V(s)$

**Phase 2 (欲求による構造化)** — 手スコアが
$s_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$ に置き換わる

- `DesireHead` (3): 欲求 $d_i(a) \\in [0,1]^6$
- `PersonalityWeights` (4): 性格重み $w_i = \\mathrm{softplus}(\\cdot) > 0$
- `FreeTermHead` (5): 自由項 $g_i(a)$ ($\\lambda_g$ でL2罰則。$|g|$ が捨て駒検出器)
- `MonotonicValueMixing` (8): $V = \\sum_i \\alpha_i V_i + \\mathrm{MLP}_b(\\bar h)$
  ($\\alpha_i \\ge 0$)

どちらの経路も **出力の形 `(B, N, 81, 2)` は同じ**。`policy.py` の `head=` で
切り替える。Gate1 (素の棋力 vs CNN) は `plain` で測り、$\\lambda_g$ 掃引と
解釈可能性の評価は `desire` で行う。

$w_i > 0$ (softplus) と $\\alpha_i \\ge 0$ (絶対値) の**正制約が解釈可能性の要**。
これがあるので「欲求 $d$ が上がれば手スコアも上がる」「駒 $i$ の価値判断が
全体の $V$ に単調に効く」と読める (QMIX の単調性と同じ思想)。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kokoro_shogi.config import ModelConfig
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.data.labels import NUM_AXES

#: 成らない / 成る
NUM_PROMOTE = 2
#: 手の移動先 × 成り の組み合わせ数
NUM_MOVE_KINDS = NUM_SQUARES * NUM_PROMOTE
#: 手 (移動先・成り) の埋め込み次元。DESIGN.md §3(3) の $u_{i,a} \\in \\mathbb{R}^{321}$ に対応
D_MOVE_EMBED = 64
#: 欲求ヘッド・自由項ヘッドの中間層
D_HEAD_HIDDEN = 32
#: 性格重みの下限 (softplus が float32 でアンダーフローして 0 になるのを防ぐ)
MIN_WEIGHT = 1e-6


class PolicyHead(nn.Module):
    """駒表現 $h_i$ → 手スコア `(B, N, 81, 2)`。

    DESIGN.md §3(3) の $u_{i,a} = [h_i \\| E_{pos}[dst(a)] \\| \\rho(a)]$ をMLPに通す形は、
    全 $40 \\times 81 \\times 2 = 6480$ 通りに対して毎回MLPを走らせることになり
    Phase 1 の素の方策には重すぎる。ここでは同じ「駒 × 移動先 × 成り」の
    分解を保ったまま、内積で書ける形にしている (Lc0 の src-dst ヘッドと同じ):

    $$s_{i,a} = \\langle W_p h_i,\\; E_{move}[dst(a), \\rho(a)] \\rangle + b_{dst(a),\\rho(a)}$$

    Phase 2 の欲求ヘッドは合法手だけを相手にするので (1局面あたり平均80手ほど)、
    そちらは DESIGN.md どおりMLPで実装できる。
    """

    def __init__(self, config: ModelConfig | None = None, d_head: int | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.d_head = d_head or config.d_model

        self.project = nn.Sequential(
            nn.Linear(config.d_model, self.d_head),
            nn.GELU(),
            nn.Linear(self.d_head, self.d_head),
        )
        self.move = nn.Embedding(NUM_MOVE_KINDS, self.d_head)
        self.bias = nn.Parameter(torch.zeros(NUM_MOVE_KINDS))

    def forward(self, hidden: Tensor) -> Tensor:
        """`hidden` `(B, N, d_model)` → 手スコア `(B, N, 81, 2)`。"""
        batch, tokens, _ = hidden.shape
        projected = self.project(hidden)  # (B, N, d_head)
        scores = projected @ self.move.weight.T + self.bias  # (B, N, 162)
        return scores.view(batch, tokens, NUM_SQUARES, NUM_PROMOTE)


class ValueHead(nn.Module):
    """駒表現の平均 $\\bar h$ → 状態価値 $V(s) \\in [-1, +1]$。

    DESIGN.md §3(8) の単調mixing $V = \\sum_i \\alpha_i V_i + \\mathrm{MLP}_b(\\bar h)$ は
    Phase 2 で入れる。Phase 1 はその第2項だけの素のcriticに相当する。
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
            nn.Tanh(),
        )

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """`hidden` `(B, N, d)` と `mask` `(B, N)` → `(B,)`。"""
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.mlp(pooled).squeeze(-1)


# --- Phase 2: 欲求による構造化 (DESIGN.md §3(3)-(5)(8)) ------------------------


class MoveConditionedMLP(nn.Module):
    """$u_{i,a} = [h_i \\| E_{pos}[dst(a)] \\| \\rho(a)]$ を受ける2層MLP。

    DESIGN.md §3(3)(5) の欲求ヘッドと自由項ヘッドは、どちらもこの形をしている
    (出力次元だけ違う)。

    **計算の工夫**: 素直に $40 \\times 81 \\times 2 = 6480$ 通りの $u_{i,a}$ を作ると
    1局面あたり 6480×321 の入力テンソルになって現実的でない。第1層は線形なので

    $$W_1 u_{i,a} = \\underbrace{W_1^{(h)} h_i}_{駒ごと}
                  + \\underbrace{W_1^{(m)} E_{move}[a]}_{手ごと}$$

    と分解できる。駒ごとの項 `(B, N, H)` と手ごとの項 `(162, H)` を作ってから
    ブロードキャストで足すので、中間テンソルは `(B, N, 162, H)` で済む。
    $H$ が小さい (既定32) ので、バッチ256で 200MB 程度。

    $E_{pos}[dst]$ と $\\rho$ は分けず、162通りの手をまとめて1つの埋め込みにしている
    (線形層に入る直前で連結するのと等価)。
    """

    def __init__(self, d_model: int, d_out: int, *, hidden: int = D_HEAD_HIDDEN) -> None:
        super().__init__()
        self.move = nn.Embedding(NUM_MOVE_KINDS, D_MOVE_EMBED)
        self.from_piece = nn.Linear(d_model, hidden)
        self.from_move = nn.Linear(D_MOVE_EMBED, hidden, bias=False)
        self.out = nn.Linear(hidden, d_out)

    def forward(self, hidden: Tensor) -> Tensor:
        """`hidden` `(B, N, d_model)` → `(B, N, 162, d_out)`。"""
        piece_term = self.from_piece(hidden).unsqueeze(2)  # (B, N, 1, H)
        move_term = self.from_move(self.move.weight)  # (162, H)
        return self.out(torch.relu(piece_term + move_term))


class DesireHead(nn.Module):
    """欲求ヘッド $d_i(a) = \\sigma(W_2\\,\\mathrm{ReLU}(W_1 u_{i,a})) \\in [0,1]^6$。

    6軸は `data/labels.DESIRE_AXES` と同じ並び
    (生存/攻撃/成り/守備/前進/再登場)。教師は棋譜から自動生成したラベルで、
    蒸留の補助損失 $L_{desire}$ に使う (DESIGN.md §4)。

    INTERFACE.md §3 の `pieces[].desire` にそのまま入る値でもある。
    """

    def __init__(self, config: ModelConfig | None = None, *, hidden: int = D_HEAD_HIDDEN) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.mlp = MoveConditionedMLP(config.d_model, NUM_AXES, hidden=hidden)

    def forward(self, hidden: Tensor) -> Tensor:
        """`(B, N, d_model)` → `(B, N, 162, 6)`。"""
        return torch.sigmoid(self.mlp(hidden))


class FreeTermHead(nn.Module):
    """自由項 $g_i(a) = W_4\\,\\mathrm{ReLU}(W_3 u_{i,a}) \\in \\mathbb{R}$。

    欲求では説明できない残差の受け皿。$\\lambda_g$ でL2罰則をかけて小さく保つので、
    それでも大きい $|g_i(a)|$ = 「本能に反した大局的判断」= **捨て駒検出器**
    (DESIGN.md §3(5))。
    """

    def __init__(self, config: ModelConfig | None = None, *, hidden: int = D_HEAD_HIDDEN) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.mlp = MoveConditionedMLP(config.d_model, 1, hidden=hidden)

    def forward(self, hidden: Tensor) -> Tensor:
        """`(B, N, d_model)` → `(B, N, 162)`。"""
        return self.mlp(hidden).squeeze(-1)


class PersonalityWeights(nn.Module):
    """性格重み $w_i = \\mathrm{softplus}(W_w[\\theta^{sp} \\| \\theta^{ind} \\| m_i]) > 0$。

    駒種性格 $\\theta^{sp}$ は駒種ごとの学習可能な埋め込み (DESIGN.md §1)。
    個体性格 $\\theta^{ind}$ [B] と感情 $m_i$ [A] は機能フラグが立ってから連結する
    (今は駒種性格のみ)。

    **softplus で正に保つのが要**。$w_i > 0$ なので
    $s_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$ は欲求に対して単調増加になり、
    「この駒は生存を重く見ている」と読める。負を許すと解釈が崩れる。

    softplus の出力に `MIN_WEIGHT` を足しているのは、入力が大きく負に振れると
    $\\log(1+e^x)$ が float32 でアンダーフローして**ちょうど0**になるため。
    0になるとその欲求軸が完全に無視され、勾配も流れなくなって復帰できない。
    数学的な正値性を浮動小数点でも守るための下限。
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        num_species: int,
        d_mood: int = 0,
        d_individual: int = 0,
    ) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.theta_species = nn.Embedding(num_species, config.d_theta)
        self.project = nn.Linear(config.d_theta + d_mood + d_individual, NUM_AXES)

    def forward(
        self, species: Tensor, mood: Tensor | None = None, individual: Tensor | None = None
    ) -> Tensor:
        """`species` `(B, N)` → 性格重み `(B, N, 6)` (全て正)。"""
        parts = [self.theta_species(species)]
        if individual is not None:
            parts.append(individual)
        if mood is not None:
            parts.append(mood)
        return nn.functional.softplus(self.project(torch.cat(parts, dim=-1))) + MIN_WEIGHT


class MonotonicValueMixing(nn.Module):
    """駒ごとの価値を単調に混ぜて $V(s)$ にする (DESIGN.md §3(8) QMIX/Qatten流)。

    $$V_i = W_v h_i, \\quad \\bar h = \\tfrac1N\\sum_i h_i, \\quad
      \\alpha_i = |W_{hyp}\\bar h|_i \\ge 0$$
    $$V(s) = \\sum_i \\alpha_i V_i + \\mathrm{MLP}_b(\\bar h)$$

    $\\alpha_i \\ge 0$ なので $\\partial V/\\partial V_i \\ge 0$ (IGM/単調性)。
    値域を $[-1,+1]$ (勝敗 $z$ と同じ) に収めるため最後に tanh を通すが、
    $\\tanh' > 0$ なので単調性は保たれる。
    この $\\alpha_i$ が「駒 $i$ の価値判断が今の形勢にどれだけ効いているか」= 発言力で、
    INTERFACE.md §3 の `pieces[].alpha` として Unity へ送る値になる。
    """

    def __init__(self, config: ModelConfig | None = None, *, max_pieces: int) -> None:
        super().__init__()
        config = config or ModelConfig()
        d_model = config.d_model
        self.max_pieces = max_pieces

        self.piece_value = nn.Linear(d_model, 1)
        self.hyper = nn.Linear(d_model, max_pieces)
        self.bias = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, hidden: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """`(B, N, d)` → `(V (B,), V_i (B, N), alpha (B, N))`。

        `alpha` は無効トークンを0にしたうえで合計1に正規化してある
        (INTERFACE.md の `alpha ∈ [0,1]` に合わせるため)。
        """
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        piece_value = self.piece_value(hidden).squeeze(-1) * mask
        alpha = self.hyper(pooled).abs() * mask
        # 合計1に正規化してから混ぜる。40駒ぶんを素で足すと初期値から tanh が
        # 飽和して勾配が消える。正のスカラーで割るだけなので単調性は保たれる。
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-6)

        value = (alpha * piece_value).sum(dim=1) + self.bias(pooled).squeeze(-1)

        return torch.tanh(value), piece_value, alpha


__all__ = [
    "D_HEAD_HIDDEN",
    "D_MOVE_EMBED",
    "MIN_WEIGHT",
    "NUM_MOVE_KINDS",
    "NUM_PROMOTE",
    "DesireHead",
    "FreeTermHead",
    "MonotonicValueMixing",
    "MoveConditionedMLP",
    "PersonalityWeights",
    "PolicyHead",
    "ValueHead",
]
