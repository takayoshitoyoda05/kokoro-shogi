"""駒たちの主張を調停する会議ループ [D] (DESIGN.md §3(6))。

初期スコア $s^{(0)}_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$ から始めて、
ラウンド $r = 1..R$:

$$P^{(r)} = \\{\\mathrm{Embed}(i, a, s^{(r-1)}_{i,a}) : (i,a) \\in \\text{top-}k(s^{(r-1)})\\}$$
$$\\{h^{(r)}_i\\} = \\mathrm{Trunk}_{L-1:L}\\big([\\{h^{(r-1)}_i\\}; P^{(r)}]\\big)
  \\;\\Rightarrow\\; (3)(5)\\text{再計算} \\;\\Rightarrow\\; s^{(r)}_{i,a}$$

Universal Transformer 流の重み共有 (trunk の最終2層をそのまま再適用) なので、
**新しいパラメータは提案の埋め込み (ProposalEmbedding) だけ**。重み共有ゆえに
ラウンド数 R は学習時と推論時で変えられる (Gate の R=1..4 棋力比較はこれを使う)。

各ラウンドの top-k 提案 (駒, 手, bid=スコア) を記録して返す。これが
INTERFACE.md §3 の `council` (議事録) と viz/narrator の実況素材になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kokoro_shogi.config import ModelConfig
from kokoro_shogi.model.heads import NUM_MOVE_KINDS

#: 1ラウンドで場に出る提案の数 (DESIGN.md §3(6) の top-k)
DEFAULT_TOP_K = 8
#: 既定のラウンド数。R=3 が設計既定だが、推論コストの警告 (DESIGN.md §9) に従い2
DEFAULT_ROUNDS = 2


@dataclass(frozen=True)
class CouncilRoundLog:
    """1ラウンドぶんの議事録 (バッチ対応、テンソルは CPU に置かない)。"""

    #: 提案した駒トークン `(B, k)`
    token: Tensor
    #: 提案した手 (0-161 = 移動先×成り) `(B, k)`
    move_kind: Tensor
    #: 主張の強さ (このラウンド開始時点のスコア) `(B, k)`
    bid: Tensor


class ProposalEmbedding(nn.Module):
    """提案 $(i, a, s)$ → 提案トークン $\\mathrm{Embed}(i,a,s) \\in \\mathbb{R}^d$。

    駒の同一性は $h_i$ から取る (駒インデックスの絶対埋め込みにすると
    トークン並び依存になるため)。スコアの寄与 `from_score` は零初期化して、
    学習開始時の提案トークンを「駒と手の素直な和」から始める。
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        d_model = config.d_model
        self.from_piece = nn.Linear(d_model, d_model)
        self.move = nn.Embedding(NUM_MOVE_KINDS, d_model)
        self.from_score = nn.Linear(1, d_model, bias=False)
        nn.init.zeros_(self.from_score.weight)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden: Tensor, token: Tensor, move_kind: Tensor, score: Tensor) -> Tensor:
        """`hidden` `(B, N, d)`, `token`/`move_kind`/`score` `(B, k)` → `(B, k, d)`。"""
        picked = torch.gather(
            hidden, 1, token.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )
        return self.norm(
            self.from_piece(picked) + self.move(move_kind) + self.from_score(score.unsqueeze(-1))
        )


def top_proposals(scores: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """スコア `(B, N, 162)` → top-k の (token, move_kind, bid) 各 `(B, k)`。

    非合法手は呼び出し側で $-10^9$ に潰してあるので、そのまま上位を取れば
    合法手だけが提案になる。
    """
    flat = scores.flatten(start_dim=1)  # (B, N*162)
    bid, index = flat.topk(k, dim=-1)
    return index // NUM_MOVE_KINDS, index % NUM_MOVE_KINDS, bid


__all__ = [
    "DEFAULT_ROUNDS",
    "DEFAULT_TOP_K",
    "CouncilRoundLog",
    "ProposalEmbedding",
    "top_proposals",
]
