"""駒同士の関係性 $r_{ij}$ [C] (DESIGN.md §3(2))。

構成は感情 [A] と同じ「状態 + 零初期化の合流点」の2部品:

- **関係状態** $R_{ij} \\in [0,1]$: 味方の紐 (SUPPORTS_ALLY) の共起をEMAで
  蓄積した「絆」。1手ごとに

  $$R^{(t+1)}_{ij} = (1-\\delta)R^{(t)}_{ij} + \\delta\\,\\mathrm{bond}_{ij}(s_t), \\qquad
    \\mathrm{bond}_{ij} = \\tfrac12(\\mathbb{1}[i\\to j\\text{ 紐}] + \\mathbb{1}[j\\to i\\text{ 紐}])$$

  片紐で0.5、相互紐で1.0へ向かう。囲いのように**長く紐を保った相手ほど絆が濃い**。
  一時的にすれ違っただけの駒とは絆にならない (瞬間の紐は B_利き が既に見ている。
  r_ij はその時間積分であることが存在意義)。INTERFACE.md §3 の
  `pieces[].relations[].r` はこの値をそのまま送る。
- **attentionバイアス** (`RelationAttentionBias`): $e_{ij} \\mathrel{+}= w_h R_{ij}$。
  Lc0流の学習可能バイアスで、挿入位置は利きバイアス $B_{利き}$ と同じ。
  ヘッドごとのスカラー重みを**零初期化**するので、フラグONでも学習前は
  OFFと完全に同じ出力 (mood の $W_m$、effect_bias と同じ流儀)。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kokoro_shogi.core.effects import SUPPORTS_ALLY

#: 絆EMAの更新率。相互紐10手で R ≈ 0.65、20手で ≈ 0.88
BOND_DELTA = 0.1


def bond_from_effect(effect: Tensor) -> Tensor:
    """利き関係行列 `(B, N, N)` (long) → その瞬間の絆 `(B, N, N)` (float)。

    片紐0.5 / 相互紐1.0。SUPPORTS_ALLY は味方間にしか立たないので
    敵味方のフィルタは不要。対角は0にする (自分との絆は無意味)。
    """
    support = (effect == SUPPORTS_ALLY).to(torch.float32)
    bond = 0.5 * (support + support.transpose(1, 2))
    eye = torch.eye(bond.shape[-1], dtype=torch.bool, device=bond.device)
    return bond.masked_fill(eye, 0.0)


def update_relations(prev: Tensor, effect: Tensor, *, delta: float = BOND_DELTA) -> Tensor:
    """$R^{(t+1)} = (1-\\delta)R^{(t)} + \\delta\\,\\mathrm{bond}(s_t)$。"""
    return (1.0 - delta) * prev + delta * bond_from_effect(effect)


def initial_relations(batch: int, tokens: int, device: torch.device | None = None) -> Tensor:
    """対局開始時 $R^{(0)} = 0$。`(B, N, N)`。"""
    return torch.zeros(batch, tokens, tokens, device=device)


class RelationAttentionBias(nn.Module):
    """関係状態 → attentionへの加算バイアス $w_h R_{ij}$。

    ヘッドごとに1スカラーの重み。零初期化なので学習開始時点では
    素のattentionと完全に一致する (EffectAttentionBias と同じ)。
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_heads))

    def forward(self, relation: Tensor) -> Tensor:
        """`relation` `(B, N, N)` → `(B, n_heads, N, N)`。"""
        return relation.unsqueeze(1) * self.weight.view(1, -1, 1, 1)


__all__ = [
    "BOND_DELTA",
    "RelationAttentionBias",
    "bond_from_effect",
    "initial_relations",
    "update_relations",
]
