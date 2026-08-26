"""全駒トークンを文脈化する共有Transformer trunk (DESIGN.md §3(1)(2))。

トークン埋め込み:

$$x_i^{(0)} = E_{sp}[c(i)] + E_{pos}[p(i)] + E_{flag}[成_i, 所有_i]
             + W_{ind}\\theta^{ind}_i + W_m m_i^{(t)}$$

self-attention (利き・関係性バイアスを加算):

$$e_{ij} = \\frac{(x_i W_Q)(x_j W_K)^\\top}{\\sqrt{d}} + B_{利き}(i,j) + r_{ij}$$

Phase 1 で有効なのは前半3項と $B_{利き}$ だけ。$\\theta^{ind}$ [B] と $m_i$ [A]、
$r_{ij}$ [C] は `configs/features.yaml` のフラグが立ってから合流する。
フラグが全て false のとき、このモジュールは「素の駒トークンTransformer」になる
(DESIGN.md 設計原則1: 面白さ機能を外しても強さの経路が残る)。

**手番トークン**: DESIGN.md §3(1) の式に手番は現れないが、ADR
(docs/decisions/2026-07-26-phase0-data-pipeline.md §2) の決定によりトークナイザは
先後の視点反転をしない。そのため手番を知る手段が他になく、全トークンに
$E_{turn}$ を加算している。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kokoro_shogi.config import FeatureFlags, ModelConfig
from kokoro_shogi.core.effects import ATTACKS_ENEMY, NO_EFFECT, SUPPORTS_ALLY
from kokoro_shogi.core.pieces import SPECIES_ORDER
from kokoro_shogi.core.tokenizer import NUM_POSITIONS

#: 駒種の種類数 (SPECIES_ORDER: 生駒8種 + 成駒6種)
NUM_SPECIES = len(SPECIES_ORDER)
#: 成り(2) × 所有(2) の組み合わせ
NUM_FLAGS = 4
#: 利き関係の種類数 (NO_EFFECT / ATTACKS_ENEMY / SUPPORTS_ALLY)
NUM_EFFECT_KINDS = max(NO_EFFECT, ATTACKS_ENEMY, SUPPORTS_ALLY) + 1


class PieceTokenEmbedding(nn.Module):
    """駒トークン列 → 埋め込み `(B, N, d_model)` (DESIGN.md §3(1))。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_model = config.d_model
        self.species = nn.Embedding(NUM_SPECIES, d_model)
        self.position = nn.Embedding(NUM_POSITIONS, d_model)
        self.flag = nn.Embedding(NUM_FLAGS, d_model)
        self.turn = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        species: Tensor,
        position: Tensor,
        owner: Tensor,
        promoted: Tensor,
        turn: Tensor,
    ) -> Tensor:
        flag = promoted * 2 + owner
        embedded = (
            self.species(species)
            + self.position(position)
            + self.flag(flag)
            + self.turn(turn).unsqueeze(1)
        )
        return self.norm(embedded)


class EffectAttentionBias(nn.Module):
    """利き関係行列 → attentionへの加算バイアス $B_{利き}(i,j)$ (DESIGN.md §3(2))。

    ヘッドごとに別の重みを学習する (Lc0 の学習可能attentionバイアスと同じ入れ方)。
    初期値0なので、学習開始時点では素のattentionと完全に一致する。
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.embedding = nn.Embedding(NUM_EFFECT_KINDS, n_heads)
        nn.init.zeros_(self.embedding.weight)

    @property
    def weight(self) -> Tensor:
        """利き種別 × ヘッド の重み `(3, n_heads)`。"""
        return self.embedding.weight

    def forward(self, effect: Tensor) -> Tensor:
        """`effect` `(B, N, N)` → `(B, n_heads, N, N)`。"""
        return self.embedding(effect).permute(0, 3, 1, 2)


class KokoroEncoderLayer(nn.Module):
    """pre-norm Transformer層 (self-attention + FFN)。

    `nn.TransformerEncoderLayer` を使わず自前で持っている理由:
    PyTorch 2.13 の `nn.TransformerEncoderLayer` は `norm_first=True` のとき
    **非ゼロの float `src_mask` を渡すと出力が NaN になる** (bool マスクとして
    解釈され全キーが塞がれる)。加算バイアスは全要素に同じ定数を足しても
    softmax で打ち消えるはずなので、これは本来あり得ない挙動。
    `nn.MultiheadAttention` を直接呼ぶ経路では正しく加算される。

    DESIGN.md §3(2) の $B_{利き}$ と Phase 3 の $r_{ij}$ [C] はどちらも
    「attentionスコアへの加算」なので、この経路を自前で持っておく必要がある。
    回帰テスト: `tests/test_trunk.py::test_nonzero_attention_bias_stays_finite`
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        normed = self.norm1(x)
        attended, _ = self.attention(
            normed, normed, normed, attn_mask=attn_mask, need_weights=False
        )
        x = x + attended
        return x + self.feed_forward(self.norm2(x))


class KokoroTrunk(nn.Module):
    """駒トークンを文脈化する共有エンコーダ。

    `forward` は `(B, N, d_model)` の駒表現 $h_i$ を返す。
    無効トークン (mask=False) の行は0で潰してある。
    """

    def __init__(
        self, config: ModelConfig | None = None, features: FeatureFlags | None = None
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.features = features or FeatureFlags()

        d_model = self.config.d_model
        self.embedding = PieceTokenEmbedding(self.config)
        self.effect_bias = EffectAttentionBias(self.config.n_heads)

        # 感情の合流点 W_m m_i [A] (DESIGN.md §3(1))。フラグOFF時はモジュール自体を
        # 作らない (state_dict を Phase 1-2 のチェックポイントと同一に保つため)。
        # 零初期化なので、ONにした直後もフラグOFFと完全に同じ出力から学習が始まる。
        if self.features.mood:
            self.mood_proj = nn.Linear(self.config.d_mood, d_model, bias=False)
            nn.init.zeros_(self.mood_proj.weight)

        self.layers = nn.ModuleList(
            KokoroEncoderLayer(d_model, self.config.n_heads, d_model * 4)
            for _ in range(self.config.n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        species: Tensor,
        position: Tensor,
        owner: Tensor,
        promoted: Tensor,
        mask: Tensor,
        turn: Tensor,
        effect: Tensor | None = None,
        mood: Tensor | None = None,
    ) -> Tensor:
        batch, tokens = species.shape
        hidden = self.embedding(species, position, owner, promoted, turn)
        if mood is not None and self.features.mood:
            hidden = hidden + self.mood_proj(mood)

        bias = self._attention_bias(effect, mask, batch, tokens, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=bias)
        hidden = self.norm(hidden)

        return hidden * mask.unsqueeze(-1)

    def _attention_bias(
        self, effect: Tensor | None, mask: Tensor, batch: int, tokens: int, dtype: torch.dtype
    ) -> Tensor:
        """加算attentionマスク `(B * n_heads, N, N)` を組み立てる。

        無効トークンは `key_padding_mask` ではなくここへ畳み込む。float の attn_mask と
        bool の key_padding_mask を同時に渡すと PyTorch が警告を出すため。
        """
        heads = self.config.n_heads

        if effect is not None:
            bias = self.effect_bias(effect).to(dtype)
        else:
            bias = torch.zeros(batch, heads, tokens, tokens, dtype=dtype, device=mask.device)

        # 無効トークンは「参照される側 (key)」だけを塞ぐ。query側を塞がないので
        # 全キーが -inf になる行は生まれず、softmax が NaN にならない。
        blocked = torch.zeros_like(bias)
        blocked.masked_fill_(~mask[:, None, None, :], float("-inf"))

        return (bias + blocked).reshape(batch * heads, tokens, tokens)


__all__ = [
    "NUM_EFFECT_KINDS",
    "NUM_FLAGS",
    "NUM_SPECIES",
    "EffectAttentionBias",
    "KokoroEncoderLayer",
    "KokoroTrunk",
    "PieceTokenEmbedding",
]
