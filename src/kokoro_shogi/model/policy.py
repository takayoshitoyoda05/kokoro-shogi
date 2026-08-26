"""trunk と各ヘッドを束ねる方策本体 (DESIGN.md §3(7))。

$$\\pi(a\\mid s_t) = \\frac{\\exp(s_{i,a}/\\tau)\\cdot\\mathbb{1}[a\\in\\mathcal{A}_i]}
                        {\\sum_j\\sum_{a'\\in\\mathcal{A}_j}\\exp(s_{j,a'}/\\tau)}$$

非合法手は $-10^9$ に置換してから softmax する。**全駒・全手をまたいだ1つの softmax**
であることが要点で (駒ごとに正規化しない)、これが DESIGN.md の言う
「微分可能な調停」＝どの駒の言い分が通るかまで含めて1つの分布になる、という設計。

温度 $\\tau$ は `configs/base.yaml` の `loss.tau`。学習時 1.0、対局時 0.1 前後。

Phase 1 の手スコアは `heads.PolicyHead` の素の src-dst スコア。Phase 2 で
$s_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$ に差し替わるが、
このクラスの外から見た形 (logits / value / マスク処理) は変わらない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kokoro_shogi.config import Config, FeatureFlags, ModelConfig, load_config
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.core.tokenizer import MAX_PIECES
from kokoro_shogi.model.heads import (
    NUM_PROMOTE,
    DesireHead,
    FreeTermHead,
    MonotonicValueMixing,
    PersonalityWeights,
    PolicyHead,
    ValueHead,
)
from kokoro_shogi.model.trunk import NUM_SPECIES, KokoroTrunk

#: 非合法手に入れる値 (DESIGN.md §3(7))
ILLEGAL_LOGIT = -1e9

#: 手スコアの作り方。plain = 素のsrc-dst (Phase 1) / desire = 欲求分解 (Phase 2)
HEAD_KINDS = ("plain", "desire")


@dataclass(frozen=True)
class PolicyOutput:
    """方策の出力。

    `logits` は `(B, N * 81 * 2)` に平坦化してある (data/dataset.action_index と同じ並び)。
    """

    logits: Tensor
    value: Tensor
    hidden: Tensor
    #: 欲求 $d_i(a)$ `(B, N, 162, 6)` — head="desire" のときだけ
    desire: Tensor | None = None
    #: 性格重み $w_i$ `(B, N, 6)` — 全て正
    personality: Tensor | None = None
    #: 自由項 $g_i(a)$ `(B, N, 162)` — |g| が捨て駒検出器
    free_term: Tensor | None = None
    #: 発言力 $\\alpha_i$ `(B, N)` — 合計1に正規化済み (INTERFACE.md の alpha)
    alpha: Tensor | None = None
    #: 駒ごとの価値 $V_i$ `(B, N)`
    piece_value: Tensor | None = None

    def log_probs(self, tau: float = 1.0) -> Tensor:
        """温度付きの対数確率 `(B, A)`。"""
        return torch.log_softmax(self.logits / tau, dim=-1)

    def probs(self, tau: float = 1.0) -> Tensor:
        return torch.softmax(self.logits / tau, dim=-1)

    def best_action(self) -> Tensor:
        """`argmax \\pi` の action index `(B,)`。"""
        return self.logits.argmax(dim=-1)


class KokoroPolicy(nn.Module):
    """駒トークン → 指し手分布と状態価値。

    `head` で手スコアの作り方を切り替える:

    - `"plain"` (Phase 1): 素の src-dst スコア。Gate1 はこれで測る
    - `"desire"` (Phase 2): $s_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$。
      $\\lambda_g$ 掃引・解釈可能性の評価・Unityへ送る内面データはこちら

    機能フラグ (`configs/features.yaml`) が全て false のときが「素の強い将棋AI」構成
    (DESIGN.md 設計原則1)。
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        features: FeatureFlags | None = None,
        *,
        tau: float = 1.0,
        head: str = "plain",
        max_pieces: int = MAX_PIECES,
    ) -> None:
        super().__init__()
        if head not in HEAD_KINDS:
            raise ValueError(f"head は {HEAD_KINDS} のいずれかです: {head}")

        self.model_config = config or ModelConfig()
        self.features = features or FeatureFlags()
        self.tau = tau
        self.head = head

        self.trunk = KokoroTrunk(self.model_config, self.features)

        if head == "plain":
            self.policy_head = PolicyHead(self.model_config)
            self.value_head = ValueHead(self.model_config)
        else:
            self.desire_head = DesireHead(self.model_config)
            self.free_term_head = FreeTermHead(self.model_config)
            self.personality = PersonalityWeights(
                self.model_config,
                num_species=NUM_SPECIES,
                # 感情 [A] は性格重みにも合流する ($w_i$ が対局中に動く仕組み)
                d_mood=self.model_config.d_mood if self.features.mood else 0,
            )
            self.mixing = MonotonicValueMixing(self.model_config, max_pieces=max_pieces)

    @classmethod
    def from_config(cls, config: Config | None = None, *, head: str = "plain") -> KokoroPolicy:
        """`configs/*.yaml` から組み立てる。"""
        config = config or load_config()
        return cls(config.model, config.features, tau=config.loss.tau, head=head)

    def forward(
        self,
        species: Tensor,
        position: Tensor,
        owner: Tensor,
        promoted: Tensor,
        mask: Tensor,
        turn: Tensor,
        effect: Tensor | None = None,
        legal: Tensor | None = None,
        mood: Tensor | None = None,
    ) -> PolicyOutput:
        hidden = self.trunk(species, position, owner, promoted, mask, turn, effect, mood)

        if self.head == "plain":
            scores = self.policy_head(hidden)
            extra: dict[str, Tensor] = {}
            value = self.value_head(hidden, mask)
        else:
            scores, extra, value = self._desire_scores(hidden, species, mask, mood)

        scores = scores.masked_fill(~mask[:, :, None, None], ILLEGAL_LOGIT)
        if legal is not None:
            scores = scores.masked_fill(~legal, ILLEGAL_LOGIT)

        return PolicyOutput(
            logits=scores.flatten(start_dim=1),
            value=value,
            hidden=hidden,
            **extra,
        )

    def _desire_scores(
        self, hidden: Tensor, species: Tensor, mask: Tensor, mood: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """$s_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$ (DESIGN.md §3(6) 初期スコア)。"""
        desire = self.desire_head(hidden)  # (B, N, 162, 6)
        free_term = self.free_term_head(hidden)  # (B, N, 162)
        weights = self.personality(
            species, mood=mood if self.features.mood else None
        )  # (B, N, 6)

        explained = (desire * weights.unsqueeze(2)).sum(dim=-1)  # (B, N, 162)
        scores = explained + free_term

        value, piece_value, alpha = self.mixing(hidden, mask)

        batch, tokens = species.shape
        return (
            scores.view(batch, tokens, NUM_SQUARES, NUM_PROMOTE),
            {
                "desire": desire,
                "personality": weights,
                "free_term": free_term,
                "alpha": alpha,
                "piece_value": piece_value,
            },
            value,
        )

    def forward_batch(self, batch: dict[str, Any], *, use_legal: bool = True) -> PolicyOutput:
        """`data/dataset.collate` が作った辞書をそのまま食う。"""
        return self(
            species=batch["species"],
            position=batch["position"],
            owner=batch["owner"],
            promoted=batch["promoted"],
            mask=batch["mask"],
            turn=batch["turn"],
            effect=batch.get("effect"),
            legal=batch.get("legal") if use_legal else None,
            mood=batch.get("mood"),
        )


__all__ = ["HEAD_KINDS", "ILLEGAL_LOGIT", "KokoroPolicy", "PolicyOutput"]
