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

#: 忠誠ハンデ [E] の強さ κ_E (DESIGN.md §3(9))。バリアント限定・棋力評価には使わない
KAPPA_LOYALTY = 0.5

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
    #: 会議 [D] の議事録 (ラウンドごとの top-k 提案)。会議OFF時は None
    council: tuple[Any, ...] | None = None

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
            if self.features.council:
                from kokoro_shogi.model.council import (
                    DEFAULT_ROUNDS,
                    DEFAULT_TOP_K,
                    ProposalEmbedding,
                )

                self.proposal_embed = ProposalEmbedding(self.model_config)
                self.council_rounds = DEFAULT_ROUNDS
                self.council_top_k = DEFAULT_TOP_K
                #: 学習時にラウンド数を引く集合 (空なら常に council_rounds)。
                #: train/mood_distill.py の --council-round-choices から設定する
                self.council_round_choices: tuple[int, ...] = ()
            self.personality = PersonalityWeights(
                self.model_config,
                num_species=NUM_SPECIES,
                # 感情 [A] は性格重みにも合流する ($w_i$ が対局中に動く仕組み)
                d_mood=self.model_config.d_mood if self.features.mood else 0,
                # 個体性格 [B] も同様 (DESIGN.md §3(4))
                d_individual=self.model_config.d_theta if self.features.individual else 0,
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
        relation: Tensor | None = None,
        rounds: int | None = None,
        individual: Tensor | None = None,
        loyalty: Tensor | None = None,
    ) -> PolicyOutput:
        hidden = self.trunk(
            species, position, owner, promoted, mask, turn, effect, mood, relation, individual
        )

        council = None
        if self.head == "plain":
            scores = self.policy_head(hidden)
            extra: dict[str, Tensor] = {}
            value = self.value_head(hidden, mask)
        else:
            scores, extra, value = self._desire_scores(
                hidden, species, mask, mood, individual, loyalty
            )
            if self.features.council:
                hidden, scores, extra, value, council = self._council(
                    hidden, scores, extra, value, species, mask, legal,
                    effect, mood, relation, rounds, individual, loyalty,
                )

        scores = self._mask_scores(scores, mask, legal)
        return PolicyOutput(
            logits=scores.flatten(start_dim=1),
            value=value,
            hidden=hidden,
            council=council,
            **extra,
        )

    @staticmethod
    def _mask_scores(scores: Tensor, mask: Tensor, legal: Tensor | None) -> Tensor:
        scores = scores.masked_fill(~mask[:, :, None, None], ILLEGAL_LOGIT)
        if legal is not None:
            scores = scores.masked_fill(~legal, ILLEGAL_LOGIT)
        return scores

    def _council(
        self,
        hidden: Tensor,
        scores: Tensor,
        extra: dict[str, Tensor],
        value: Tensor,
        species: Tensor,
        mask: Tensor,
        legal: Tensor | None,
        effect: Tensor | None,
        mood: Tensor | None,
        relation: Tensor | None,
        rounds: int | None,
        individual: Tensor | None = None,
        loyalty: Tensor | None = None,
    ):
        """会議調停 [D] (DESIGN.md §3(6))。

        重み共有なので `rounds` は推論時に自由に変えられる (Gate の R=1..4 比較)。
        各ラウンドの提案は**そのラウンド開始時点のスコア** $s^{(r-1)}$ で選ぶ。

        **random loop sampling** (2026-09-19、STARS/ICML2026 由来): `council_round_choices`
        を設定すると、学習時 (`self.training`) に限りラウンド数をその集合から一様に引く。
        固定 R だけで学習すると R を変えたときに性能が崩れる (実測: R=2 で 46.16% がピーク、
        R=3 で 43.74%、R=4 で 40.26%)。学習中に R を散らすと、どの深さでも成立する
        潜在動力学を学ぶことが期待できる。`rounds` を明示した場合はそちらが優先される
        (評価時の R 掃引は従来どおり)。
        """
        from kokoro_shogi.model.council import CouncilRoundLog, top_proposals

        if rounds is not None:
            resolved = rounds
        elif self.training and self.council_round_choices:
            index = int(torch.randint(len(self.council_round_choices), (1,)))
            resolved = self.council_round_choices[index]
        else:
            resolved = self.council_rounds
        logs: list[CouncilRoundLog] = []
        for _ in range(resolved):
            masked = self._mask_scores(scores, mask, legal)
            token, move_kind, bid = top_proposals(
                masked.flatten(start_dim=2), self.council_top_k
            )
            proposals = self.proposal_embed(hidden, token, move_kind, bid)
            hidden = self.trunk.reapply_last_layers(
                hidden, proposals, mask, effect, relation
            )
            scores, extra, value = self._desire_scores(
                hidden, species, mask, mood, individual, loyalty
            )
            logs.append(CouncilRoundLog(token=token, move_kind=move_kind, bid=bid))

        return hidden, scores, extra, value, tuple(logs)

    def _desire_scores(
        self,
        hidden: Tensor,
        species: Tensor,
        mask: Tensor,
        mood: Tensor | None = None,
        individual: Tensor | None = None,
        loyalty: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """$s_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$ (DESIGN.md §3(6) 初期スコア)。

        忠誠 [E, バリアント]: `loyalty` `(B, N)` (寝返っていない駒は0を渡す) が
        来たら $d_i(a) \\leftarrow d_i(a)(1 - \\kappa_E\\,\\mathrm{loyalty}_i)$。
        """
        desire = self.desire_head(hidden)  # (B, N, 162, 6)
        if loyalty is not None and self.features.loyalty:
            desire = desire * (1.0 - KAPPA_LOYALTY * loyalty).clamp(min=0.0)[:, :, None, None]
        free_term = self.free_term_head(hidden)  # (B, N, 162)
        weights = self.personality(
            species,
            mood=mood if self.features.mood else None,
            individual=individual if self.features.individual else None,
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

    def forward_batch(
        self, batch: dict[str, Any], *, use_legal: bool = True, rounds: int | None = None
    ) -> PolicyOutput:
        """`data/dataset.collate` が作った辞書をそのまま食う。"""
        return self(
            rounds=rounds,
            species=batch["species"],
            position=batch["position"],
            owner=batch["owner"],
            promoted=batch["promoted"],
            mask=batch["mask"],
            turn=batch["turn"],
            effect=batch.get("effect"),
            legal=batch.get("legal") if use_legal else None,
            mood=batch.get("mood"),
            relation=batch.get("relation"),
            individual=batch.get("individual"),
            loyalty=batch.get("loyalty"),
        )


__all__ = ["HEAD_KINDS", "ILLEGAL_LOGIT", "KokoroPolicy", "PolicyOutput"]
