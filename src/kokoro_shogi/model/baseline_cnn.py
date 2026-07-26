"""Gate1 の比較対象になる CNN ベースライン (dlshogi 流の ResNet)。

DESIGN.md Phase 1 の Gate は「一致率が **CNNベースライン比 -2%以内**」。
比較対象を自前で持たないと判定できないので、同じデータ・同じ損失・同じ行動空間で
学習できる CNN をここに置く。

**何を統制しているか**: 違うのは trunk だけ (駒トークンTransformer vs 盤面CNN)。
方策ヘッド・価値ヘッド・合法手マスク・損失・一致率の計算は `KokoroPolicy` と共有する。
Gate1 が問うているのは「駒をトークンにして大丈夫か」なので、ここを揃えないと
ヘッドの差が混ざって何を測ったのか分からなくなる。

そのため CNN も **駒トークン単位で** 手スコアを出す:
盤面を畳み込んだ特徴マップから、各駒トークンがいるマスの特徴ベクトルを取り出して
(持ち駒トークンは学習可能な「駒台ベクトル」を使う) 共通の `PolicyHead` に渡す。

入力プレーン (9×9 × 43):

- 盤上の駒: 駒種14 × 所有2 = 28面
- 持ち駒: 生駒7種 × 所有2 = 14面 (枚数を18で正規化した定数面)
- 手番: 1面 (トークナイザが視点反転しないため。trunk.py の注記と同じ理由)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kokoro_shogi.config import Config, ModelConfig, load_config
from kokoro_shogi.core.pieces import (
    HAND_INDEX_TO_SPECIES,
    SPECIES_ORDER,
    SPECIES_TO_HAND_INDEX,
    base_species,
)
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.model.heads import PolicyHead, ValueHead
from kokoro_shogi.model.policy import ILLEGAL_LOGIT, PolicyOutput

NUM_SPECIES = len(SPECIES_ORDER)
NUM_HAND_KINDS = len(HAND_INDEX_TO_SPECIES)
#: 盤上 28面 + 持ち駒 14面 + 手番 1面
NUM_PLANES = NUM_SPECIES * 2 + NUM_HAND_KINDS * 2 + 1
#: 持ち駒の枚数を正規化する定数 (歩が最大18枚)
HAND_SATURATION = 18.0

#: 駒種index → 持ち駒スロット (0-6)。玉は持ち駒にならないので -1
_HAND_SLOT = [
    SPECIES_TO_HAND_INDEX.get(base_species(species), -1) for species in SPECIES_ORDER
]


def build_planes(
    species: Tensor, position: Tensor, owner: Tensor, mask: Tensor, turn: Tensor
) -> Tensor:
    """駒トークン → CNN入力プレーン `(B, 43, 9, 9)`。

    盤面を作り直さずトークンから直接組むので、`data/dataset.py` の出力をそのまま使える。
    """
    batch = species.shape[0]
    device = species.device
    planes = torch.zeros(batch, NUM_PLANES, NUM_SQUARES, device=device)

    hand_slot = torch.tensor(_HAND_SLOT, device=device, dtype=torch.long)

    on_board = mask & (position < NUM_SQUARES)
    in_hand = mask & (position >= NUM_SQUARES)

    batch_index = torch.arange(batch, device=device).unsqueeze(1).expand_as(species)

    # 盤上: (駒種, 所有) の面のそのマスを1にする
    board_plane = species * 2 + owner
    planes[
        batch_index[on_board],
        board_plane[on_board],
        position[on_board],
    ] = 1.0

    # 持ち駒: (生駒, 所有) の面を枚数ぶん加算し、定数面として全マスに広げる
    hand_counts = torch.zeros(batch, NUM_HAND_KINDS * 2, device=device)
    slot = hand_slot[species] * 2 + owner
    valid = in_hand & (hand_slot[species] >= 0)
    hand_counts.index_put_(
        (batch_index[valid], slot[valid]),
        torch.ones(int(valid.sum()), device=device),
        accumulate=True,
    )
    hand_offset = NUM_SPECIES * 2
    planes[:, hand_offset : hand_offset + NUM_HAND_KINDS * 2, :] = (
        hand_counts / HAND_SATURATION
    ).unsqueeze(-1)

    planes[:, -1, :] = turn.to(planes.dtype).unsqueeze(-1)

    return planes.view(batch, NUM_PLANES, 9, 9)


class ResidualBlock(nn.Module):
    """conv3x3 → BN → ReLU を2つ重ねた残差ブロック (dlshogi と同形)。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.body(x))


class BaselineCNN(nn.Module):
    """盤面CNN trunk + `KokoroPolicy` と共通のヘッド。

    出力は `PolicyOutput` なので、学習ループ・一致率の計算をそのまま共有できる。
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        channels: int = 192,
        blocks: int = 10,
    ) -> None:
        super().__init__()
        self.model_config = config or ModelConfig()
        d_model = self.model_config.d_model

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.to_token = nn.Linear(channels, d_model)
        #: 持ち駒トークンは盤上に対応するマスがないので、学習可能なベクトルを使う
        self.hand_token = nn.Parameter(torch.zeros(d_model))

        self.policy_head = PolicyHead(self.model_config)
        self.value_head = ValueHead(self.model_config)

    @classmethod
    def from_config(cls, config: Config | None = None, **kwargs: int) -> BaselineCNN:
        config = config or load_config()
        return cls(config.model, **kwargs)

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
    ) -> PolicyOutput:
        del promoted, effect  # 盤面プレーンに畳み込み済み / CNNは利き行列を使わない

        planes = build_planes(species, position, owner, mask, turn)
        feature = self.blocks(self.stem(planes))  # (B, C, 9, 9)
        feature = feature.flatten(2).transpose(1, 2)  # (B, 81, C)

        hidden = self._gather_tokens(feature, position, mask)

        scores = self.policy_head(hidden)
        scores = scores.masked_fill(~mask[:, :, None, None], ILLEGAL_LOGIT)
        if legal is not None:
            scores = scores.masked_fill(~legal, ILLEGAL_LOGIT)

        return PolicyOutput(
            logits=scores.flatten(start_dim=1),
            value=self.value_head(hidden, mask),
            hidden=hidden,
        )

    def _gather_tokens(self, feature: Tensor, position: Tensor, mask: Tensor) -> Tensor:
        """各駒トークンがいるマスの特徴を取り出す `(B, N, d_model)`。"""
        on_board = mask & (position < NUM_SQUARES)
        square = position.clamp(max=NUM_SQUARES - 1)

        gathered = torch.gather(
            feature, 1, square.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        )
        hidden = self.to_token(gathered)

        hidden = torch.where(on_board.unsqueeze(-1), hidden, self.hand_token)
        return hidden * mask.unsqueeze(-1)

    def forward_batch(self, batch: dict[str, Tensor], *, use_legal: bool = True) -> PolicyOutput:
        return self(
            species=batch["species"],
            position=batch["position"],
            owner=batch["owner"],
            promoted=batch["promoted"],
            mask=batch["mask"],
            turn=batch["turn"],
            legal=batch.get("legal") if use_legal else None,
        )


__all__ = ["HAND_SATURATION", "NUM_PLANES", "BaselineCNN", "ResidualBlock", "build_planes"]
