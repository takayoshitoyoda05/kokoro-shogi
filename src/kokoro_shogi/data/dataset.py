"""Phase 0 が作った npz シャードを PyTorch の Dataset にする (Phase 1 蒸留の入口)。

`scripts/make_labels.py` が書いた 1局面 = 1行のシャードを読み、モデルが食える形へ
組み立て直す。ADR (docs/decisions/2026-07-26-phase0-data-pipeline.md §5) の決定どおり
**利き行列はシャードに入っていない**ので、ここでトークンから盤面を復元して計算する。

1局面あたりここで作るもの:

- 駒トークン (species / position / owner / promoted / mask / turn) … シャードそのまま
- 利き関係行列 `(40, 40)` … DESIGN.md §3(2) の attention バイアス $B_{利き}$ の素材
- 合法手マスク `(40, 81, 2)` … DESIGN.md §3(7) の $\\mathbb{1}[a \\in \\mathcal{A}_i]$
- 教師 (move_token, move_to, move_promote) と勝敗 z、欲求ラベル6軸

**行動空間**: 手は「どの駒トークンが」「どのマスへ」「成るか」の3つ組で表す。
$40 \\times 81 \\times 2 = 6480$ 通り。DESIGN.md §5 出典表の「src-dst方策ヘッド」に対応する。

**打ちの駒の選び方**: 同じ駒種の持ち駒が複数あるとき、どのトークンが動いたかは
一意に決まらない。`core/piece_state.py` の `_take_from_hand` が piece_id 順で最初の
1枚を選ぶので、ここでも同じ規則 (トークンは piece_id 昇順なので index 最小) を使う。
この2箇所がずれると教師の move_token が合法手マスクの外に出るため、
`tests/test_dataset.py` で全局面の整合を検証している。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cshogi
import numpy as np

from kokoro_shogi.core.effects import piece_effect_matrix
from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES, SPECIES_ORDER, base_species
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.core.tokenizer import MAX_PIECES, NUM_POSITIONS, TokenizedPosition, to_sfen
from kokoro_shogi.data.labels import NUM_AXES

#: 手の成分数 (移動先81 × 成る/成らない2)
NUM_MOVE_TO = NUM_SQUARES
NUM_PROMOTE = 2
#: 方策の出力次元 40 × 81 × 2
NUM_ACTIONS = MAX_PIECES * NUM_MOVE_TO * NUM_PROMOTE

#: シャードに入っている列 (make_labels.py の ShardBuilder.add と対応)
SHARD_COLUMNS = (
    "species",
    "position",
    "owner",
    "promoted",
    "mask",
    "turn",
    "move",
    "move_token",
    "move_to",
    "move_promote",
    "result",
    "labels",
    "game_index",
)


def action_index(move_token: int, move_to: int, move_promote: int) -> int:
    """(駒トークン, 移動先, 成り) → 方策の出力index。"""
    return (move_token * NUM_MOVE_TO + move_to) * NUM_PROMOTE + move_promote


def decode_action(index: int) -> tuple[int, int, int]:
    """方策の出力index → (駒トークン, 移動先, 成り)。"""
    move_promote = index % NUM_PROMOTE
    rest = index // NUM_PROMOTE
    return rest // NUM_MOVE_TO, rest % NUM_MOVE_TO, move_promote


@dataclass(frozen=True)
class Position:
    """1局面ぶんの学習サンプル (numpy)。"""

    species: np.ndarray  # (40,) int64
    position: np.ndarray  # (40,) int64
    owner: np.ndarray  # (40,) int64
    promoted: np.ndarray  # (40,) int64
    mask: np.ndarray  # (40,) bool
    turn: int
    effect: np.ndarray  # (40, 40) int64
    legal: np.ndarray  # (40, 81, 2) bool
    action: int  # 教師手の action index
    result: int  # 勝敗 z (手番側視点) ∈ {-1, 0, +1}
    labels: np.ndarray  # (40, 6) float32 欲求ラベル


def tokens_to_board(
    species: np.ndarray,
    position: np.ndarray,
    owner: np.ndarray,
    promoted: np.ndarray,
    mask: np.ndarray,
    turn: int,
) -> cshogi.Board:
    """トークン列から cshogi の盤面を復元する。

    SFENの4要素目 (手数) は合法手生成に影響しないので 1 を入れる。
    """
    tokens = TokenizedPosition(
        species=species,
        position=position,
        owner=owner,
        promoted=promoted,
        mask=mask,
        turn=int(turn),
        move_number=1,
    )
    return cshogi.Board(to_sfen(tokens))


def legal_move_mask(
    board: cshogi.Board,
    position: np.ndarray,
    owner: np.ndarray,
    species: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """合法手マスク `(40, 81, 2)` を作る。

    盤上の駒は「マス → トークン」で、打ちは「駒種 → その駒種の持ち駒トークンのうち
    index 最小のもの」で対応付ける (piece_state.PieceIdTracker._take_from_hand と同じ規則)。
    """
    legal = np.zeros((MAX_PIECES, NUM_MOVE_TO, NUM_PROMOTE), dtype=bool)

    square_to_token: dict[int, int] = {}
    hand_token: dict[str, int] = {}
    turn = int(board.turn)

    for index in range(MAX_PIECES):
        if not mask[index]:
            continue
        square = int(position[index])
        if square < NUM_SQUARES:
            square_to_token[square] = index
        elif int(owner[index]) == turn:
            held = base_species(SPECIES_ORDER[int(species[index])])
            # トークンは piece_id 昇順なので、最初に見つかったものが最小index
            hand_token.setdefault(held, index)

    for move in board.legal_moves:
        to_square = cshogi.move_to(move)
        promote = int(cshogi.move_is_promotion(move))

        if cshogi.move_is_drop(move):
            held = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
            token = hand_token.get(held)
        else:
            token = square_to_token.get(cshogi.move_from(move))

        if token is not None:
            legal[token, to_square, promote] = True

    return legal


class ShardDataset:
    """npz シャード群を読み込む Dataset。

    シャードは丸ごとメモリに載せる (1局面 約453B なので 1M局面で約450MB)。
    `max_positions` で上限を切れる。

    `with_legal` / `with_effect` を False にすると盤面の復元を省いて高速になるが、
    合法手マスクなしの方策は DESIGN.md §3(7) を満たさないので評価には使わないこと。
    """

    def __init__(
        self,
        paths: list[Path],
        *,
        max_positions: int | None = None,
        with_legal: bool = True,
        with_effect: bool = True,
    ) -> None:
        if not paths:
            raise ValueError("シャードが1つも指定されていません。")

        self.with_legal = with_legal
        self.with_effect = with_effect

        columns: dict[str, list[np.ndarray]] = {name: [] for name in SHARD_COLUMNS}
        total = 0
        for path in paths:
            with np.load(path) as shard:
                missing = set(SHARD_COLUMNS) - set(shard.files)
                if missing:
                    raise ValueError(f"{path}: 列が足りません: {sorted(missing)}")
                take = len(shard["move_to"])
                if max_positions is not None:
                    take = min(take, max_positions - total)
                for name in SHARD_COLUMNS:
                    columns[name].append(shard[name][:take])
            total += take
            if max_positions is not None and total >= max_positions:
                break

        self._columns = {name: np.concatenate(values) for name, values in columns.items()}
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Position:
        column = self._columns
        species = column["species"][index].astype(np.int64)
        position = column["position"][index].astype(np.int64)
        owner = column["owner"][index].astype(np.int64)
        promoted = column["promoted"][index].astype(np.int64)
        mask = column["mask"][index]
        turn = int(column["turn"][index])

        effect = np.zeros((MAX_PIECES, MAX_PIECES), dtype=np.int64)
        legal = np.zeros((MAX_PIECES, NUM_MOVE_TO, NUM_PROMOTE), dtype=bool)

        if self.with_legal or self.with_effect:
            board = tokens_to_board(species, position, owner, promoted, mask, turn)
            if self.with_effect:
                squares = [
                    int(square) if flag and square < NUM_SQUARES else -1
                    for square, flag in zip(position, mask, strict=True)
                ]
                effect = piece_effect_matrix(board, squares).astype(np.int64)
            if self.with_legal:
                legal = legal_move_mask(board, position, owner, species, mask)

        return Position(
            species=species,
            position=position,
            owner=owner,
            promoted=promoted,
            mask=mask,
            turn=turn,
            effect=effect,
            legal=legal,
            action=action_index(
                int(column["move_token"][index]),
                int(column["move_to"][index]),
                int(column["move_promote"][index]),
            ),
            result=int(column["result"][index]),
            labels=column["labels"][index].astype(np.float32) / 255.0,
        )

    def __iter__(self) -> Iterator[Position]:
        for index in range(len(self)):
            yield self[index]


def collate(batch: list[Position]) -> dict[str, Any]:
    """Position のリストを torch のバッチテンソルにまとめる。

    torch はここでだけ import する (前処理環境には torch が入っていないため)。
    """
    import torch

    stack = np.stack
    return {
        "species": torch.from_numpy(stack([item.species for item in batch])),
        "position": torch.from_numpy(stack([item.position for item in batch])),
        "owner": torch.from_numpy(stack([item.owner for item in batch])),
        "promoted": torch.from_numpy(stack([item.promoted for item in batch])),
        "mask": torch.from_numpy(stack([item.mask for item in batch])),
        "turn": torch.tensor([item.turn for item in batch], dtype=torch.long),
        "effect": torch.from_numpy(stack([item.effect for item in batch])),
        "legal": torch.from_numpy(stack([item.legal for item in batch])),
        "action": torch.tensor([item.action for item in batch], dtype=torch.long),
        "result": torch.tensor([item.result for item in batch], dtype=torch.float32),
        "labels": torch.from_numpy(stack([item.labels for item in batch])),
    }


def find_shards(directory: Path | str) -> list[Path]:
    """ディレクトリ以下の shard_*.npz を名前順で集める。"""
    return sorted(Path(directory).rglob("shard_*.npz"))


def split_shards(paths: list[Path], val_ratio: float = 0.05) -> tuple[list[Path], list[Path]]:
    """シャード単位で train / val に分ける。

    局面単位ではなく**シャード単位**で分けるのは、同一局の局面が train と val の
    両方に入る (実質的なリーク) のを避けるため。シャードは棋譜順に詰められている。

    val を**先頭から**取るのは、最後のシャードが端数になりやすいため
    (1,000,069局面を10万ずつ詰めると最後は69局面しかなく、検証セットとして使えない)。
    どちらの端も「別の対局」であることに変わりはないので、満杯である先頭を使う。
    """
    if len(paths) < 2:
        raise ValueError("train/val に分けるにはシャードが2つ以上必要です。")
    val_count = max(1, round(len(paths) * val_ratio))
    val_count = min(val_count, len(paths) - 1)
    return paths[val_count:], paths[:val_count]


__all__ = [
    "MAX_PIECES",
    "NUM_ACTIONS",
    "NUM_AXES",
    "NUM_MOVE_TO",
    "NUM_POSITIONS",
    "NUM_PROMOTE",
    "Position",
    "ShardDataset",
    "action_index",
    "collate",
    "decode_action",
    "find_shards",
    "legal_move_mask",
    "split_shards",
    "tokens_to_board",
]
