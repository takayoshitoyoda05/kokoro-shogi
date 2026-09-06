"""利き関係を抽出し、attentionバイアスの素材を作る。

DESIGN.md §3(2) の $B_{利き}(i,j)$ は「駒 $i$ が駒 $j$ に利いているか」を
attention のスコアに加算するもの。cshogi は利き (bitboard) をPythonへ公開して
いないため、駒の動きからここで生成する。

用語:
- **利き**: その駒が次に到達できるマス。自駒がいるマスも含む (= 紐が付いている)
- **紐**: 味方の駒に利きが通っている状態 (取り返せる)

盤の座標は cshogi のマス番号 (`筋index * 9 + 段index`)。先手 (owner=0) の前進は
段が減る方向 (7七 → 7六)。将棋の駒はすべて左右対称なので、後手の利きは段方向を
反転するだけで得られる。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kokoro_shogi.core.pieces import BLACK, Species, piece_code_to_owner, piece_code_to_species
from kokoro_shogi.core.squares import NUM_SQUARES

if TYPE_CHECKING:
    import cshogi

#: (筋方向, 段方向)。先手視点で段が減る向きが前進。
_Direction = tuple[int, int]

_FORWARD = (0, -1)
_BACK = (0, 1)
_LEFT = (-1, 0)
_RIGHT = (1, 0)
_FORWARD_LEFT = (-1, -1)
_FORWARD_RIGHT = (1, -1)
_BACK_LEFT = (-1, 1)
_BACK_RIGHT = (1, 1)

_GOLD_STEPS: tuple[_Direction, ...] = (
    _FORWARD, _FORWARD_LEFT, _FORWARD_RIGHT, _LEFT, _RIGHT, _BACK,
)
_KING_STEPS: tuple[_Direction, ...] = (
    _FORWARD, _BACK, _LEFT, _RIGHT, _FORWARD_LEFT, _FORWARD_RIGHT, _BACK_LEFT, _BACK_RIGHT,
)
_BISHOP_SLIDES: tuple[_Direction, ...] = (
    _FORWARD_LEFT, _FORWARD_RIGHT, _BACK_LEFT, _BACK_RIGHT,
)
_ROOK_SLIDES: tuple[_Direction, ...] = (_FORWARD, _BACK, _LEFT, _RIGHT)

#: 1マスだけ動く利き (先手視点)
STEP_MOVES: dict[Species, tuple[_Direction, ...]] = {
    "FU": (_FORWARD,),
    "KE": ((-1, -2), (1, -2)),
    "GI": (_FORWARD, _FORWARD_LEFT, _FORWARD_RIGHT, _BACK_LEFT, _BACK_RIGHT),
    "KI": _GOLD_STEPS,
    "OU": _KING_STEPS,
    "TO": _GOLD_STEPS,
    "NY": _GOLD_STEPS,
    "NK": _GOLD_STEPS,
    "NG": _GOLD_STEPS,
    "UM": _ROOK_SLIDES,   # 馬は角の走り + 王の縦横1マス
    "RY": _BISHOP_SLIDES,  # 竜は飛の走り + 王の斜め1マス
    "KY": (),
    "KA": (),
    "HI": (),
}

#: 走る利き (先手視点)
SLIDE_MOVES: dict[Species, tuple[_Direction, ...]] = {
    "KY": (_FORWARD,),
    "KA": _BISHOP_SLIDES,
    "HI": _ROOK_SLIDES,
    "UM": _BISHOP_SLIDES,
    "RY": _ROOK_SLIDES,
    "FU": (),
    "KE": (),
    "GI": (),
    "KI": (),
    "OU": (),
    "TO": (),
    "NY": (),
    "NK": (),
    "NG": (),
}


def _shift(square: int, direction: _Direction, owner: int) -> int | None:
    """マスを1つ動かす。盤外なら None。後手は段方向を反転する。"""
    file_index, rank_index = divmod(square, 9)
    delta_file, delta_rank = direction
    if owner != BLACK:
        delta_rank = -delta_rank

    file_index += delta_file
    rank_index += delta_rank
    if not (0 <= file_index < 9 and 0 <= rank_index < 9):
        return None
    return file_index * 9 + rank_index


def attacks_from(species: Species, owner: int, square: int, occupied: frozenset[int]) -> list[int]:
    """1枚の駒の利きマスを返す。自駒のいるマス (紐) も含む。

    `occupied` は盤上の全駒のマス集合 (走り駒の遮蔽判定に使う)。
    """
    targets: list[int] = []

    for direction in STEP_MOVES[species]:
        destination = _shift(square, direction, owner)
        if destination is not None:
            targets.append(destination)

    for direction in SLIDE_MOVES[species]:
        current = square
        while True:
            current = _shift(current, direction, owner)  # type: ignore[assignment]
            if current is None:
                break
            targets.append(current)
            if current in occupied:  # 駒に当たったらそこで止まる
                break

    return targets


def board_occupancy(board: cshogi.Board) -> frozenset[int]:
    """盤上に駒があるマスの集合。"""
    return frozenset(square for square in range(NUM_SQUARES) if board.piece(square) != 0)


def attack_lists(board: cshogi.Board) -> dict[int, list[int]]:
    """盤上の各駒について、その駒の利きマス一覧を返す (キーは駒のいるマス)。"""
    occupied = board_occupancy(board)
    result: dict[int, list[int]] = {}
    for square in occupied:
        code = board.piece(square)
        species = piece_code_to_species(code)
        owner = piece_code_to_owner(code)
        result[square] = attacks_from(species, owner, square, occupied)
    return result


def attack_counts(board: cshogi.Board) -> np.ndarray:
    """各マスへの利き数を `(2, 81)` の配列で返す (添字は所有者)。"""
    counts = np.zeros((2, NUM_SQUARES), dtype=np.int8)
    for square, targets in attack_lists(board).items():
        owner = piece_code_to_owner(board.piece(square))
        for target in targets:
            counts[owner][target] += 1
    return counts


def is_attacked_by(board: cshogi.Board, square: int, owner: int) -> bool:
    """`square` に `owner` の利きがあるか。"""
    return bool(attack_counts(board)[owner][square])


#: piece_effect_matrix の値。attentionバイアスの埋め込み索引として使う
NO_EFFECT = 0
ATTACKS_ENEMY = 1  # i が敵駒 j に利いている (取れる)
SUPPORTS_ALLY = 2  # i が味方 j に利いている (紐)


def piece_effect_matrix(board: cshogi.Board, squares: list[int]) -> np.ndarray:
    """駒トークンの並び `squares` に対する利き関係行列 `(N, N)` を返す。

    `matrix[i][j]` は駒 i から駒 j への関係 (NO_EFFECT / ATTACKS_ENEMY / SUPPORTS_ALLY)。
    DESIGN.md §3(2) の attention バイアス $B_{利き}(i,j)$ の素材になる。
    持ち駒のトークンは盤上にいないので、`squares` に -1 を入れておけば全て NO_EFFECT になる。
    """
    size = len(squares)
    matrix = np.zeros((size, size), dtype=np.int8)

    square_to_index = {square: index for index, square in enumerate(squares) if square >= 0}
    lists = attack_lists(board)

    for index, square in enumerate(squares):
        if square < 0:
            continue
        owner = piece_code_to_owner(board.piece(square))
        for target in lists.get(square, ()):
            target_index = square_to_index.get(target)
            if target_index is None:
                continue
            target_owner = piece_code_to_owner(board.piece(target))
            matrix[index][target_index] = (
                SUPPORTS_ALLY if target_owner == owner else ATTACKS_ENEMY
            )

    return matrix
