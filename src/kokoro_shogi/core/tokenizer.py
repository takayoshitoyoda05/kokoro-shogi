"""cshogiの盤面を駒トークンの特徴列へ変換する (DESIGN.md §3(1))。

$x_i^{(0)} = E_{sp}[c(i)] + E_{pos}[p(i)] + E_{flag}[成_i, 所有_i] + W_{ind}θ^{ind}_i + W_m m_i$

このモジュールが作るのは前半3項の **索引** まで。埋め込み行列 $E$ はモデル側
(model/trunk.py) が持つ。θ_ind と m_i は機能フラグ [B]/[A] が有効になってから
PieceState 経由で合流する。

設計上の約束:

- **駒は盤から消えない**。取られた駒は持ち駒になるので、本将棋のトークン数は
  常に40。持ち駒には専用の位置ID (81=先手の駒台 / 82=後手の駒台) を割り当てる。
- **トークンの並びは piece_id 順で固定する** (tracker を渡した場合)。並びが手ごとに
  変わると、駒に紐づく感情GRUの状態 [A] や個体性格 [B] を追えなくなるため。
- **先後の視点反転はここではやらない**。手番側を常に先手として学習させたい場合は、
  盤面のほうを `cshogi.Board(rotate_sfen(...))` で反転させてからトークン化する。
  そうしないと「トークン→SFEN の往復が元に戻る」性質が壊れる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.pieces import (
    BLACK,
    SPECIES_ORDER,
    SPECIES_TO_INDEX,
    SPECIES_TO_SFEN_LETTER,
    Species,
    base_species,
    piece_code_to_owner,
    piece_code_to_species,
)
from kokoro_shogi.core.squares import HAND_SQUARE, NUM_SQUARES, str_to_sq

if TYPE_CHECKING:
    import cshogi

#: 本将棋の駒は取られても持ち駒になるだけなので、常に40枚
MAX_PIECES = 40

#: 盤上81マス + 駒台2つ (先手/後手)
HAND_POSITION = (NUM_SQUARES, NUM_SQUARES + 1)
NUM_POSITIONS = NUM_SQUARES + 2

#: SFENの持ち駒の並び順 (飛 角 金 銀 桂 香 歩)
_HAND_ORDER: tuple[Species, ...] = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")


@dataclass(frozen=True)
class TokenizedPosition:
    """1局面ぶんの駒トークン列。

    配列はすべて長さ `MAX_PIECES` に0詰めしてあり、有効なトークンは `mask` が True。
    """

    #: 駒種índex 0-13 (SPECIES_ORDER の並び)
    species: np.ndarray
    #: 位置index 0-80 が盤上、81/82 が駒台
    position: np.ndarray
    #: 0=先手, 1=後手
    owner: np.ndarray
    #: 成っていれば1
    promoted: np.ndarray
    #: 有効なトークンなら True
    mask: np.ndarray
    #: 手番 (0=先手)
    turn: int
    #: 手数 (SFENの4要素目)
    move_number: int
    #: トークンに対応する piece_id。tracker を渡さなかった場合は None
    piece_ids: tuple[str, ...] | None = None

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    def index_of(self, piece_id: str) -> int:
        """piece_id からトークン位置を引く。"""
        if self.piece_ids is None:
            raise ValueError("piece_ids がありません (tracker を渡してトークン化してください)。")
        return self.piece_ids.index(piece_id)

    def squares(self) -> list[int]:
        """各トークンの cshogi マス番号。持ち駒と空きトークンは -1。

        core/effects.py の `piece_effect_matrix` にそのまま渡せる形。
        """
        return [
            int(position) if flag and position < NUM_SQUARES else -1
            for position, flag in zip(self.position, self.mask, strict=True)
        ]


class PieceTokenizer:
    """局面 → 駒トークン列。

    `max_pieces` を変えれば 5五将棋 (10枚) など他の変種にも使える。
    """

    def __init__(self, max_pieces: int = MAX_PIECES) -> None:
        self.max_pieces = max_pieces

    def tokenize(
        self, board: cshogi.Board, tracker: PieceIdTracker | None = None
    ) -> TokenizedPosition:
        """盤面をトークン化する。

        `tracker` を渡すと並びが piece_id 順に固定され、`piece_ids` が付く。
        渡さない場合は盤上のマス番号順 → 持ち駒の順に並べる。
        """
        if tracker is not None:
            entries = self._entries_from_tracker(tracker)
        else:
            entries = self._entries_from_board(board)

        if len(entries) > self.max_pieces:
            raise ValueError(f"駒が多すぎます: {len(entries)} > {self.max_pieces}")

        species = np.zeros(self.max_pieces, dtype=np.int8)
        position = np.zeros(self.max_pieces, dtype=np.int8)
        owner = np.zeros(self.max_pieces, dtype=np.int8)
        promoted = np.zeros(self.max_pieces, dtype=np.int8)
        mask = np.zeros(self.max_pieces, dtype=bool)

        piece_ids: list[str] = []
        for index, (piece_id, piece_species, piece_owner, piece_position) in enumerate(entries):
            species[index] = SPECIES_TO_INDEX[piece_species]
            position[index] = piece_position
            owner[index] = piece_owner
            promoted[index] = int(piece_species != base_species(piece_species))
            mask[index] = True
            if piece_id is not None:
                piece_ids.append(piece_id)

        return TokenizedPosition(
            species=species,
            position=position,
            owner=owner,
            promoted=promoted,
            mask=mask,
            turn=int(board.turn),
            move_number=int(board.move_number),
            piece_ids=tuple(piece_ids) if tracker is not None else None,
        )

    # --- トークンの並びを決める ------------------------------------------

    @staticmethod
    def _entries_from_tracker(
        tracker: PieceIdTracker,
    ) -> list[tuple[str | None, Species, int, int]]:
        entries = []
        for state in sorted(tracker.states.values(), key=lambda item: item.piece_id):
            position = (
                HAND_POSITION[state.owner]
                if state.square == HAND_SQUARE
                else str_to_sq(state.square)
            )
            entries.append((state.piece_id, state.species, state.owner, position))
        return entries

    @staticmethod
    def _entries_from_board(
        board: cshogi.Board,
    ) -> list[tuple[str | None, Species, int, int]]:
        entries: list[tuple[str | None, Species, int, int]] = []

        for square in range(NUM_SQUARES):
            code = board.piece(square)
            if code:
                entries.append(
                    (None, piece_code_to_species(code), piece_code_to_owner(code), square)
                )

        from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES

        for hand_owner, hand in enumerate(board.pieces_in_hand):
            for hand_index, count in enumerate(hand):
                species = HAND_INDEX_TO_SPECIES[hand_index]
                entries.extend(
                    (None, species, hand_owner, HAND_POSITION[hand_owner]) for _ in range(count)
                )

        return entries


def to_sfen(tokens: TokenizedPosition) -> str:
    """トークン列からSFENを復元する (往復の検証用)。

    cshogi の `Board.sfen()` と文字単位で一致する形式で出す。
    """
    grid: dict[int, tuple[Species, int]] = {}
    hands: list[dict[Species, int]] = [{}, {}]

    for index in range(len(tokens.mask)):
        if not tokens.mask[index]:
            continue
        species = SPECIES_ORDER[int(tokens.species[index])]
        owner = int(tokens.owner[index])
        position = int(tokens.position[index])

        if position < NUM_SQUARES:
            grid[position] = (species, owner)
        else:
            held = base_species(species)
            hands[owner][held] = hands[owner].get(held, 0) + 1

    ranks: list[str] = []
    for rank_index in range(9):
        text = ""
        empty = 0
        for file_index in reversed(range(9)):  # SFENは9筋 (左) から
            entry = grid.get(file_index * 9 + rank_index)
            if entry is None:
                empty += 1
                continue
            if empty:
                text += str(empty)
                empty = 0
            text += _sfen_piece(*entry)
        if empty:
            text += str(empty)
        ranks.append(text)

    return " ".join(
        [
            "/".join(ranks),
            "b" if tokens.turn == BLACK else "w",
            _sfen_hands(hands),
            str(tokens.move_number),
        ]
    )


def _sfen_piece(species: Species, owner: int) -> str:
    base = base_species(species)
    letter = SPECIES_TO_SFEN_LETTER[base]
    if owner != BLACK:
        letter = letter.lower()
    return ("+" + letter) if species != base else letter


def _sfen_hands(hands: list[dict[Species, int]]) -> str:
    text = ""
    for owner in (0, 1):
        for species in _HAND_ORDER:
            count = hands[owner].get(species, 0)
            if not count:
                continue
            letter = SPECIES_TO_SFEN_LETTER[species]
            if owner != BLACK:
                letter = letter.lower()
            text += (str(count) if count > 1 else "") + letter
    return text or "-"
