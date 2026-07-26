"""駒1体が生涯を通じて持つ内部状態 (DESIGN.md §2) と、恒久IDの追跡器。

将棋では取られた駒が相手の持ち駒として盤に戻る (DESIGN.md が言う「エージェントの転生」)。
そのため「この駒」を一意に指す識別子は、盤面のマスでも駒種でもなく **恒久ID (piece_id)**
でなければならない。`PieceIdTracker` が対局開始時にIDを配り、移動・成り・捕獲・打ちを
またいで追跡し続ける。

piece_id の形式は INTERFACE.md §3 の ``<初期位置><_gen世代><_連番>``::

    P77_gen0_0003
    ^ SFEN1文字 (生駒)
     ^^ 初期位置の 筋段
        ^^^^ 世代 (血統 [B/F] で使う)
             ^^^^ 対局内の連番 (盤上のマス番号順に1から採番)

このモジュールは Phase 0 の駒トークン化 (core/tokenizer.py) と、週1のサンプルJSONL生成
(scripts/gen_sample_jsonl.py) の両方から使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kokoro_shogi.core.pieces import (
    HAND_INDEX_TO_SPECIES,
    SPECIES_TO_SFEN_LETTER,
    Species,
    base_species,
    piece_code_to_owner,
    piece_code_to_species,
    promoted_species,
)
from kokoro_shogi.core.squares import DROP_FROM, HAND_SQUARE, NUM_SQUARES, sq_to_str

if TYPE_CHECKING:  # cshogi は実行時にのみ必要 (型注釈のためだけに import しない)
    import cshogi


@dataclass
class CareerStats:
    """対局を跨いで蓄積する駒の戦績 (DESIGN.md §2 / INTERFACE.md §5)。

    永続化は Phase 5 (persist/store.py) の担当。ここでは器だけ定義する。
    """

    games: int = 0
    survivals: int = 0
    promotions: int = 0
    checks: int = 0
    mvp_count: int = 0

    @property
    def survival_rate(self) -> float:
        return self.survivals / self.games if self.games else 0.0


@dataclass
class PieceState:
    """駒1体の状態 (DESIGN.md §2 の PieceState)。

    Phase 0-1 で使うのは piece_id / species / owner / square まで。
    theta_individual 以降は後続Phaseで機能フラグと共に有効化する。
    """

    piece_id: str
    species: Species
    owner: int
    #: 出生時の所有者。owner と異なれば「敵に転生した駒」[E]
    origin_owner: int
    #: 盤上なら "76" 形式、持ち駒なら "hand"
    square: str

    # --- 以下は後続Phaseで使う (今は既定値のまま) ---
    #: 個体性格 θ_ind [B] — Phase 5
    theta_individual: Any | None = None
    #: 感情GRUの隠れ状態 m_i [A] — Phase 3
    mood: Any | None = None
    #: 忠誠 [E] — Phase 5
    loyalty: float = 1.0
    #: 他駒との関係性 r_ij [C] — Phase 3
    relations: dict[str, float] = field(default_factory=dict)
    #: キャリア実績 [B] — Phase 5
    career: CareerStats = field(default_factory=CareerStats)
    #: 血統 (親のpiece_id) [B/F] — Phase 6
    lineage: list[str] = field(default_factory=list)

    @property
    def base_species(self) -> Species:
        """成りを外した駒種。"""
        return base_species(self.species)

    @property
    def in_hand(self) -> bool:
        return self.square == HAND_SQUARE

    @property
    def is_promoted(self) -> bool:
        return self.species != self.base_species

    @property
    def is_defector(self) -> bool:
        """出生時と所有者が変わっている (取られて相手の駒になった) か [E]。"""
        return self.owner != self.origin_owner


@dataclass(frozen=True)
class MoveRecord:
    """1手を piece_id の言葉で表したもの。

    INTERFACE.md の last_move へそのまま変換できるが、ワイヤ形式には依存しない
    (変換は logging 層の責務)。
    """

    piece_id: str
    #: 移動元。打ちは "00"
    from_square: str
    to_square: str
    capture: bool
    promote: bool
    drop: bool
    #: 取った駒の piece_id (取っていなければ None)
    captured_piece_id: str | None = None


class PieceIdTracker:
    """対局を通じて piece_id ↔ 盤面 の対応を維持する。

    使い方::

        board = cshogi.Board()
        tracker = PieceIdTracker(board)
        for move in moves:
            record = tracker.apply_move(board, move)   # push する前に呼ぶ
            board.push(move)
    """

    def __init__(self, board: cshogi.Board, generation: int = 0) -> None:
        self.generation = generation
        self._states: dict[str, PieceState] = {}
        self._by_square: dict[int, str] = {}

        for owner_hand in board.pieces_in_hand:
            if any(owner_hand):
                raise ValueError(
                    "PieceIdTracker は持ち駒のない局面 (通常は平手初期局面) から開始してください。"
                )

        serial = 0
        for square in range(NUM_SQUARES):
            code = board.piece(square)
            if code == 0:
                continue

            serial += 1
            species = piece_code_to_species(code)
            owner = piece_code_to_owner(code)
            square_text = sq_to_str(square)
            letter = SPECIES_TO_SFEN_LETTER[base_species(species)]
            piece_id = f"{letter}{square_text}_gen{generation}_{serial:04d}"

            self._states[piece_id] = PieceState(
                piece_id=piece_id,
                species=species,
                owner=owner,
                origin_owner=owner,
                square=square_text,
            )
            self._by_square[square] = piece_id

    # --- 参照 ---------------------------------------------------------------

    @property
    def states(self) -> dict[str, PieceState]:
        """piece_id → PieceState の全体。"""
        return self._states

    def get(self, piece_id: str) -> PieceState:
        return self._states[piece_id]

    def piece_id_at(self, square: int) -> str | None:
        """cshogi のマス番号にいる駒の piece_id。空マスなら None。"""
        return self._by_square.get(square)

    def pieces_in_hand(self, owner: int) -> list[PieceState]:
        """指定の対局者の持ち駒を piece_id 順で返す。"""
        return sorted(
            (state for state in self._states.values() if state.in_hand and state.owner == owner),
            key=lambda state: state.piece_id,
        )

    # --- 更新 ---------------------------------------------------------------

    def apply_move(self, board: cshogi.Board, move: int) -> MoveRecord:
        """1手を適用して piece_id の対応を更新する。

        `board` は **その手を指す前** の局面。呼び出し後に `board.push(move)` する。
        """
        import cshogi

        to_square = cshogi.move_to(move)
        to_text = sq_to_str(to_square)
        mover = board.turn

        if cshogi.move_is_drop(move):
            species = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
            piece_id = self._take_from_hand(mover, species)
            state = self._states[piece_id]
            state.square = to_text
            self._by_square[to_square] = piece_id
            return MoveRecord(
                piece_id=piece_id,
                from_square=DROP_FROM,
                to_square=to_text,
                capture=False,
                promote=False,
                drop=True,
            )

        from_square = cshogi.move_from(move)
        piece_id = self._by_square.pop(from_square, None)
        if piece_id is None:
            raise ValueError(f"移動元に駒がありません: {sq_to_str(from_square)}")

        state = self._states[piece_id]
        if state.owner != mover:
            raise ValueError(
                f"手番 {mover} が相手の駒 {piece_id} "
                f"(所有者 {state.owner}) を動かそうとしています。"
            )

        captured_piece_id = self._capture(to_square, mover)

        if cshogi.move_is_promotion(move):
            state.species = promoted_species(state.species)

        state.square = to_text
        self._by_square[to_square] = piece_id

        return MoveRecord(
            piece_id=piece_id,
            from_square=sq_to_str(from_square),
            to_square=to_text,
            capture=captured_piece_id is not None,
            promote=cshogi.move_is_promotion(move),
            drop=False,
            captured_piece_id=captured_piece_id,
        )

    def _capture(self, square: int, mover: int) -> str | None:
        """移動先の駒を取り、取った側の持ち駒にする (転生)。"""
        captured_piece_id = self._by_square.pop(square, None)
        if captured_piece_id is None:
            return None

        captured = self._states[captured_piece_id]
        captured.owner = mover
        captured.species = captured.base_species  # 成りは取られると外れる
        captured.square = HAND_SQUARE
        return captured_piece_id

    def _take_from_hand(self, owner: int, species: Species) -> str:
        """持ち駒から該当駒種を1枚選ぶ。同種が複数あれば piece_id 順で最初の1枚。"""
        for state in self.pieces_in_hand(owner):
            if state.base_species == species:
                return state.piece_id
        raise ValueError(f"対局者 {owner} は持ち駒に {species} を持っていません。")
