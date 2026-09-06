"""駒トークン化 (core/tokenizer.py) と恒久ID追跡 (core/piece_state.py) の性質を検証する。

DESIGN.md §3(1) の駒トークン化は、盤面の情報を落とさずに駒単位へ並べ替える操作。
落としていないことを「SFENへ戻せる」で確かめ、駒を取り違えていないことを
「恒久IDが捕獲・成り・打ちをまたいで追える」で確かめる。
"""

from __future__ import annotations

import random

import cshogi
import pytest

from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.pieces import (
    SPECIES_TO_HAND_INDEX,
    piece_code_to_owner,
    piece_code_to_species,
)
from kokoro_shogi.core.squares import DROP_FROM, HAND_SQUARE, sq_to_str, str_to_sq
from kokoro_shogi.core.tokenizer import HAND_POSITION, MAX_PIECES, PieceTokenizer, to_sfen

PLAYOUT_GAMES = 8
PLAYOUT_PLIES = 60


def random_playout(seed: int, plies: int = PLAYOUT_PLIES):
    """ランダムな合法手で対局を進めながら (board, tracker, move) を返す。"""
    rng = random.Random(seed)
    board = cshogi.Board()
    tracker = PieceIdTracker(board)

    for _ in range(plies):
        if board.is_game_over():
            return
        moves = list(board.legal_moves)
        if not moves:
            return
        move = rng.choice(moves)
        yield board, tracker, move
        tracker.apply_move(board, move)
        board.push(move)


# --- マス変換 ---------------------------------------------------------------


def test_square_conversion_matches_interface_example() -> None:
    """INTERFACE.md §6 の例: 7g7f は from "77" to "76"。"""
    board = cshogi.Board()
    move = board.move_from_usi("7g7f")
    assert sq_to_str(cshogi.move_from(move)) == "77"
    assert sq_to_str(cshogi.move_to(move)) == "76"


@pytest.mark.parametrize("square", range(81))
def test_square_round_trip(square: int) -> None:
    assert str_to_sq(sq_to_str(square)) == square


@pytest.mark.parametrize("text", ["00", "0", "1", "aa", "99a", "", "hand"])
def test_invalid_square_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        str_to_sq(text)


# --- トークン化 -------------------------------------------------------------


def test_initial_position_has_40_tokens() -> None:
    board = cshogi.Board()
    tokens = PieceTokenizer().tokenize(board)
    assert tokens.count == 40
    assert len(tokens.mask) == MAX_PIECES


def test_piece_count_stays_40_because_captures_become_hand_pieces() -> None:
    """駒は盤から消えない。取られた駒は駒台のトークンになる。"""
    tokenizer = PieceTokenizer()
    for board, tracker, _ in random_playout(seed=11):
        assert tokenizer.tokenize(board).count == 40
        assert tokenizer.tokenize(board, tracker).count == 40


def test_hand_pieces_use_dedicated_position_ids() -> None:
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    for usi in ("7g7f", "3c3d", "8h2b+"):
        move = board.move_from_usi(usi)
        tracker.apply_move(board, move)
        board.push(move)

    tokens = PieceTokenizer().tokenize(board, tracker)
    captured = tokens.index_of("B22_gen0_0006")  # 後手の角が先手の持ち駒になった
    assert tokens.position[captured] == HAND_POSITION[0]
    assert tokens.owner[captured] == 0
    assert tokens.squares()[captured] == -1


@pytest.mark.parametrize("seed", range(PLAYOUT_GAMES))
def test_sfen_round_trip(seed: int) -> None:
    """SFEN → トークン → SFEN が元に戻る (情報を落としていない)。"""
    tokenizer = PieceTokenizer()
    for board, tracker, _ in random_playout(seed):
        assert to_sfen(tokenizer.tokenize(board)) == board.sfen()
        assert to_sfen(tokenizer.tokenize(board, tracker)) == board.sfen()


def test_token_order_is_stable_across_moves() -> None:
    """tracker を渡した並びは手が進んでも変わらない (感情GRUが駒を追えるため)。"""
    tokenizer = PieceTokenizer()
    orders = [
        tokenizer.tokenize(board, tracker).piece_ids
        for board, tracker, _ in random_playout(seed=5)
    ]
    assert len(orders) > 10
    assert all(order == orders[0] for order in orders)


# --- 恒久IDの追跡 -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(PLAYOUT_GAMES))
def test_tracker_matches_board(seed: int) -> None:
    """追跡している駒種・所有者・位置が、cshogiの盤面と常に一致する。"""
    for board, tracker, _ in random_playout(seed):
        for square in range(81):
            code = board.piece(square)
            piece_id = tracker.piece_id_at(square)
            if code == 0:
                assert piece_id is None
                continue
            assert piece_id is not None
            state = tracker.get(piece_id)
            assert state.species == piece_code_to_species(code)
            assert state.owner == piece_code_to_owner(code)
            assert state.square == sq_to_str(square)

        for owner in (0, 1):
            counted = [0] * 7
            for state in tracker.pieces_in_hand(owner):
                counted[SPECIES_TO_HAND_INDEX[state.base_species]] += 1
            assert list(board.pieces_in_hand[owner]) == counted


def test_piece_id_survives_capture_promotion_and_drop() -> None:
    """1枚の駒を、取られ → 相手の持ち駒 → 打たれ → 成り まで追いかける。"""
    board = cshogi.Board()
    tracker = PieceIdTracker(board)

    def play(usi: str):
        move = board.move_from_usi(usi)
        record = tracker.apply_move(board, move)
        board.push(move)
        return record

    play("7g7f")
    play("3c3d")

    # 先手の角が後手の角を取って成る
    record = play("8h2b+")
    assert record.capture and record.promote
    assert record.piece_id == "B88_gen0_0035"
    assert record.captured_piece_id == "B22_gen0_0006"

    captured = tracker.get("B22_gen0_0006")
    assert captured.in_hand
    assert captured.owner == 0, "取られた駒は相手の持ち駒になる (転生)"
    assert captured.origin_owner == 1, "出生時の所有者は変わらない"
    assert captured.is_defector
    assert captured.species == "KA", "成りは取られると外れる"

    # 成った角を取り返されると、成りが外れた状態で後手の持ち駒になる
    play("3a2b")
    promoted_back = tracker.get("B88_gen0_0035")
    assert promoted_back.in_hand and promoted_back.owner == 1
    assert promoted_back.species == "KA"

    # 先手が転生した角を打つ
    record = play("B*5e")
    assert record.drop and record.from_square == DROP_FROM
    assert record.piece_id == "B22_gen0_0006"
    assert tracker.get("B22_gen0_0006").square == "55"

    # 打った駒が成る
    play("2b3c")
    record = play("5e3c+")
    assert record.piece_id == "B22_gen0_0006" and record.promote
    assert tracker.get("B22_gen0_0006").species == "UM"
    assert tracker.get("B22_gen0_0006").is_promoted


def test_all_piece_ids_are_unique_and_well_formed() -> None:
    tracker = PieceIdTracker(cshogi.Board())
    assert len(tracker.states) == 40
    assert len(set(tracker.states)) == 40
    for piece_id, state in tracker.states.items():
        assert piece_id[0] in "PLNSGBRK"
        assert f"_gen{tracker.generation}_" in piece_id
        assert state.origin_owner == state.owner


def test_tracker_rejects_position_with_hand_pieces() -> None:
    """恒久IDは初期位置から振るので、持ち駒のある局面からは始められない。"""
    board = cshogi.Board()
    board.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b P 1")
    with pytest.raises(ValueError, match="持ち駒のない局面"):
        PieceIdTracker(board)


def test_tracker_rejects_moving_opponent_piece() -> None:
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    other = cshogi.Board()
    other.push_usi("7g7f")
    white_move = other.move_from_usi("3c3d")  # 後手の手を先手番の盤に渡す
    with pytest.raises(ValueError, match="相手の駒"):
        tracker.apply_move(board, white_move)


def test_hand_square_constants_match_interface() -> None:
    """持ち駒のマス表記と打ちの移動元は INTERFACE.md §3 の規定どおり。"""
    assert HAND_SQUARE == "hand"
    assert DROP_FROM == "00"
