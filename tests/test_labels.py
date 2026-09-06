"""利き抽出 (core/effects.py) と欲求ラベル (data/labels.py) の性質を検証する。

cshogi は利き (bitboard) をPythonに出さないので effects.py が自前で生成している。
自前実装が正しいことは、**cshogi 自身の手生成と突き合わせて**確かめる:
王手がかかっていない局面では「ある駒の擬似合法手の行き先」は
「その駒の利き から 自駒のいるマスを除いたもの」と一致するはず。
"""

from __future__ import annotations

import random

import cshogi
import numpy as np
import pytest

from kokoro_shogi.core.effects import (
    ATTACKS_ENEMY,
    NO_EFFECT,
    SUPPORTS_ALLY,
    attack_counts,
    attack_lists,
    attacks_from,
    board_occupancy,
    piece_effect_matrix,
)
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.pieces import piece_code_to_owner
from kokoro_shogi.core.squares import str_to_sq
from kokoro_shogi.core.tokenizer import PieceTokenizer
from kokoro_shogi.data.labels import (
    ADVANCE,
    ATTACK,
    DEFEND,
    DESIRE_AXES,
    NUM_AXES,
    PROMOTE,
    REDEPLOY,
    SURVIVE,
    compute_desire_labels,
)

EMPTY: frozenset[int] = frozenset()


# --- 利きの生成 -------------------------------------------------------------


def test_pawn_moves_one_square_forward() -> None:
    """先手の歩は段が減る向き (7七 → 7六)。後手はその逆。"""
    assert attacks_from("FU", 0, str_to_sq("77"), EMPTY) == [str_to_sq("76")]
    assert attacks_from("FU", 1, str_to_sq("33"), EMPTY) == [str_to_sq("34")]


def test_pawn_on_last_rank_has_no_attack() -> None:
    assert attacks_from("FU", 0, str_to_sq("71"), EMPTY) == []


def test_knight_jumps_two_ranks() -> None:
    assert set(attacks_from("KE", 0, str_to_sq("77"), EMPTY)) == {
        str_to_sq("85"),
        str_to_sq("65"),
    }


def test_lance_slides_until_blocked() -> None:
    open_board = attacks_from("KY", 0, str_to_sq("19"), EMPTY)
    assert len(open_board) == 8
    blocked = attacks_from("KY", 0, str_to_sq("19"), frozenset({str_to_sq("16")}))
    assert blocked == [str_to_sq("18"), str_to_sq("17"), str_to_sq("16")]
    assert str_to_sq("15") not in blocked, "駒に当たったらその先へは利かない"


def test_gold_and_promoted_pieces_share_movement() -> None:
    square = str_to_sq("55")
    gold = set(attacks_from("KI", 0, square, EMPTY))
    for species in ("TO", "NY", "NK", "NG"):
        assert set(attacks_from(species, 0, square, EMPTY)) == gold


def test_horse_is_bishop_plus_king_steps() -> None:
    square = str_to_sq("55")
    horse = set(attacks_from("UM", 0, square, EMPTY))
    bishop = set(attacks_from("KA", 0, square, EMPTY))
    assert bishop < horse
    assert horse - bishop == {
        str_to_sq("54"),
        str_to_sq("56"),
        str_to_sq("45"),
        str_to_sq("65"),
    }


def test_dragon_is_rook_plus_diagonal_steps() -> None:
    square = str_to_sq("55")
    dragon = set(attacks_from("RY", 0, square, EMPTY))
    rook = set(attacks_from("HI", 0, square, EMPTY))
    assert rook < dragon
    assert dragon - rook == {
        str_to_sq("44"),
        str_to_sq("64"),
        str_to_sq("46"),
        str_to_sq("66"),
    }


@pytest.mark.parametrize("seed", range(4))
def test_attacks_agree_with_cshogi_move_generation(seed: int) -> None:
    """自前の利きが cshogi の手生成と一致する (これが effects.py の正しさの根拠)。"""
    rng = random.Random(seed)
    board = cshogi.Board()

    for _ in range(60):
        if board.is_game_over():
            break

        if not board.is_check():
            # 王手中の擬似合法手は「王手回避手」に絞られてしまうので比較対象にしない
            destinations: dict[int, set[int]] = {}
            for move in board.pseudo_legal_moves:
                if cshogi.move_is_drop(move):
                    continue
                destinations.setdefault(cshogi.move_from(move), set()).add(cshogi.move_to(move))

            lists = attack_lists(board)
            own = {
                square
                for square in board_occupancy(board)
                if piece_code_to_owner(board.piece(square)) == board.turn
            }
            for square in own:
                assert set(lists[square]) - own == destinations.get(square, set())

        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))


def test_attack_counts_on_initial_position() -> None:
    counts = attack_counts(cshogi.Board())
    assert counts.shape == (2, 81)
    # 先手の歩は前のマスへ利いている
    assert counts[0][str_to_sq("76")] >= 1
    # 自陣最奥に相手の利きは無い
    assert counts[1][str_to_sq("19")] == 0


# --- 利き関係行列 -----------------------------------------------------------


def test_effect_matrix_distinguishes_support_and_attack() -> None:
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    for usi in ("7g7f", "3c3d"):
        move = board.move_from_usi(usi)
        tracker.apply_move(board, move)
        board.push(move)

    tokens = PieceTokenizer().tokenize(board, tracker)
    matrix = piece_effect_matrix(board, tokens.squares())
    assert matrix.shape == (len(tokens.mask), len(tokens.mask))

    # 8八の角は9七の歩・7九の銀・9九の香に紐を付けている
    bishop = tokens.index_of("B88_gen0_0035")
    assert SUPPORTS_ALLY in set(matrix[bishop])

    # 7七が空いたので、同じ斜め筋の2二角まで利きが通っている
    assert ATTACKS_ENEMY in set(matrix[bishop])

    # 紐と攻撃は別の値として区別される
    assert SUPPORTS_ALLY != ATTACKS_ENEMY != NO_EFFECT


def test_effect_matrix_marks_capturable_enemy() -> None:
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    for usi in ("7g7f", "3c3d"):
        move = board.move_from_usi(usi)
        tracker.apply_move(board, move)
        board.push(move)

    tokens = PieceTokenizer().tokenize(board, tracker)
    matrix = piece_effect_matrix(board, tokens.squares())

    # 8八の角と2二の角は同じ斜め筋にいて、互いに取り合える
    black_bishop = tokens.index_of("B88_gen0_0035")
    white_bishop = tokens.index_of("B22_gen0_0006")
    assert matrix[black_bishop][white_bishop] == ATTACKS_ENEMY
    assert matrix[white_bishop][black_bishop] == ATTACKS_ENEMY


def test_effect_matrix_ignores_hand_pieces() -> None:
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    for usi in ("7g7f", "3c3d", "8h2b+"):
        move = board.move_from_usi(usi)
        tracker.apply_move(board, move)
        board.push(move)

    tokens = PieceTokenizer().tokenize(board, tracker)
    matrix = piece_effect_matrix(board, tokens.squares())
    captured = tokens.index_of("B22_gen0_0006")  # 持ち駒になった角
    assert set(matrix[captured]) == {NO_EFFECT}
    assert set(matrix[:, captured]) == {NO_EFFECT}


# --- 欲求ラベル -------------------------------------------------------------


def _labels_for(usi_moves: list[str]):
    board = cshogi.Board()
    moves = []
    for usi in usi_moves:
        move = board.move_from_usi(usi)
        # 非合法手を書くと move の中身が壊れて追跡側が意味不明な失敗をするので、
        # テスト側の手順ミスをここで止める
        assert board.is_legal(move), f"テストの手順が非合法です: {usi} ({board.sfen()})"
        moves.append(move)
        board.push(move)
    return compute_desire_labels(moves)


def test_label_shape_and_range() -> None:
    labels = _labels_for(["7g7f", "3c3d", "8h2b+", "3a2b", "B*5e"])
    assert labels.labels.shape == (5, 40, NUM_AXES)
    assert labels.mask.all(), "本将棋は常に40枚なので全トークンが有効"
    assert labels.labels.min() >= 0.0
    assert labels.labels.max() <= 1.0
    assert len(DESIRE_AXES) == NUM_AXES


def test_survive_is_zero_for_a_piece_about_to_be_captured() -> None:
    """2二の角は3手目に取られる → その時点から k手先読みで survive=0。"""
    labels = _labels_for(["7g7f", "3c3d", "8h2b+", "3a2b", "B*5e", "5c5d"])
    index = labels.piece_ids.index("B22_gen0_0006")
    assert labels.labels[0, index, SURVIVE] == 0.0, "1手目の時点で4手以内に取られる"


def test_survive_counts_a_reincarnated_piece_as_alive() -> None:
    """取られても k手以内に打ち直されれば「t+k で盤上」なので生存扱いになる。

    将棋特有の「持ち駒 = エージェントの転生」(DESIGN.md [E]) が、
    生存ラベルの定義にそのまま効いている例。取られた瞬間に survive=0 と
    決めつけないのが本将棋での正しい読み方。
    """
    labels = _labels_for(["7g7f", "3c3d", "8h2b+", "3a2b", "B*5e", "5c5d"])
    index = labels.piece_ids.index("B22_gen0_0006")
    assert labels.labels[3, index, SURVIVE] == 1.0, "5手目に先手の駒として打ち直される"


def test_survive_is_one_for_a_safe_piece() -> None:
    labels = _labels_for(["7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e"])
    index = labels.piece_ids.index("K59_gen0_0022")  # 先手玉は動かない
    assert labels.labels[:, index, SURVIVE].min() == 1.0


def test_attack_label_marks_the_capturing_piece() -> None:
    labels = _labels_for(["7g7f", "3c3d", "8h2b+", "3a2b", "B*5e"])
    bishop = labels.piece_ids.index("B88_gen0_0035")
    assert labels.labels[0, bishop, ATTACK] == 1.0, "3手目に角が取る = 4手先読みに入る"


def test_promote_label_marks_the_promoting_piece() -> None:
    labels = _labels_for(["7g7f", "3c3d", "8h2b+", "3a2b", "B*5e"])
    bishop = labels.piece_ids.index("B88_gen0_0035")
    assert labels.labels[0, bishop, PROMOTE] == 1.0


def test_redeploy_label_marks_a_dropped_hand_piece() -> None:
    """持ち駒になった角が打たれる局面で redeploy が立つ。"""
    labels = _labels_for(["7g7f", "3c3d", "8h2b+", "3a2b", "B*5e", "5c5d"])
    index = labels.piece_ids.index("B22_gen0_0006")
    # 4手目 (0始まり) の時点でこの駒は持ち駒。次の手で打たれる
    assert labels.labels[4, index, REDEPLOY] == 1.0
    assert labels.labels[0, index, REDEPLOY] == 0.0, "まだ盤上にいる駒には立たない"


def test_advance_label_marks_forward_movement() -> None:
    labels = _labels_for(["7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e"])
    pawn = labels.piece_ids.index("P27_gen0_0008")  # 2七の歩が2六→2五と進む
    assert labels.labels[0, pawn, ADVANCE] == 1.0


def test_defend_is_high_for_pieces_guarding_the_king() -> None:
    labels = _labels_for(["7g7f", "3c3d"])
    gold = labels.piece_ids.index("G49_gen0_0018")  # 4九の金は5九の玉に紐を付けている
    pawn = labels.piece_ids.index("P17_gen0_0003")  # 端歩は誰も守っていない
    assert labels.labels[0, gold, DEFEND] > labels.labels[0, pawn, DEFEND]


def test_defend_uses_current_position_not_lookahead() -> None:
    """defend だけは先読みせず、その局面の利きで決まる。"""
    labels = _labels_for(["7g7f", "3c3d", "2g2f", "8c8d"])
    assert not np.array_equal(labels.labels[0, :, DEFEND], labels.labels[3, :, DEFEND])
