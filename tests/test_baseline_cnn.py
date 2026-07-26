"""Gate1 の比較対象 (model/baseline_cnn.py) を検証する。

ベースラインが間違っていると Gate1 の判定そのものが無意味になるので、
入力プレーンが盤面を正しく写しているかを実局面で確かめる。
"""

from __future__ import annotations

import random

import cshogi
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import ModelConfig  # noqa: E402
from kokoro_shogi.core.piece_state import PieceIdTracker  # noqa: E402
from kokoro_shogi.core.pieces import SPECIES_TO_HAND_INDEX  # noqa: E402
from kokoro_shogi.core.squares import NUM_SQUARES  # noqa: E402
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer  # noqa: E402
from kokoro_shogi.data.dataset import NUM_ACTIONS, legal_move_mask  # noqa: E402
from kokoro_shogi.model.baseline_cnn import (  # noqa: E402
    HAND_SATURATION,
    NUM_PLANES,
    BaselineCNN,
    build_planes,
)

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4)


def tokens_of(board: cshogi.Board, tracker: PieceIdTracker) -> dict[str, torch.Tensor]:
    tokens = PieceTokenizer().tokenize(board, tracker)
    return {
        "species": torch.from_numpy(tokens.species.astype("int64")).unsqueeze(0),
        "position": torch.from_numpy(tokens.position.astype("int64")).unsqueeze(0),
        "owner": torch.from_numpy(tokens.owner.astype("int64")).unsqueeze(0),
        "promoted": torch.from_numpy(tokens.promoted.astype("int64")).unsqueeze(0),
        "mask": torch.from_numpy(tokens.mask).unsqueeze(0),
        "turn": torch.tensor([tokens.turn], dtype=torch.long),
        "legal": torch.from_numpy(
            legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
        ).unsqueeze(0),
    }


def plane_args(tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """build_planes が取る引数だけ抜き出す。

    `promoted` は渡さない — 成りは species (SPECIES_ORDER に成駒6種が入っている) が
    既に区別しているので、プレーンでは別扱いにしない。
    """
    return {key: tokens[key] for key in ("species", "position", "owner", "mask", "turn")}


def advanced_position(seed: int, plies: int = 60) -> tuple[cshogi.Board, PieceIdTracker]:
    """持ち駒と成駒が出てくるところまで進めた局面。"""
    rng = random.Random(seed)
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        move = rng.choice(moves)
        tracker.apply_move(board, move)
        board.push(move)
    return board, tracker


def test_initial_position_planes_match_the_board() -> None:
    """平手初期局面: 盤上プレーンの合計が40、持ち駒プレーンは全て0。"""
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    planes = build_planes(**plane_args(tokens_of(board, tracker)))

    assert planes.shape == (1, NUM_PLANES, 9, 9)
    board_planes = planes[0, : 14 * 2]
    assert float(board_planes.sum()) == pytest.approx(40.0)
    assert float(planes[0, 14 * 2 : 14 * 2 + 7 * 2].sum()) == 0.0
    assert float(planes[0, -1].sum()) == 0.0  # 先手番


def test_planes_place_every_piece_on_its_square() -> None:
    """各駒が (駒種, 所有) の面の正しいマスに1つだけ立っている。"""
    board, tracker = advanced_position(seed=5)
    tokens = tokens_of(board, tracker)
    planes = build_planes(**plane_args(tokens))[0]

    flat = planes.view(NUM_PLANES, NUM_SQUARES)
    on_board = 0
    for index in range(MAX_PIECES):
        square = int(tokens["position"][0, index])
        if square >= NUM_SQUARES:
            continue
        plane = int(tokens["species"][0, index]) * 2 + int(tokens["owner"][0, index])
        assert float(flat[plane, square]) == 1.0
        on_board += 1

    assert float(flat[: 14 * 2].sum()) == pytest.approx(float(on_board))


def test_hand_planes_count_the_pieces_in_hand() -> None:
    """持ち駒プレーンの値が (枚数 / 18) の定数面になっている。"""
    board, tracker = advanced_position(seed=11, plies=90)
    tokens = tokens_of(board, tracker)
    planes = build_planes(**plane_args(tokens))[0]

    expected: dict[tuple[int, int], int] = {}
    for state in tracker.states.values():
        if state.in_hand:
            key = (SPECIES_TO_HAND_INDEX[state.base_species], state.owner)
            expected[key] = expected.get(key, 0) + 1

    offset = 14 * 2
    for (slot, owner), count in expected.items():
        plane = planes[offset + slot * 2 + owner]
        assert float(plane.min()) == float(plane.max())  # 定数面
        assert float(plane[0, 0]) == pytest.approx(count / HAND_SATURATION)

    total_in_hand = sum(expected.values())
    assert float(planes[offset : offset + 7 * 2].sum()) == pytest.approx(
        total_in_hand / HAND_SATURATION * NUM_SQUARES
    )
    assert total_in_hand > 0, "持ち駒が出る局面で検証したい"


def test_turn_plane_follows_the_side_to_move() -> None:
    board, tracker = advanced_position(seed=3, plies=1)
    tokens = tokens_of(board, tracker)
    planes = build_planes(**plane_args(tokens))[0]

    assert float(planes[-1].mean()) == float(board.turn)


def test_forward_produces_a_valid_distribution() -> None:
    """KokoroPolicy と同じ出力形・同じマスク規約で返る (比較可能であること)。"""
    torch.manual_seed(0)
    model = BaselineCNN(SMALL, channels=16, blocks=2).eval()

    board, tracker = advanced_position(seed=7)
    batch = tokens_of(board, tracker)
    with torch.no_grad():
        output = model.forward_batch(batch)

    assert output.logits.shape == (1, NUM_ACTIONS)
    probs = output.probs()
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1))
    assert float(probs[~batch["legal"].flatten(start_dim=1)].sum()) == 0.0
    assert bool((output.value.abs() <= 1.0).all())


def test_hand_tokens_share_one_learned_vector() -> None:
    """持ち駒トークンは盤上のマスを持たないので、共通の駒台ベクトルになる。"""
    torch.manual_seed(0)
    model = BaselineCNN(SMALL, channels=16, blocks=2).eval()
    with torch.no_grad():
        model.hand_token.normal_()

    board, tracker = advanced_position(seed=11, plies=90)
    batch = tokens_of(board, tracker)
    with torch.no_grad():
        hidden = model.forward_batch(batch).hidden[0]

    in_hand = (batch["position"][0] >= NUM_SQUARES) & batch["mask"][0]
    assert bool(in_hand.any()), "持ち駒が出る局面で検証したい"

    rows = hidden[in_hand]
    assert np.allclose(rows.numpy(), rows[0].numpy())
