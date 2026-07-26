"""方策出力 (model/policy.py) の性質を検証する。

DESIGN.md §3(7) の方策

$$\\pi(a\\mid s_t) = \\frac{\\exp(s_{i,a}/\\tau)\\cdot\\mathbb{1}[a\\in\\mathcal{A}_i]}
                        {\\sum_j\\sum_{a'\\in\\mathcal{A}_j}\\exp(s_{j,a'}/\\tau)}$$

が満たすべき性質を、重みが乱数の初期状態でも成り立つ不変条件として確かめる。
学習の良し悪しではなく「分布として壊れていないか」を見るテスト。
"""

from __future__ import annotations

import random

import cshogi
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import FeatureFlags, ModelConfig  # noqa: E402
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer  # noqa: E402
from kokoro_shogi.data.dataset import (  # noqa: E402
    NUM_ACTIONS,
    action_index,
    decode_action,
    legal_move_mask,
)
from kokoro_shogi.model.policy import KokoroPolicy  # noqa: E402

#: テストを速くするための小型構成 (性質はモデルの大きさに依らない)
SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4)


def build_batch(boards: list[cshogi.Board]) -> dict[str, torch.Tensor]:
    """盤面のリストから forward_batch が食える辞書を作る。"""
    tokenizer = PieceTokenizer()
    columns: dict[str, list] = {key: [] for key in ("species", "position", "owner", "promoted")}
    masks, turns, legals = [], [], []

    for board in boards:
        tokens = tokenizer.tokenize(board)
        for key in columns:
            columns[key].append(getattr(tokens, key).astype("int64"))
        masks.append(tokens.mask)
        turns.append(tokens.turn)
        legals.append(
            legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
        )

    import numpy as np

    batch = {key: torch.from_numpy(np.stack(value)) for key, value in columns.items()}
    batch["mask"] = torch.from_numpy(np.stack(masks))
    batch["turn"] = torch.tensor(turns, dtype=torch.long)
    batch["legal"] = torch.from_numpy(np.stack(legals))
    return batch


def random_boards(count: int, seed: int = 42) -> list[cshogi.Board]:
    """ランダムな手順で進めた局面を集める。"""
    rng = random.Random(seed)
    boards = []
    board = cshogi.Board()
    for _ in range(count):
        for _ in range(rng.randint(1, 12)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                board = cshogi.Board()
                moves = list(board.legal_moves)
            board.push(rng.choice(moves))
        boards.append(cshogi.Board(board.sfen()))
    return boards


@pytest.fixture(scope="module")
def model() -> KokoroPolicy:
    torch.manual_seed(0)
    return KokoroPolicy(SMALL, FeatureFlags()).eval()


def test_probabilities_sum_to_one(model: KokoroPolicy) -> None:
    """Σπ = 1 (全駒・全手をまたいだ1つの分布になっている)。"""
    batch = build_batch(random_boards(6))
    with torch.no_grad():
        probs = model.forward_batch(batch).probs()

    assert probs.shape == (6, NUM_ACTIONS)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(6))


def test_illegal_moves_have_zero_probability(model: KokoroPolicy) -> None:
    """非合法手の確率が0である。"""
    batch = build_batch(random_boards(6, seed=7))
    with torch.no_grad():
        probs = model.forward_batch(batch).probs()

    illegal = ~batch["legal"].flatten(start_dim=1)
    assert float(probs[illegal].sum()) == 0.0
    # 合法手側には確率が残っている (全部0で「非合法0」を満たす退化を防ぐ)
    assert float(probs[~illegal].sum()) == pytest.approx(6.0, abs=1e-4)


def test_single_legal_move_gets_all_probability(model: KokoroPolicy) -> None:
    """合法手が1つしかなければ、その手の確率が ≈ 1 になる。"""
    batch = build_batch(random_boards(4, seed=3))
    legal = batch["legal"].flatten(start_dim=1)

    # 各局面の合法手を1つだけ残す
    only = torch.zeros_like(legal)
    for row in range(legal.shape[0]):
        only[row, int(legal[row].nonzero()[0])] = True
    batch["legal"] = only.view_as(batch["legal"])

    with torch.no_grad():
        probs = model.forward_batch(batch).probs()

    torch.testing.assert_close(probs.max(dim=-1).values, torch.ones(4))


def test_temperature_sharpens_the_distribution(model: KokoroPolicy) -> None:
    """温度を下げると分布が先鋭化する (DESIGN.md §3(7): 対局時 τ≈0.1)。"""
    batch = build_batch(random_boards(4, seed=11))
    with torch.no_grad():
        output = model.forward_batch(batch)

    sharp = output.probs(tau=0.1).max(dim=-1).values
    flat = output.probs(tau=1.0).max(dim=-1).values
    assert bool((sharp >= flat).all())


def test_value_is_in_range(model: KokoroPolicy) -> None:
    """状態価値は [-1, +1] (勝敗 z と同じ値域)。"""
    batch = build_batch(random_boards(6, seed=5))
    with torch.no_grad():
        value = model.forward_batch(batch).value

    assert value.shape == (6,)
    assert bool((value.abs() <= 1.0).all())


def test_best_action_decodes_to_a_legal_move() -> None:
    """argmax π が必ず合法手に落ちる (行動indexの分解と合法手マスクの整合)。"""
    torch.manual_seed(1)
    policy = KokoroPolicy(SMALL, FeatureFlags()).eval()

    boards = random_boards(8, seed=13)
    batch = build_batch(boards)
    with torch.no_grad():
        actions = policy.forward_batch(batch).best_action()

    tokenizer = PieceTokenizer()
    for index, board in enumerate(boards):
        token, to_square, promote = decode_action(int(actions[index]))
        assert 0 <= token < MAX_PIECES
        tokens = tokenizer.tokenize(board)
        assert bool(batch["legal"][index, token, to_square, promote])
        # 分解 → 再構成で元のindexへ戻る
        assert action_index(token, to_square, promote) == int(actions[index])
        assert tokens.mask[token]
