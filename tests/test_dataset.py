"""学習用シャードの読み出し (data/dataset.py) の性質を検証する。

Phase 0 のシャードと Phase 1 のモデルをつなぐ層なので、ここがずれると
「学習は回るが教師が間違っている」という一番気づきにくい壊れ方をする。
特に危ないのが**打ちの駒の選び方**で、
`piece_state.PieceIdTracker._take_from_hand` (シャードを作った側) と
`dataset.legal_move_mask` (学習で使う側) が同じ駒を選ばないと、
教師手が合法手マスクの外に出て損失が -1e9 のセルを指すことになる。

棋譜がない環境 (CI・他メンバーの手元) でも動くよう、主な検証は
ランダム対局から作った局面で行い、実シャードを使うテストは無ければskipする。
"""

from __future__ import annotations

import random
from pathlib import Path

import cshogi
import numpy as np
import pytest

from kokoro_shogi.config import REPO_ROOT
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer
from kokoro_shogi.data.dataset import (
    NUM_ACTIONS,
    NUM_MOVE_TO,
    NUM_PROMOTE,
    ShardDataset,
    action_index,
    decode_action,
    find_shards,
    legal_move_mask,
    split_shards,
    tokens_to_board,
)

PLAYOUT_GAMES = 6
PLAYOUT_PLIES = 50


def playout(seed: int, plies: int = PLAYOUT_PLIES):
    """ランダムな合法手で進めながら (board, tracker, move) を返す。"""
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


def test_action_index_round_trip() -> None:
    """(駒, 移動先, 成り) ↔ action index が1対1で往復する。"""
    seen = set()
    for token in range(MAX_PIECES):
        for to_square in range(NUM_MOVE_TO):
            for promote in range(NUM_PROMOTE):
                index = action_index(token, to_square, promote)
                assert 0 <= index < NUM_ACTIONS
                assert decode_action(index) == (token, to_square, promote)
                seen.add(index)
    assert len(seen) == NUM_ACTIONS


@pytest.mark.parametrize("seed", range(PLAYOUT_GAMES))
def test_tokens_to_board_restores_the_position(seed: int) -> None:
    """トークンから盤面を復元すると元のSFENに戻る。"""
    tokenizer = PieceTokenizer()

    for board, tracker, _ in playout(seed):
        tokens = tokenizer.tokenize(board, tracker)
        restored = tokens_to_board(
            tokens.species, tokens.position, tokens.owner, tokens.promoted, tokens.mask, tokens.turn
        )
        # 手数 (SFENの4要素目) は合法手生成に影響しないので比較から外す
        assert restored.sfen().rsplit(" ", 1)[0] == board.sfen().rsplit(" ", 1)[0]


@pytest.mark.parametrize("seed", range(PLAYOUT_GAMES))
def test_legal_mask_covers_every_legal_move(seed: int) -> None:
    """合法手マスクの True の数が合法手の数と一致する。

    2つの合法手が同じ (駒, 移動先, 成り) に潰れていないこと = 行動空間が
    手を取りこぼしていないことの確認。
    """
    tokenizer = PieceTokenizer()

    for board, tracker, _ in playout(seed):
        tokens = tokenizer.tokenize(board, tracker)
        mask = legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
        assert int(mask.sum()) == len(list(board.legal_moves))


@pytest.mark.parametrize("seed", range(PLAYOUT_GAMES))
def test_teacher_move_is_inside_the_legal_mask(seed: int) -> None:
    """棋譜の指し手が必ず合法手マスクの中に入る。

    シャード生成 (scripts/make_labels.py) と同じ規則で move_token を決めているか、
    特に同種の持ち駒が複数あるときの打ちで一致しているかを見る。
    """
    tokenizer = PieceTokenizer()

    for board, tracker, move in playout(seed):
        tokens = tokenizer.tokenize(board, tracker)
        order = {piece_id: index for index, piece_id in enumerate(tokens.piece_ids or ())}

        if cshogi.move_is_drop(move):
            species = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
            candidates = [
                state
                for state in tracker.pieces_in_hand(board.turn)
                if state.base_species == species
            ]
            moving = order[candidates[0].piece_id]
        else:
            moving = order[tracker.piece_id_at(cshogi.move_from(move))]

        mask = legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
        assert mask[moving, cshogi.move_to(move), int(cshogi.move_is_promotion(move))]


def test_split_shards_does_not_overlap() -> None:
    """train と val が同じシャードを共有しない (局単位のリーク防止)。"""
    paths = [REPO_ROOT / f"shard_{index:04d}.npz" for index in range(10)]
    train, val = split_shards(paths, val_ratio=0.2)

    assert not set(train) & set(val)
    assert len(train) + len(val) == len(paths)
    assert len(val) == 2

    with pytest.raises(ValueError):
        split_shards(paths[:1])


# --- 実シャードを使う検証 (棋譜がなければskip) ---------------------------------


@pytest.fixture(scope="module")
def shard_dataset() -> ShardDataset:
    paths = find_shards(REPO_ROOT / "data" / "shards")
    if not paths:
        pytest.skip("data/shards がありません (scripts/make_labels.py で生成してください)")
    return ShardDataset(paths[:1], max_positions=200)


def test_games_do_not_span_shards() -> None:
    """1局が2つのシャードに分かれていない。

    Phase 3 の感情GRU [A] は1局を通した系列を必要とする。局がシャードを
    跨いでいると、系列を作るたびに隣のシャードを読む羽目になる。
    `scripts/make_labels.py` の `flush_if_full` が局の境界でだけ切る前提。
    """
    paths = find_shards(REPO_ROOT / "data" / "shards" / "2024")
    if len(paths) < 2:
        pytest.skip("data/shards/2024 に複数シャードがありません")

    previous_last: int | None = None
    for path in paths:
        with np.load(path) as shard:
            games = shard["game_index"]
        if previous_last is not None:
            assert games[0] != previous_last, f"{path.name} が前のシャードと同じ局から始まっている"
        previous_last = int(games[-1])


def test_shard_rows_are_consistent(shard_dataset: ShardDataset) -> None:
    """実シャードの各行で、教師手が合法手マスクに入っていて値域も正しい。"""
    for index in range(len(shard_dataset)):
        item = shard_dataset[index]

        assert item.legal.flatten()[item.action], f"{index}行目の教師手がマスク外"
        assert item.result in (-1, 0, 1)
        assert item.mask.sum() == MAX_PIECES  # 本将棋は常に40枚
        assert 0.0 <= float(item.labels.min()) and float(item.labels.max()) <= 1.0
        assert item.effect.shape == (MAX_PIECES, MAX_PIECES)
        assert set(np.unique(item.effect)) <= {0, 1, 2}


def test_find_shards_skips_teacher_side_files(tmp_path: Path) -> None:
    """*.teacher.npz は本体と同じ glob に当たるが、列が違うので除外される。"""
    from kokoro_shogi.data.dataset import find_shards

    (tmp_path / "shard_0000.npz").write_bytes(b"")
    (tmp_path / "shard_0000.teacher.npz").write_bytes(b"")
    (tmp_path / "shard_0001.npz").write_bytes(b"")
    found = find_shards(tmp_path)
    assert [p.name for p in found] == ["shard_0000.npz", "shard_0001.npz"]
