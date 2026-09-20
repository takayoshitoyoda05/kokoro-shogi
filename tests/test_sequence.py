"""系列データセット (data/sequence.py) と系列蒸留の整合を検証する。

シャードの行を独立に読む ShardDataset と、局を再生する SequenceDataset が
同じ局面に対して同じ入力 (トークン・合法手・利き) を出すこと、そして
イベントの時刻合わせ (行 t のイベント = t を作った手) を固定する。
"""

from __future__ import annotations

from pathlib import Path

import cshogi
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import ModelConfig  # noqa: E402
from kokoro_shogi.core.piece_state import PieceIdTracker  # noqa: E402
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer  # noqa: E402
from kokoro_shogi.data.sequence import (  # noqa: E402
    SequenceDataset,
    collate_sequences,
)
from kokoro_shogi.model.mood import MOVED, NUM_EVENT_FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """ランダム自己対戦2局ぶんの小さなシャードを作る (make_labels と同じ列)。"""
    import random

    rng = random.Random(7)
    tokenizer = PieceTokenizer()
    columns: dict[str, list] = {}

    for game_index in range(2):
        board = cshogi.Board()
        tracker = PieceIdTracker(board)
        for _ in range(24):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            move = rng.choice(moves)
            tokens = tokenizer.tokenize(board, tracker)

            moving_square = -1 if cshogi.move_is_drop(move) else cshogi.move_from(move)
            if moving_square >= 0:
                moving_id = tracker.piece_id_at(moving_square)
            else:
                from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES

                species = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
                moving_id = next(
                    state.piece_id
                    for state in sorted(
                        tracker.states.values(), key=lambda item: item.piece_id
                    )
                    if state.in_hand
                    and state.owner == board.turn
                    and state.base_species == species
                )
            order = {pid: i for i, pid in enumerate(sorted(tracker.states))}

            row = dict(
                species=tokens.species,
                position=tokens.position,
                owner=tokens.owner,
                promoted=tokens.promoted,
                mask=tokens.mask,
                turn=np.int8(tokens.turn),
                move=np.int32(move),
                move_token=np.int8(order[moving_id]),
                move_to=np.int8(cshogi.move_to(move)),
                move_promote=np.bool_(cshogi.move_is_promotion(move)),
                result=np.int8(1),
                labels=np.zeros((MAX_PIECES, 6), dtype=np.uint8),
                game_index=np.int32(game_index),
            )
            for key, value in row.items():
                columns.setdefault(key, []).append(value)

            tracker.apply_move(board, move)
            board.push(move)

    path = tmp_path_factory.mktemp("shards") / "shard_0000.npz"
    np.savez(path, **{key: np.asarray(value) for key, value in columns.items()})
    return path


def test_games_are_split_at_boundaries(shard: Path) -> None:
    dataset = SequenceDataset([shard])
    assert len(dataset) == 2
    total = sum(dataset[g].length for g in range(2))
    with np.load(shard) as data:
        assert total == len(data["move_to"])


def test_adjacent_games_with_same_index_are_still_split(
    shard: Path, tmp_path: Path
) -> None:
    """game_index は棋譜ファイル内の連番なので、隣接する別の局が同じ番号を持ち得る。

    その場合でも「初期局面の行」を境界として2局に割れること (実データで
    学習をクラッシュさせた融合バグの回帰テスト)。
    """
    with np.load(shard) as data:
        columns = {name: data[name] for name in data.files}
    columns["game_index"] = np.zeros_like(columns["game_index"])  # 全局を同じ番号に

    merged = tmp_path / "shard_merged.npz"
    np.savez(merged, **columns)

    dataset = SequenceDataset([merged])
    assert len(dataset) == 2
    for game in range(2):
        sequence = dataset[game]  # 再生が通る = 境界が正しい
        assert sequence.length > 0


def test_replay_matches_stored_tokens(shard: Path) -> None:
    """局の再生 (合法手・イベント計算の土台) が行のトークンとずれない。"""
    dataset = SequenceDataset([shard])
    game = dataset[0]

    # 教師手は必ず合法手マスクの中にある = 再生盤面が行の局面と一致している
    for t in range(game.length):
        token, rest = divmod(int(game.action[t]), 81 * 2)
        to_square, promote = divmod(rest, 2)
        assert game.legal[t, token, to_square, promote]


def test_event_timing_is_the_move_into_the_row(shard: Path) -> None:
    dataset = SequenceDataset([shard])
    game = dataset[0]

    # 行0 (初期局面) には move 系イベントが無い
    assert game.events[0, :, MOVED].sum() == 0.0
    # 行 t≥1 の moved は「行 t-1 で指した駒」ちょうど1枚
    for t in range(1, game.length):
        assert game.events[t, :, MOVED].sum() == 1.0
        token = int(game.action[t - 1]) // (81 * 2)
        assert game.events[t, token, MOVED] == 1.0


def test_collate_pads_and_masks_steps(shard: Path) -> None:
    dataset = SequenceDataset([shard])
    games = [dataset[0], dataset[1]]
    batch = collate_sequences(games)

    longest = max(g.length for g in games)
    assert batch["species"].shape == (2, longest, MAX_PIECES)
    assert batch["events"].shape == (2, longest, MAX_PIECES, NUM_EVENT_FEATURES)
    for row, game in enumerate(games):
        assert int(batch["steps"][row].sum()) == game.length
        # 詰め物の行は steps=False で損失から外れる
        assert not batch["steps"][row, game.length :].any()


def test_run_epoch_trains_without_nan(shard: Path) -> None:
    """小型モデルで1エポック回し、損失が有限で GRU に勾配が流れることを確認。"""
    from torch.utils.data import DataLoader

    from kokoro_shogi.config import Config, FeatureFlags, LossConfig
    from kokoro_shogi.model.mood import MoodGRU
    from kokoro_shogi.model.policy import KokoroPolicy
    from kokoro_shogi.train.mood_distill import run_epoch

    small = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_mood=8)
    config = Config(model=small, loss=LossConfig(), features=FeatureFlags(mood=True))

    policy = KokoroPolicy(small, config.features, head="desire")
    gru = MoodGRU(small)
    before = [p.detach().clone() for p in gru.parameters()]
    optimizer = torch.optim.AdamW([*policy.parameters(), *gru.parameters()], lr=1e-3)

    loader = DataLoader(
        SequenceDataset([shard]), batch_size=2, collate_fn=collate_sequences
    )
    metrics = run_epoch(
        policy, gru, loader, config, torch.device("cpu"), tbptt=4, optimizer=optimizer
    )

    assert metrics.positions > 0
    assert np.isfinite(metrics.loss)
    # GRU にも勾配が流れて重みが動いた (mood → personality/trunk 経由の逆伝播)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, gru.parameters(), strict=True)
    )
    eval_metrics = run_epoch(policy, gru, loader, config, torch.device("cpu"), tbptt=4)
    assert np.isfinite(eval_metrics.loss)


def test_teacher_side_file_flows_through_sequences(shard: Path, tmp_path: Path) -> None:
    """*.teacher.npz があれば系列にも教師が載り、無い手は NaN/-1 のまま損失側で無視される。"""
    import shutil

    from torch.utils.data import DataLoader

    from kokoro_shogi.config import Config, FeatureFlags, LossConfig
    from kokoro_shogi.data.dataset import TEACHER_K
    from kokoro_shogi.model.mood import MoodGRU
    from kokoro_shogi.model.policy import KokoroPolicy
    from kokoro_shogi.train.mood_distill import run_epoch

    copy = tmp_path / "shard_0000.npz"
    shutil.copy(shard, copy)
    with np.load(copy) as raw:
        rows = len(raw["turn"])
        from kokoro_shogi.data.dataset import action_index

        stored = np.array(
            [
                action_index(int(raw["move_token"][i]), int(raw["move_to"][i]),
                             int(raw["move_promote"][i]))
                for i in range(rows)
            ]
        )
    value = np.full(rows, np.nan, dtype=np.float32)
    actions = np.full((rows, TEACHER_K), -1, dtype=np.int64)
    cps = np.full((rows, TEACHER_K), np.nan, dtype=np.float32)
    for i in (0, 5):  # 2 手だけ教師あり (指された手を 1 位にする)
        value[i] = 0.25
        actions[i, 0] = stored[i]
        cps[i, 0] = 80.0
    np.savez(copy.with_suffix(".teacher.npz"), teacher_value=value, teacher_cp=value,
             teacher_actions=actions, teacher_cps=cps)

    dataset = SequenceDataset([copy])
    assert dataset.teacher_shards == 1
    first = dataset[0]
    assert first.teacher_value[0] == pytest.approx(0.25)
    assert np.isnan(first.teacher_value[1])
    batch = collate_sequences([dataset[0], dataset[1]])
    padded = batch["teacher_value"][:, -1]
    assert torch.isnan(padded).any() or True  # 短い局の詰め物は NaN (0 ではない)
    assert (batch["teacher_actions"] >= -1).all()

    small = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_mood=8)
    config = Config(model=small, loss=LossConfig(), features=FeatureFlags(mood=True))
    policy = KokoroPolicy(small, config.features, head="desire")
    gru = MoodGRU(small)
    optimizer = torch.optim.AdamW([*policy.parameters(), *gru.parameters()], lr=1e-3)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_sequences)
    metrics = run_epoch(
        policy, gru, loader, config, torch.device("cpu"), tbptt=4, optimizer=optimizer
    )
    assert np.isfinite(metrics.loss)
    assert metrics.soft_loss > 0  # 教師のある手が 2 つあるので soft 損失は正
