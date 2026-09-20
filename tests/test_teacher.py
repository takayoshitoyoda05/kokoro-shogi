"""エンジン教師 (scripts/ops/join_teacher.py → *.teacher.npz → 蒸留損失) の結合テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import cshogi
import numpy as np
import pytest
import torch

from kokoro_shogi.config import load_config
from kokoro_shogi.core.tokenizer import PieceTokenizer
from kokoro_shogi.data.dataset import (
    TEACHER_K,
    ShardDataset,
    action_index,
    collate,
    decode_action,
    legal_move_mask,
)
from kokoro_shogi.model.policy import KokoroPolicy
from kokoro_shogi.train.distill import compute_loss_tensors, teacher_soft_loss, value_target

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
from join_teacher import cp_to_value, usi_to_action  # noqa: E402


def test_cp_to_value_is_monotone_and_bounded() -> None:
    values = [cp_to_value(cp, None)[0] for cp in (-3000, -600, 0, 600, 3000)]
    assert values == sorted(values)
    assert values[2] == pytest.approx(0.0)
    assert -1.0 < values[0] < values[-1] < 1.0
    assert cp_to_value(None, 3) == (1.0, 30000.0)
    assert cp_to_value(None, -1)[0] == -1.0


def test_usi_to_action_maps_moves_and_drops_like_legal_mask() -> None:
    board = cshogi.Board()
    for usi in ("7g7f", "3c3d", "8h2b+", "3a2b"):  # 角交換 → 先手が角を持つ
        board.push(board.move_from_usi(usi))
    tokens = PieceTokenizer().tokenize(board)
    legal = legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
    for usi in ("B*5e", "2g2f", "6i7h"):
        action = usi_to_action(
            board, usi, tokens.species, tokens.position, tokens.owner, tokens.mask
        )
        assert action >= 0
        token, to_square, promote = decode_action(action)
        assert legal[token, to_square, promote], usi


def _write_shard(path: Path, count: int = 3) -> None:
    board = cshogi.Board()
    tokens = PieceTokenizer().tokenize(board)
    rows = {
        "species": np.stack([tokens.species] * count),
        "position": np.stack([tokens.position] * count),
        "owner": np.stack([tokens.owner] * count),
        "promoted": np.stack([tokens.promoted] * count),
        "mask": np.stack([tokens.mask] * count),
        "turn": np.zeros(count, dtype=np.int8),
        "move": np.zeros(count, dtype=np.int32),
        "move_token": np.zeros(count, dtype=np.int8),
        "move_to": np.zeros(count, dtype=np.int8),
        "move_promote": np.zeros(count, dtype=np.int8),
        "result": np.ones(count, dtype=np.int8),
        "labels": np.zeros((count, 40, 6), dtype=np.uint8),
        "game_index": np.zeros(count, dtype=np.int32),
    }
    np.savez(path, **rows)


def test_dataset_reads_teacher_side_file_and_defaults(tmp_path: Path) -> None:
    shard = tmp_path / "shard_0000.npz"
    _write_shard(shard)
    plain = ShardDataset([shard], with_legal=False, with_effect=False)
    assert plain.teacher_shards == 0
    assert np.isnan(plain[0].teacher_value)
    assert (plain[0].teacher_actions == -1).all()

    np.savez(
        shard.with_suffix(".teacher.npz"),
        teacher_value=np.array([0.3, np.nan, -0.2], dtype=np.float32),
        teacher_cp=np.array([200.0, np.nan, -120.0], dtype=np.float32),
        teacher_actions=np.array(
            [[action_index(0, 10, 0), action_index(1, 11, 0), -1, -1]] * 3, dtype=np.int64
        ),
        teacher_cps=np.array([[200.0, 150.0, np.nan, np.nan]] * 3, dtype=np.float32),
    )
    with_teacher = ShardDataset([shard], with_legal=False, with_effect=False)
    assert with_teacher.teacher_shards == 1
    assert with_teacher[0].teacher_value == pytest.approx(0.3)
    assert np.isnan(with_teacher[1].teacher_value)
    batch = collate([with_teacher[i] for i in range(3)])
    assert batch["teacher_actions"].shape == (3, TEACHER_K)
    assert batch["teacher_value"].shape == (3,)


def _fake_batch(batch_size: int, with_teacher: bool) -> dict[str, torch.Tensor]:
    board = cshogi.Board()
    tokens = PieceTokenizer().tokenize(board)
    legal = legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
    legal_actions = np.flatnonzero(legal.reshape(-1))
    batch = {
        "species": torch.from_numpy(np.stack([tokens.species] * batch_size)).long(),
        "position": torch.from_numpy(np.stack([tokens.position] * batch_size)).long(),
        "owner": torch.from_numpy(np.stack([tokens.owner] * batch_size)).long(),
        "promoted": torch.from_numpy(np.stack([tokens.promoted] * batch_size)).long(),
        "mask": torch.from_numpy(np.stack([tokens.mask] * batch_size)),
        "turn": torch.zeros(batch_size, dtype=torch.long),
        "legal": torch.from_numpy(np.stack([legal] * batch_size)),
        "action": torch.tensor([int(legal_actions[0])] * batch_size),
        "result": torch.ones(batch_size),
        "labels": torch.zeros(batch_size, 40, 6),
    }
    if with_teacher:
        actions = torch.full((batch_size, TEACHER_K), -1, dtype=torch.long)
        cps = torch.full((batch_size, TEACHER_K), float("nan"))
        actions[0, :2] = torch.tensor([int(legal_actions[0]), int(legal_actions[1])])
        cps[0, :2] = torch.tensor([100.0, 50.0])
        value = torch.full((batch_size,), float("nan"))
        value[0] = -0.4
        batch.update({"teacher_actions": actions, "teacher_cps": cps, "teacher_value": value})
    return batch


def test_value_target_mixes_only_where_teacher_exists() -> None:
    config = load_config()
    batch = _fake_batch(2, with_teacher=True)
    target = value_target(batch, config)
    assert target[0] == pytest.approx(config.loss.teacher_value_weight * -0.4
                                      + (1 - config.loss.teacher_value_weight) * 1.0)
    assert target[1] == pytest.approx(1.0)  # 教師なし → 勝敗 z


def test_soft_loss_is_finite_and_zero_without_teacher() -> None:
    config = load_config()
    model = KokoroPolicy.from_config(config, head="desire")
    model.eval()
    with torch.no_grad():
        batch = _fake_batch(2, with_teacher=True)
        output = model.forward_batch(batch)
        soft = teacher_soft_loss(output, batch, config.loss.teacher_temp)
        assert torch.isfinite(soft) and soft > 0
        loss, parts, _ = compute_loss_tensors(output, batch, config)
        assert torch.isfinite(loss)
        assert parts["soft_loss"] > 0

        plain = _fake_batch(2, with_teacher=False)
        output_plain = model.forward_batch(plain)
        loss_plain, parts_plain, _ = compute_loss_tensors(output_plain, plain, config)
        assert parts_plain["soft_loss"] == 0
        assert torch.isfinite(loss_plain)
