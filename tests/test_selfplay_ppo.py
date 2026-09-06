"""自己対戦PPO (train/selfplay_ppo.py) の数式部分を検証する。

対戦・更新の統合はスモーク (実行時) に任せ、ここでは negamax GAE の
符号規約 — 「価値・報酬・advantage は常に手番側視点」 — を固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.train.selfplay_ppo import GAMMA, LAMBDA, negamax_gae  # noqa: E402


def test_terminal_reward_goes_to_the_last_mover() -> None:
    """価値がすべて0なら、最終手の advantage = 勝者視点の±1。"""
    values = np.zeros(4, dtype=np.float32)
    turns = np.array([0, 1, 0, 1])  # 最後に指したのは後手
    adv = negamax_gae(values, turns, winner=1)

    assert adv[-1] == pytest.approx(1.0)  # 勝った側の最終手は+1
    # 1手前 (敗者の手番) には符号反転して伝播する
    assert adv[-2] == pytest.approx(-GAMMA * LAMBDA)
    assert adv[-3] == pytest.approx((GAMMA * LAMBDA) ** 2)


def test_loser_last_move_gets_negative_advantage() -> None:
    values = np.zeros(3, dtype=np.float32)
    turns = np.array([0, 1, 0])  # 最後は先手の手番
    adv = negamax_gae(values, turns, winner=1)  # 後手勝ち
    assert adv[-1] == pytest.approx(-1.0)


def test_draw_has_zero_reward() -> None:
    values = np.zeros(5, dtype=np.float32)
    turns = np.array([0, 1, 0, 1, 0])
    adv = negamax_gae(values, turns, winner=None)
    assert np.allclose(adv, 0.0)


def test_overconfident_value_yields_negative_advantage() -> None:
    """勝てると読んでいた (V=0.9) のに引き分けなら advantage は負。"""
    values = np.full(2, 0.9, dtype=np.float32)
    turns = np.array([0, 1])
    adv = negamax_gae(values, turns, winner=None)
    assert adv[-1] < 0
    # δ_{T-1} = 0 + (-0) - 0.9 = -0.9
    assert adv[-1] == pytest.approx(-0.9)


def test_lambda_zero_reduces_to_td_residual() -> None:
    from kokoro_shogi.train import selfplay_ppo

    values = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    turns = np.array([0, 1, 0])
    adv = negamax_gae(values, turns, winner=0)
    # 手計算 (γ=1, λ=0.95): 後ろから
    d2 = 1.0 + 0.0 - 0.3
    d1 = -0.3 - (-0.2)
    d0 = 0.2 - 0.1
    a2 = d2
    a1 = d1 + LAMBDA * (-a2)
    a0 = d0 + LAMBDA * (-a1)
    assert adv[2] == pytest.approx(a2)
    assert adv[1] == pytest.approx(a1)
    assert adv[0] == pytest.approx(a0)


def _rollout_with_window(bptt: int, plies: int):
    from kokoro_shogi.config import load_config
    from kokoro_shogi.model.mood import MoodGRU
    from kokoro_shogi.model.policy import KokoroPolicy
    from kokoro_shogi.train.selfplay_ppo import SelfPlayEnv

    config = load_config()
    torch.manual_seed(0)
    gru = MoodGRU(config.model)
    model = KokoroPolicy(config.model, config.features, head="desire")
    env = SelfPlayEnv(model, gru, torch.device("cpu"), bptt=bptt)
    observations = []
    for _ in range(plies):
        obs = env.observe(gru)
        assert obs is not None
        observations.append(obs)
        move = next(iter(env.board.legal_moves))
        env.moves.append(move)
        env.record = env.tracker.apply_move(env.board, move)
        env.board.push(move)
    return gru, observations


def test_recompute_mood_matches_rollout_including_short_windows() -> None:
    from kokoro_shogi.train.selfplay_ppo import recompute_mood

    gru, observations = _rollout_with_window(bptt=4, plies=7)
    assert [o["window_len"] for o in observations] == [1, 2, 3, 4, 4, 4, 4]
    mood = recompute_mood(gru, observations, torch.device("cpu"))
    stored = torch.from_numpy(np.stack([o["mood"] for o in observations]))
    assert torch.allclose(mood, stored, atol=1e-5)


def test_recompute_mood_propagates_gradient_to_gru() -> None:
    from kokoro_shogi.train.selfplay_ppo import recompute_mood

    gru, observations = _rollout_with_window(bptt=3, plies=5)
    mood = recompute_mood(gru, observations, torch.device("cpu"))
    mood.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gru.parameters())
