"""文化リーグの淘汰 (train/league.py select_and_mutate) を検証する。

2026-09-04 に観測した「新生文化の回転ドア」対策 (猶予期間) と、対照実験用の
乱択淘汰・ハイパー固定が意図どおりに働くかを、モデルなしの偽 Culture で確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.train.league import Culture, select_and_mutate  # noqa: E402


def make_cultures(win_rates: list[float], ages: list[int] | None = None) -> list[Culture]:
    ages = ages or [5] * len(win_rates)
    return [
        Culture(
            name=f"culture{k}",
            theta_sp=torch.full((8, 4), float(k)) + torch.randn(8, 4) * 0.1,
            tau=0.8 + 0.1 * k,
            lambda_g=0.01 * (k + 1),
            win_rate=wr,
            age=age,
        )
        for k, (wr, age) in enumerate(zip(win_rates, ages, strict=True))
    ]


def test_fitness_replaces_lowest_with_copy_of_best() -> None:
    cultures = make_cultures([0.7, 0.5, 0.3])
    renewed, parent = select_and_mutate(cultures, np.random.default_rng(0), replaced=1)
    assert renewed == ["culture2"]
    assert parent == "culture0"
    assert cultures[2].age == 0
    # 複製+変異: 親に近く同一ではない
    assert torch.allclose(cultures[2].theta_sp, cultures[0].theta_sp, atol=0.1)
    assert not torch.equal(cultures[2].theta_sp, cultures[0].theta_sp)


def test_grace_protects_newborn() -> None:
    # 最下位 culture2 は生まれたて (age 0) → 猶予中なので次に低い culture1 が淘汰される
    cultures = make_cultures([0.7, 0.5, 0.3], ages=[5, 5, 0])
    renewed, _ = select_and_mutate(cultures, np.random.default_rng(0), replaced=1, grace=2)
    assert renewed == ["culture1"]
    assert cultures[2].win_rate == 0.3  # 触られていない


def test_grace_falls_back_to_oldest_when_all_protected() -> None:
    cultures = make_cultures([0.7, 0.5, 0.3], ages=[5, 1, 0])
    renewed, _ = select_and_mutate(cultures, np.random.default_rng(0), replaced=1, grace=2)
    assert renewed == ["culture1"]  # 最良を除いた中で最年長


def test_grace_zero_matches_legacy_behaviour() -> None:
    cultures = make_cultures([0.7, 0.5, 0.3], ages=[5, 5, 0])
    renewed, _ = select_and_mutate(cultures, np.random.default_rng(0), replaced=1, grace=0)
    assert renewed == ["culture2"]


def test_random_selection_ignores_win_rate() -> None:
    counts: dict[str, int] = {}
    parents: dict[str, int] = {}
    for seed in range(200):
        cultures = make_cultures([0.9, 0.5, 0.1])
        renewed, parent = select_and_mutate(
            cultures, np.random.default_rng(seed), replaced=1, selection="random"
        )
        counts[renewed[0]] = counts.get(renewed[0], 0) + 1
        parents[parent] = parents.get(parent, 0) + 1
        assert renewed[0] != parent
    # 勝率最上位の culture0 も淘汰され、最下位の culture2 もコピー元になる
    assert counts["culture0"] > 30 and parents["culture2"] > 30


def test_fix_hypers_inherits_parent_values() -> None:
    cultures = make_cultures([0.7, 0.5, 0.3])
    select_and_mutate(cultures, np.random.default_rng(0), replaced=1, fix_hypers=True)
    assert cultures[2].tau == cultures[0].tau
    assert cultures[2].lambda_g == cultures[0].lambda_g


def test_jittered_hypers_differ_from_parent() -> None:
    cultures = make_cultures([0.7, 0.5, 0.3])
    select_and_mutate(cultures, np.random.default_rng(0), replaced=1, fix_hypers=False)
    assert cultures[2].tau != cultures[0].tau


def test_ages_increment_only_for_survivors() -> None:
    # main() と同じ更新規則を模す: 置換されなかった文化だけ age += 1
    cultures = make_cultures([0.7, 0.5, 0.3], ages=[3, 3, 3])
    renewed, _ = select_and_mutate(cultures, np.random.default_rng(0), replaced=1)
    for culture in cultures:
        if culture.name not in renewed:
            culture.age += 1
    assert [c.age for c in cultures] == [4, 4, 0]
