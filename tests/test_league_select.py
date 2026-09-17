"""文化リーグの淘汰 (train/league.py select_and_mutate) を検証する。

2026-09-04 に観測した「新生文化の回転ドア」対策 (猶予期間) と、対照実験用の
乱択淘汰・ハイパー固定が意図どおりに働くかを、モデルなしの偽 Culture で確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.train.league import (  # noqa: E402
    Culture,
    crossover_theta,
    select_and_mutate,
)


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


# --- ES 交叉 (2026-09-11 追加) ----------------------------------------------


def make_flat_cultures(values: list[float], win_rates: list[float]) -> list[Culture]:
    """θ_sp が行ごとに定数の文化。交叉で「どの親から来た行か」を判定できる。"""
    return [
        Culture(
            name=f"culture{k}",
            theta_sp=torch.full((8, 4), float(value)),
            tau=0.8,
            lambda_g=0.01,
            win_rate=wr,
        )
        for k, (value, wr) in enumerate(zip(values, win_rates, strict=True))
    ]


def test_crossover_none_keeps_single_parent_copy() -> None:
    cultures = make_flat_cultures([0.0, 1.0, 2.0], [0.7, 0.5, 0.3])
    renewed, parent = select_and_mutate(
        cultures, np.random.default_rng(0), replaced=1, sigma_relative=0.0
    )
    assert renewed == ["culture2"] and parent == "culture0"
    assert torch.equal(cultures[2].theta_sp, cultures[0].theta_sp)


def test_uniform_crossover_mixes_two_parents_per_species() -> None:
    cultures = make_flat_cultures([0.0, 1.0, 2.0], [0.7, 0.5, 0.3])
    renewed, parent = select_and_mutate(
        cultures, np.random.default_rng(0), replaced=1,
        sigma_relative=0.0, crossover="uniform",
    )
    assert renewed == ["culture2"]
    assert parent == "culture0+culture1"  # 淘汰対象を除いた上位2つ
    child = cultures[2].theta_sp
    # 親は 0.0 と 1.0 の定数なので、子の各成分は β そのもの
    assert child.min() >= 0.0 and child.max() <= 1.0
    per_species = child[:, 0]
    assert len(torch.unique(per_species)) == 8  # 駒種ごとに独立に β を引いている
    assert torch.allclose(child, per_species[:, None].expand_as(child))  # 行内は同じ β


def test_blend_crossover_uses_one_beta_for_whole_phi() -> None:
    cultures = make_flat_cultures([0.0, 1.0, 2.0], [0.7, 0.5, 0.3])
    select_and_mutate(
        cultures, np.random.default_rng(0), replaced=1,
        sigma_relative=0.0, crossover="blend",
    )
    child = cultures[2].theta_sp
    assert len(torch.unique(child)) == 1  # Φ 全体で β は 1 つ
    assert 0.0 < float(child[0, 0]) < 1.0


def test_crossover_never_mates_with_the_replaced_culture() -> None:
    # 2文化しかなければ相手がいないので単親コピーに落ちる
    cultures = make_flat_cultures([0.0, 1.0], [0.7, 0.3])
    renewed, parent = select_and_mutate(
        cultures, np.random.default_rng(0), replaced=1,
        sigma_relative=0.0, crossover="uniform",
    )
    assert renewed == ["culture1"] and parent == "culture0"
    assert torch.equal(cultures[1].theta_sp, cultures[0].theta_sp)


def test_crossover_resets_inherited_fitness() -> None:
    cultures = make_flat_cultures([0.0, 1.0, 2.0], [0.7, 0.5, 0.3])
    for culture in cultures:
        culture.fitness = culture.win_rate
    select_and_mutate(
        cultures, np.random.default_rng(0), replaced=1, crossover="uniform"
    )
    assert cultures[2].fitness is None  # 生まれ直した文化は過去の適応度を引き継がない


# --- 適応度 EMA / 変異のみ --------------------------------------------------


def test_fitness_ema_overrides_last_generation_win_rate() -> None:
    # その世代の勝率では culture0 が最下位だが、EMA では最上位 → 淘汰されない
    cultures = make_cultures([0.2, 0.5, 0.6])
    cultures[0].fitness, cultures[1].fitness, cultures[2].fitness = 0.9, 0.5, 0.2
    renewed, parent = select_and_mutate(cultures, np.random.default_rng(0), replaced=1)
    assert parent == "culture0"
    assert renewed == ["culture2"]


def test_drift_mutates_everyone_and_replaces_nobody() -> None:
    cultures = make_cultures([0.7, 0.5, 0.3])  # ages は既定の 5
    before = [c.theta_sp.clone() for c in cultures]
    renewed, parent = select_and_mutate(
        cultures, np.random.default_rng(0), replaced=1, selection="drift"
    )
    assert renewed == [] and parent == "-"
    for culture, original in zip(cultures, before, strict=True):
        assert not torch.equal(culture.theta_sp, original)  # 全員が動く
        assert culture.age == 5  # 誰も生まれ直していない
    assert [c.win_rate for c in cultures] == [0.7, 0.5, 0.3]  # 勝率は触らない


# --- BLX-α (2026-09-11: 交叉が分散を縮める問題への対処) ---------------------


def variance_ratio(alpha: float, seed: int = 0) -> float:
    """独立な二親を交叉したとき、子の分散が親の分散の何倍になるかを実測する。"""
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    parents = torch.randn(2, 4096, 1, generator=generator)
    children = torch.stack([
        crossover_theta(parents[0], parents[1], "uniform", rng, alpha=alpha)
        for _ in range(64)
    ])
    return float(children.var()) / float(parents.var())


def test_plain_crossover_shrinks_variance_to_two_thirds() -> None:
    # E[β²]+E[(1-β)²] = 1/3 + 1/3。子は必ず両親の内側にしか置けない
    assert variance_ratio(0.0) == pytest.approx(2 / 3, rel=0.05)


def test_blx_alpha_expands_variance_as_the_formula_says() -> None:
    # 倍率 = 2((1+2α)²/12 + 1/4)
    for alpha in (0.366, 0.5):
        expected = 2 * ((1 + 2 * alpha) ** 2 / 12 + 0.25)
        assert variance_ratio(alpha) == pytest.approx(expected, rel=0.05)
    assert variance_ratio(0.366) == pytest.approx(1.0, rel=0.05)  # α≈0.366 で中立


def test_blx_alpha_places_children_outside_the_parents() -> None:
    rng = np.random.default_rng(0)
    parents = (torch.zeros(64, 1), torch.ones(64, 1))
    inside = crossover_theta(*parents, "uniform", rng, alpha=0.0)
    assert inside.min() >= 0.0 and inside.max() <= 1.0
    outside = crossover_theta(*parents, "uniform", rng, alpha=0.5)
    assert outside.min() < 0.0 or outside.max() > 1.0  # 親の外にも子が置ける
    assert outside.min() >= -0.5 and outside.max() <= 1.5  # ただし U(-α, 1+α) の範囲内


def test_crossover_alpha_reaches_select_and_mutate() -> None:
    cultures = make_flat_cultures([0.0, 1.0, 2.0], [0.7, 0.5, 0.3])
    select_and_mutate(
        cultures, np.random.default_rng(3), replaced=1,
        sigma_relative=0.0, crossover="uniform", crossover_alpha=0.5,
    )
    child = cultures[2].theta_sp  # 親は 0.0 と 1.0 の定数なので子の値 = β
    assert child.min() < 0.0 or child.max() > 1.0
