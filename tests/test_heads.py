"""ヘッド出力 (model/heads.py) の性質を検証する。

Phase 1 の `PolicyHead` / `ValueHead` に加え、Phase 2 の欲求ヘッド・性格重み・
自由項・単調mixing を検証する。

Phase 2 で一番大事なのは**正制約**: $w_i > 0$ と $\\alpha_i \\ge 0$。
これが崩れると「欲求が上がればスコアも上がる」「駒の価値判断が全体の形勢に
単調に効く」という読み方ができなくなり、解釈可能性の主張が成り立たなくなる。

参照: DESIGN.md §3(3)-(8)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import ModelConfig  # noqa: E402
from kokoro_shogi.core.squares import NUM_SQUARES  # noqa: E402
from kokoro_shogi.core.tokenizer import MAX_PIECES  # noqa: E402
from kokoro_shogi.data.labels import NUM_AXES  # noqa: E402
from kokoro_shogi.model.heads import (  # noqa: E402
    NUM_MOVE_KINDS,
    NUM_PROMOTE,
    DesireHead,
    FreeTermHead,
    MonotonicValueMixing,
    PersonalityWeights,
    PolicyHead,
    ValueHead,
)
from kokoro_shogi.model.trunk import NUM_SPECIES  # noqa: E402

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4)
BATCH = 4


def test_policy_head_shape() -> None:
    """手スコアは (駒, 移動先, 成り) の3つ組で出る。"""
    head = PolicyHead(SMALL)
    hidden = torch.randn(BATCH, MAX_PIECES, SMALL.d_model)

    scores = head(hidden)

    assert scores.shape == (BATCH, MAX_PIECES, NUM_SQUARES, NUM_PROMOTE)
    assert torch.isfinite(scores).all()


def test_policy_head_is_per_token() -> None:
    """ある駒の表現を変えても、他の駒の手スコアは動かない。

    手スコアが駒ごとに独立していること (調停は softmax 側でやる) の確認。
    """
    head = PolicyHead(SMALL).eval()
    hidden = torch.randn(1, MAX_PIECES, SMALL.d_model)

    with torch.no_grad():
        before = head(hidden)
        changed = hidden.clone()
        changed[0, 3] = torch.randn(SMALL.d_model)
        after = head(changed)

    assert not torch.allclose(before[0, 3], after[0, 3])
    torch.testing.assert_close(
        torch.cat([before[0, :3], before[0, 4:]]), torch.cat([after[0, :3], after[0, 4:]])
    )


def test_value_head_range() -> None:
    """状態価値は tanh で [-1, +1] に収まる (勝敗 z と同じ値域)。"""
    head = ValueHead(SMALL)
    hidden = torch.randn(BATCH, MAX_PIECES, SMALL.d_model) * 10
    mask = torch.ones(BATCH, MAX_PIECES, dtype=torch.bool)

    value = head(hidden, mask)

    assert value.shape == (BATCH,)
    assert bool((value.abs() <= 1.0).all())


def test_value_head_ignores_masked_tokens() -> None:
    """無効トークンの中身は状態価値に影響しない。

    5五将棋 (10枚) など駒数の少ない変種へ持っていくときに効く性質。
    """
    head = ValueHead(SMALL).eval()
    hidden = torch.randn(1, MAX_PIECES, SMALL.d_model)
    mask = torch.zeros(1, MAX_PIECES, dtype=torch.bool)
    mask[0, :10] = True

    with torch.no_grad():
        before = head(hidden, mask)
        noisy = hidden.clone()
        noisy[0, 10:] = torch.randn(MAX_PIECES - 10, SMALL.d_model) * 100
        after = head(noisy, mask)

    torch.testing.assert_close(before, after)


# --- Phase 2: 欲求による構造化 (DESIGN.md §3(3)-(5)(8)) ------------------------


def test_personality_weights_are_positive() -> None:
    """性格重みが $w_i > 0$ を常に満たす (DESIGN.md §3(4))。

    softplus の正制約が解釈可能性の前提。極端な入力でも0以下にならないことを見る。
    """
    torch.manual_seed(0)
    head = PersonalityWeights(SMALL, num_species=NUM_SPECIES)
    with torch.no_grad():  # 重みを大きく振っても符号は変わらないはず
        head.project.weight.normal_(std=50.0)
        head.project.bias.normal_(std=50.0)
        head.theta_species.weight.normal_(std=50.0)

    species = torch.randint(0, NUM_SPECIES, (BATCH, MAX_PIECES))
    with torch.no_grad():
        weights = head(species)

    assert weights.shape == (BATCH, MAX_PIECES, NUM_AXES)
    assert bool((weights > 0).all())
    assert torch.isfinite(weights).all()


def test_mixing_coefficients_are_non_negative() -> None:
    """単調mixingの係数が $\\alpha_i \\ge 0$ を満たす (DESIGN.md §3(8) IGM/単調性)。"""
    torch.manual_seed(0)
    mixing = MonotonicValueMixing(SMALL, max_pieces=MAX_PIECES)
    hidden = torch.randn(BATCH, MAX_PIECES, SMALL.d_model) * 5
    mask = torch.ones(BATCH, MAX_PIECES, dtype=torch.bool)

    with torch.no_grad():
        value, piece_value, alpha = mixing(hidden, mask)

    assert bool((alpha >= 0).all())
    torch.testing.assert_close(alpha.sum(dim=1), torch.ones(BATCH))
    assert value.shape == (BATCH,)
    assert bool((value.abs() <= 1.0).all())
    assert piece_value.shape == (BATCH, MAX_PIECES)


def test_value_is_monotonic_in_each_piece_value() -> None:
    """$\\partial V/\\partial V_i \\ge 0$ (IGM)。駒の価値を上げれば $V$ は下がらない。

    $\\alpha_i \\ge 0$ と $\\tanh' > 0$ から従う性質を、実際に勾配を取って確かめる。
    """
    torch.manual_seed(0)
    mixing = MonotonicValueMixing(SMALL, max_pieces=MAX_PIECES)
    hidden = torch.randn(1, MAX_PIECES, SMALL.d_model, requires_grad=True)
    mask = torch.ones(1, MAX_PIECES, dtype=torch.bool)

    value, piece_value, _ = mixing(hidden, mask)
    gradient = torch.autograd.grad(value.sum(), piece_value, retain_graph=True)[0]

    assert bool((gradient >= 0).all())


def test_mixing_ignores_masked_tokens() -> None:
    """無効トークンは alpha が0で、V にも寄与しない。"""
    torch.manual_seed(0)
    mixing = MonotonicValueMixing(SMALL, max_pieces=MAX_PIECES)
    hidden = torch.randn(1, MAX_PIECES, SMALL.d_model)
    mask = torch.zeros(1, MAX_PIECES, dtype=torch.bool)
    mask[0, :10] = True

    with torch.no_grad():
        before, _, alpha = mixing(hidden, mask)
        noisy = hidden.clone()
        noisy[0, 10:] = torch.randn(MAX_PIECES - 10, SMALL.d_model) * 100
        after, _, _ = mixing(noisy, mask)

    assert float(alpha[0, 10:].abs().max()) == 0.0
    torch.testing.assert_close(before, after)


def test_desire_head_is_in_unit_range() -> None:
    """欲求 $d_i(a) \\in [0,1]^6$ (BCEの教師と同じ値域)。"""
    torch.manual_seed(0)
    head = DesireHead(SMALL)
    hidden = torch.randn(BATCH, MAX_PIECES, SMALL.d_model) * 10

    with torch.no_grad():
        desire = head(hidden)

    assert desire.shape == (BATCH, MAX_PIECES, NUM_MOVE_KINDS, NUM_AXES)
    assert bool(((desire >= 0) & (desire <= 1)).all())


def test_score_increases_with_desire() -> None:
    """$s_{i,a} = \\langle w_i, d_i(a)\\rangle + g_i(a)$ が欲求に対して単調増加。

    $w_i > 0$ の意味そのもの。「生存欲が上がったのにスコアが下がる」ことが
    起きないので、性格重みを「その駒が何を重視しているか」と読める。
    """
    torch.manual_seed(0)
    personality = PersonalityWeights(SMALL, num_species=NUM_SPECIES)
    species = torch.randint(0, NUM_SPECIES, (1, MAX_PIECES))

    with torch.no_grad():
        weights = personality(species)

    desire = torch.rand(1, MAX_PIECES, NUM_MOVE_KINDS, NUM_AXES)
    free_term = torch.randn(1, MAX_PIECES, NUM_MOVE_KINDS)
    before = (desire * weights.unsqueeze(2)).sum(-1) + free_term

    raised = desire.clone()
    raised[..., 0] = torch.clamp(raised[..., 0] + 0.1, max=1.0)
    after = (raised * weights.unsqueeze(2)).sum(-1) + free_term

    assert bool((after >= before - 1e-6).all())


def test_free_term_head_shape() -> None:
    """自由項はスカラー (駒 × 手ごとに1つ)。"""
    torch.manual_seed(0)
    head = FreeTermHead(SMALL)
    with torch.no_grad():
        free_term = head(torch.randn(BATCH, MAX_PIECES, SMALL.d_model))

    assert free_term.shape == (BATCH, MAX_PIECES, NUM_MOVE_KINDS)
    assert torch.isfinite(free_term).all()
