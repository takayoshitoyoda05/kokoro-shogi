"""機能フラグ (configs/features.yaml) の性質を検証する。

DESIGN.md 設計原則1「面白さ機能を外しても素の強いpolicy netが残る」と
設計原則2「全機能は configs の機能フラグで個別にON/OFFできる」を、
テストで固定する。

Phase 1 時点では mood [A] / relations [C] / council [D] などが未実装なので、
フラグを立てても出力は変わらない。それでも**フラグを立てて壊れないこと**を
今のうちにテストしておくと、Phase 3 以降で機能を足すときに
「フラグOFFの経路を壊した」ことに即座に気づける。

参照: configs/features.yaml / DESIGN.md §3
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from kokoro_shogi.config import FeatureFlags, ModelConfig, load_config

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.core.tokenizer import MAX_PIECES, NUM_POSITIONS  # noqa: E402
from kokoro_shogi.data.dataset import NUM_ACTIONS  # noqa: E402
from kokoro_shogi.model.policy import KokoroPolicy  # noqa: E402

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4)
BATCH = 3

FLAG_NAMES = tuple(field.name for field in fields(FeatureFlags))


def dummy_batch(seed: int = 0) -> dict[str, torch.Tensor]:
    """形だけ正しいランダム入力 (盤面としての整合は問わない)。"""
    generator = torch.Generator().manual_seed(seed)
    return {
        "species": torch.randint(0, 14, (BATCH, MAX_PIECES), generator=generator),
        "position": torch.randint(0, NUM_POSITIONS, (BATCH, MAX_PIECES), generator=generator),
        "owner": torch.randint(0, 2, (BATCH, MAX_PIECES), generator=generator),
        "promoted": torch.randint(0, 2, (BATCH, MAX_PIECES), generator=generator),
        "mask": torch.ones(BATCH, MAX_PIECES, dtype=torch.bool),
        "turn": torch.randint(0, 2, (BATCH,), generator=generator),
        "effect": torch.randint(0, 3, (BATCH, MAX_PIECES, MAX_PIECES), generator=generator),
    }


def test_default_flags_are_all_false() -> None:
    """既定構成 = 素の強い将棋AI (DESIGN.md 設計原則1)。"""
    assert not any(getattr(FeatureFlags(), name) for name in FLAG_NAMES)


def test_repository_config_matches_the_dataclass() -> None:
    """configs/features.yaml のキーが FeatureFlags と一致する。

    yaml 側にtypoがあると load_config が例外を出す (config.py の _build)。
    """
    features = load_config().features
    assert set(FLAG_NAMES) == {field.name for field in fields(type(features))}


def test_minimal_configuration_runs() -> None:
    """全フラグ false の最小構成で forward が例外なく通る。"""
    torch.manual_seed(0)
    model = KokoroPolicy(SMALL, FeatureFlags()).eval()

    with torch.no_grad():
        output = model.forward_batch(dummy_batch())

    assert output.logits.shape == (BATCH, NUM_ACTIONS)
    assert output.value.shape == (BATCH,)
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize("flag", FLAG_NAMES)
def test_each_flag_keeps_shape_and_normalisation(flag: str) -> None:
    """フラグを1つずつ true にしても出力の形状と Σπ = 1 が保たれる。"""
    torch.manual_seed(0)
    model = KokoroPolicy(SMALL, replace(FeatureFlags(), **{flag: True})).eval()

    with torch.no_grad():
        output = model.forward_batch(dummy_batch(seed=1))

    assert output.logits.shape == (BATCH, NUM_ACTIONS)
    torch.testing.assert_close(output.probs().sum(dim=-1), torch.ones(BATCH))


def test_effect_bias_starts_neutral() -> None:
    """利きバイアスの初期値は0なので、学習開始時は素のattentionと一致する。

    DESIGN.md §3(2) のバイアスが「後から効いてくる補助構造」であることの担保。
    """
    torch.manual_seed(0)
    model = KokoroPolicy(SMALL, FeatureFlags()).eval()
    batch = dummy_batch(seed=2)

    with torch.no_grad():
        with_effect = model.forward_batch(batch).logits
        without = model.forward_batch({**batch, "effect": None}).logits

    torch.testing.assert_close(with_effect, without)
