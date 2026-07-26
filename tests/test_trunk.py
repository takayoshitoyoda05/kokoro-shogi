"""共有Transformer trunk (model/trunk.py) の性質を検証する。

DESIGN.md §3(2) の attention バイアス

$$e_{ij} = \\frac{(x_i W_Q)(x_j W_K)^\\top}{\\sqrt{d}} + B_{利き}(i,j) + r_{ij}$$

は「スコアへの加算」なので、加算する値が全要素で同じなら softmax で打ち消えて
出力は変わらない。この当たり前の性質が壊れていないことを確かめる。

`test_nonzero_attention_bias_stays_finite` は素の `nn.TransformerEncoderLayer` に
戻したときに落ちる回帰テスト (KokoroEncoderLayer の docstring 参照)。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import ModelConfig  # noqa: E402
from kokoro_shogi.core.tokenizer import MAX_PIECES, NUM_POSITIONS  # noqa: E402
from kokoro_shogi.model.trunk import KokoroTrunk  # noqa: E402

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4)
BATCH = 3


def dummy_tokens(seed: int = 0, valid: int = MAX_PIECES) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(BATCH, MAX_PIECES, dtype=torch.bool)
    mask[:, :valid] = True
    return {
        "species": torch.randint(0, 14, (BATCH, MAX_PIECES), generator=generator),
        "position": torch.randint(0, NUM_POSITIONS, (BATCH, MAX_PIECES), generator=generator),
        "owner": torch.randint(0, 2, (BATCH, MAX_PIECES), generator=generator),
        "promoted": torch.randint(0, 2, (BATCH, MAX_PIECES), generator=generator),
        "mask": mask,
        "turn": torch.randint(0, 2, (BATCH,), generator=generator),
    }


@pytest.fixture(scope="module")
def trunk() -> KokoroTrunk:
    torch.manual_seed(0)
    return KokoroTrunk(SMALL).eval()


def test_output_shape(trunk: KokoroTrunk) -> None:
    with torch.no_grad():
        hidden = trunk(**dummy_tokens())
    assert hidden.shape == (BATCH, MAX_PIECES, SMALL.d_model)
    assert torch.isfinite(hidden).all()


def test_nonzero_attention_bias_stays_finite(trunk: KokoroTrunk) -> None:
    """利きバイアスが非ゼロでも出力が有限。

    PyTorch 2.13 の `nn.TransformerEncoderLayer` は norm_first=True で非ゼロの
    float attn_mask を渡すと NaN を返す。そこへ戻した瞬間に落ちるよう固定しておく。
    """
    tokens = dummy_tokens(seed=1)
    effect = torch.randint(0, 3, (BATCH, MAX_PIECES, MAX_PIECES))

    # 学習で 0 から離れた状態を再現する
    with torch.no_grad():
        trunk.effect_bias.weight.normal_(std=0.5)
        hidden = trunk(**tokens, effect=effect)
        trunk.effect_bias.weight.zero_()

    assert torch.isfinite(hidden).all()


def test_uniform_bias_does_not_change_the_output(trunk: KokoroTrunk) -> None:
    """全要素が同じバイアスは softmax で打ち消える (加算バイアスであることの確認)。"""
    tokens = dummy_tokens(seed=2)
    uniform = torch.zeros(BATCH, MAX_PIECES, MAX_PIECES, dtype=torch.long)

    with torch.no_grad():
        without = trunk(**tokens)
        trunk.effect_bias.weight.fill_(0.7)  # 全種別に同じ値 → 実質どこも同じ加算
        with_bias = trunk(**tokens, effect=uniform)
        trunk.effect_bias.weight.zero_()

    torch.testing.assert_close(without, with_bias, atol=1e-5, rtol=1e-5)


def test_masked_tokens_are_zeroed_and_ignored(trunk: KokoroTrunk) -> None:
    """無効トークンは出力が0で、他のトークンの表現にも影響しない。

    5五将棋 (10枚) など駒数の少ない変種へ持っていくときに効く性質。
    """
    tokens = dummy_tokens(seed=3, valid=12)

    with torch.no_grad():
        before = trunk(**tokens)

        noisy = dict(tokens)
        generator = torch.Generator().manual_seed(99)
        for key in ("species", "position", "owner", "promoted"):
            changed = tokens[key].clone()
            changed[:, 12:] = torch.randint(
                0, 2 if key in ("owner", "promoted") else 14,
                (BATCH, MAX_PIECES - 12),
                generator=generator,
            )
            noisy[key] = changed
        after = trunk(**noisy)

    assert float(before[:, 12:].abs().max()) == 0.0
    torch.testing.assert_close(before[:, :12], after[:, :12])
