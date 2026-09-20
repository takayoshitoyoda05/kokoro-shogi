"""関係性 [C] (model/relations.py) と会議 [D] (model/council.py) の性質を検証する。

重みが乱数でも成り立つ不変条件 (零初期化の同値性・値域・重み共有・議事録の形) を
固定する。学習の良し悪しは見ない。
"""

from __future__ import annotations

import cshogi
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import FeatureFlags, ModelConfig  # noqa: E402
from kokoro_shogi.core.effects import SUPPORTS_ALLY, piece_effect_matrix  # noqa: E402
from kokoro_shogi.core.piece_state import PieceIdTracker  # noqa: E402
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer  # noqa: E402
from kokoro_shogi.data.dataset import legal_move_mask  # noqa: E402
from kokoro_shogi.model.council import top_proposals  # noqa: E402
from kokoro_shogi.model.policy import KokoroPolicy  # noqa: E402
from kokoro_shogi.model.relations import (  # noqa: E402
    BOND_DELTA,
    bond_from_effect,
    initial_relations,
    update_relations,
)

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4, d_mood=8)


def tokenize_initial():
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    tokens = PieceTokenizer().tokenize(board, tracker)
    effect = piece_effect_matrix(board, tokens.squares()).astype(np.int64)
    legal = legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
    return tokens, effect, legal


def build_inputs(tokens, effect, legal):
    as_long = lambda a: torch.from_numpy(a.astype(np.int64))[None]
    return dict(
        species=as_long(tokens.species),
        position=as_long(tokens.position),
        owner=as_long(tokens.owner),
        promoted=as_long(tokens.promoted),
        mask=torch.from_numpy(tokens.mask)[None],
        turn=torch.tensor([tokens.turn]),
        effect=torch.from_numpy(effect)[None],
        legal=torch.from_numpy(legal)[None],
    )


# --- 関係性 [C] --------------------------------------------------------------


def test_bond_is_symmetric_and_in_range() -> None:
    tokens, effect, _ = tokenize_initial()
    bond = bond_from_effect(torch.from_numpy(effect)[None])

    assert torch.allclose(bond, bond.transpose(1, 2))  # 絆は対称
    assert bond.min() >= 0.0 and bond.max() <= 1.0
    assert bond.diagonal(dim1=1, dim2=2).sum() == 0.0  # 自分との絆はない
    # 初期局面には紐が多数ある (囲い以前でも金銀は互いに紐)
    assert (effect == SUPPORTS_ALLY).sum() > 0
    assert bond.sum() > 0


def test_relation_ema_accumulates_toward_bond() -> None:
    tokens, effect, _ = tokenize_initial()
    effect_t = torch.from_numpy(effect)[None]
    relation = initial_relations(1, MAX_PIECES)

    for _ in range(30):
        relation = update_relations(relation, effect_t)

    bond = bond_from_effect(effect_t)
    # 同じ局面を保てば R は bond へ収束する (EMAの不動点)
    assert torch.allclose(relation, bond * (1 - (1 - BOND_DELTA) ** 30), atol=1e-6)
    assert relation.max() <= 1.0


def test_relations_flag_off_keeps_checkpoint_compatibility() -> None:
    model = KokoroPolicy(SMALL, FeatureFlags(), head="desire")
    keys = model.state_dict().keys()
    assert not any("relation" in key or "proposal" in key for key in keys)


def test_relation_zero_init_matches_flag_off_output() -> None:
    torch.manual_seed(0)
    base = KokoroPolicy(SMALL, FeatureFlags(), head="desire")
    torch.manual_seed(0)
    with_rel = KokoroPolicy(SMALL, FeatureFlags(relations=True), head="desire")
    state = base.state_dict()
    missing, unexpected = with_rel.load_state_dict(state, strict=False)
    assert not unexpected and all("relation" in k for k in missing)

    tokens, effect, legal = tokenize_initial()
    inputs = build_inputs(tokens, effect, legal)
    relation = torch.rand(1, MAX_PIECES, MAX_PIECES)

    with torch.no_grad():
        plain = base(**inputs)
        biased = with_rel(**inputs, relation=relation)
    # 重みが零初期化なので、関係を渡してもOFFと完全一致から始まる
    assert torch.allclose(plain.logits, biased.logits, atol=1e-5)


# --- 会議 [D] ----------------------------------------------------------------


def test_top_proposals_picks_only_legal_moves() -> None:
    tokens, effect, legal = tokenize_initial()
    scores = torch.randn(1, MAX_PIECES, 81 * 2)
    masked = scores.masked_fill(
        ~torch.from_numpy(legal)[None].flatten(start_dim=2), -1e9
    )
    token, move_kind, bid = top_proposals(masked, k=8)

    assert token.shape == (1, 8) and bid.shape == (1, 8)
    legal_flat = legal.reshape(MAX_PIECES, -1)
    for token_i, move_i in zip(token[0].tolist(), move_kind[0].tolist(), strict=True):
        assert legal_flat[token_i, move_i]  # 提案は必ず合法手


def test_council_returns_logs_and_valid_distribution() -> None:
    model = KokoroPolicy(SMALL, FeatureFlags(council=True), head="desire")
    tokens, effect, legal = tokenize_initial()
    inputs = build_inputs(tokens, effect, legal)

    with torch.no_grad():
        output = model(**inputs, rounds=3)

    assert output.council is not None and len(output.council) == 3
    for round_log in output.council:
        assert round_log.bid.shape == (1, model.council_top_k)
    probs = output.probs()
    assert torch.isfinite(probs).all()
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
    # 合法手以外に確率が漏れない
    legal_mask = torch.from_numpy(legal)[None].flatten(start_dim=1)
    assert float(probs[~legal_mask].sum()) < 1e-6


def test_council_rounds_change_scores_but_r0_matches_off() -> None:
    """重み共有の検証: rounds=0 は会議OFFと同じ。rounds>0 でスコアが動く。"""
    torch.manual_seed(1)
    model = KokoroPolicy(SMALL, FeatureFlags(council=True), head="desire")
    tokens, effect, legal = tokenize_initial()
    inputs = build_inputs(tokens, effect, legal)

    with torch.no_grad():
        r0 = model(**inputs, rounds=0)
        r1 = model(**inputs, rounds=1)
        r2 = model(**inputs, rounds=2)

    assert r0.council == ()
    # rounds=0 は素の1パスと一致し、ラウンドを重ねるとスコアが更新される
    assert not torch.allclose(r0.logits, r1.logits)
    assert not torch.allclose(r1.logits, r2.logits)


def test_council_gradients_flow_through_rounds() -> None:
    model = KokoroPolicy(SMALL, FeatureFlags(council=True), head="desire")
    tokens, effect, legal = tokenize_initial()
    inputs = build_inputs(tokens, effect, legal)

    output = model(**inputs, rounds=2)
    loss = torch.logsumexp(output.logits, dim=-1).sum()
    loss.backward()
    assert model.proposal_embed.from_piece.weight.grad is not None
    assert model.proposal_embed.from_piece.weight.grad.abs().sum() > 0


def test_random_loop_sampling_varies_rounds_only_in_training() -> None:
    """council_round_choices は学習時だけラウンド数を散らし、評価時は既定 R に戻る。

    会議は trunk 最終2層の重み共有再適用なので、ラウンド数が変われば議事録の長さが変わる。
    それを数えることで、実際に引かれた R を外から観測できる。
    """
    from kokoro_shogi.config import FeatureFlags, ModelConfig
    from kokoro_shogi.model.policy import KokoroPolicy

    small = ModelConfig(d_model=32, n_layers=2, n_heads=4)
    model = KokoroPolicy(small, FeatureFlags(council=True), head="desire")
    assert model.council_round_choices == ()

    batch = _council_batch(small)
    # 既定 (choices 空) は学習中でもラウンド数が動かない
    model.train()
    assert {len(model(**batch).council) for _ in range(8)} == {model.council_rounds}

    model.council_round_choices = (1, 2, 3, 4)
    model.train()
    seen = {len(model(**batch).council) for _ in range(40)}
    assert seen <= {1, 2, 3, 4}
    assert len(seen) > 1, f"学習時にラウンド数が散らばっていない: {seen}"

    # 評価時は既定に固定される (物差しの再現性のため)
    model.eval()
    assert {len(model(**batch).council) for _ in range(8)} == {model.council_rounds}

    # rounds を明示したときは学習中でも指定が勝つ (R 掃引の評価が壊れない)
    model.train()
    assert {len(model(**batch, rounds=3).council) for _ in range(8)} == {3}


def _council_batch(config) -> dict:
    """会議が走る最小の入力 (平手初期局面)。"""
    import cshogi

    from kokoro_shogi.core.tokenizer import PieceTokenizer
    from kokoro_shogi.data.dataset import legal_move_mask

    board = cshogi.Board()
    tokens = PieceTokenizer().tokenize(board)
    legal = legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)

    def long_(array):
        return torch.from_numpy(array.astype("int64"))[None]

    return {
        "species": long_(tokens.species),
        "position": long_(tokens.position),
        "owner": long_(tokens.owner),
        "promoted": long_(tokens.promoted),
        "mask": torch.from_numpy(tokens.mask)[None],
        "turn": torch.zeros(1, dtype=torch.long),
        "legal": torch.from_numpy(legal)[None],
    }
