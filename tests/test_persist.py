"""永続化 [B] (persist/store.py) と個体性格・忠誠 [E] のモデル配線を検証する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import FeatureFlags, ModelConfig  # noqa: E402
from kokoro_shogi.model.policy import KAPPA_LOYALTY, KokoroPolicy  # noqa: E402
from kokoro_shogi.persist.store import PieceStore  # noqa: E402

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4, d_mood=8, d_theta=4)


@pytest.fixture()
def store(tmp_path: Path) -> PieceStore:
    with PieceStore(tmp_path / "pieces.sqlite3", d_individual=4) as opened:
        yield opened


def test_unknown_pieces_start_with_zero_theta(store: PieceStore) -> None:
    theta = store.load_theta(["P77_gen0_0003", "K59_gen0_0022"])
    assert theta.shape == (2, 4)
    assert (theta == 0).all()  # 新入りは無個性から


def test_theta_roundtrip(store: PieceStore) -> None:
    store.load_theta(["P77_gen0_0003"])
    vec = np.arange(4, dtype=np.float32)
    store.save_theta({"P77_gen0_0003": vec})
    assert np.allclose(store.load_theta(["P77_gen0_0003"])[0], vec)


def test_career_accumulates(store: PieceStore) -> None:
    ids = ["P77_gen0_0003", "R28_gen0_0009"]
    store.load_theta(ids)
    store.record_game(
        survived={ids[0]: True, ids[1]: False},
        promoted={ids[0]: False, ids[1]: True},
        contribution={ids[0]: 0.5, ids[1]: 2.0},
        mvp_id=ids[1],
    )
    store.record_game(
        survived={ids[0]: False, ids[1]: True},
        promoted={},
        contribution={ids[0]: -0.2},
        mvp_id=ids[0],
    )
    careers = {c["piece_id"]: c for c in store.careers()}
    assert careers[ids[0]]["games"] == 2
    assert careers[ids[0]]["survival_rate"] == pytest.approx(0.5)
    assert careers[ids[0]]["mvp_count"] == 1
    assert careers[ids[1]]["promotions"] == 1
    assert careers[ids[1]]["contribution"] == pytest.approx(2.0)


def test_breed_mixes_parents_and_records_lineage(store: PieceStore) -> None:
    parents = ["P77_gen0_0003", "P27_gen0_0008"]
    store.load_theta(parents)
    store.save_theta(
        {parents[0]: np.full(4, 1.0, np.float32), parents[1]: np.full(4, -1.0, np.float32)}
    )
    rng = np.random.default_rng(0)
    child = store.breed(parents[0], parents[1], "P77_gen1_0003", sigma=0.0, rng=rng)

    # 交叉: β∈[0,1] の内分点 (σ=0なのでノイズなし)
    assert child.min() >= -1.0 and child.max() <= 1.0
    assert (child == child[0]).all()  # 全次元同じβ
    assert store.lineage_of("P77_gen1_0003") == sorted(parents)
    # 子は世代1として登録され、θを読み戻せる
    assert np.allclose(store.load_theta(["P77_gen1_0003"])[0], child)


# --- モデル配線 [B]/[E] -------------------------------------------------------


def test_individual_flag_off_keeps_checkpoint_compatibility() -> None:
    model = KokoroPolicy(SMALL, FeatureFlags(), head="desire")
    assert not any("individual" in key for key in model.state_dict())


def test_individual_zero_init_matches_flag_off_hidden() -> None:
    torch.manual_seed(0)
    base = KokoroPolicy(SMALL, FeatureFlags(), head="desire")
    torch.manual_seed(0)
    with_ind = KokoroPolicy(SMALL, FeatureFlags(individual=True), head="desire")
    state = {
        k: v
        for k, v in base.state_dict().items()
        if not k.startswith("personality.project")
    }
    with_ind.load_state_dict(state, strict=False)

    import cshogi

    from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer

    tokens = PieceTokenizer().tokenize(cshogi.Board())
    as_long = lambda a: torch.from_numpy(a.astype("int64"))[None]
    inputs = dict(
        species=as_long(tokens.species),
        position=as_long(tokens.position),
        owner=as_long(tokens.owner),
        promoted=as_long(tokens.promoted),
        mask=torch.from_numpy(tokens.mask)[None],
        turn=torch.tensor([tokens.turn]),
    )
    theta = torch.randn(1, MAX_PIECES, SMALL.d_theta)
    with torch.no_grad():
        plain = base(**inputs)
        individual = with_ind(**inputs, individual=theta)
    # W_ind は零初期化なので trunk 出力は完全一致 (personality経由の差だけが残る)
    assert torch.allclose(plain.hidden, individual.hidden, atol=1e-6)


def test_loyalty_handicap_shrinks_defector_scores() -> None:
    """忠誠ハンデ [E]: 寝返り駒の欲求が (1-κ_E·loyalty) 倍に縮む。"""
    import cshogi

    from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer

    torch.manual_seed(2)
    model = KokoroPolicy(SMALL, FeatureFlags(loyalty=True), head="desire")
    tokens = PieceTokenizer().tokenize(cshogi.Board())
    as_long = lambda a: torch.from_numpy(a.astype("int64"))[None]
    inputs = dict(
        species=as_long(tokens.species),
        position=as_long(tokens.position),
        owner=as_long(tokens.owner),
        promoted=as_long(tokens.promoted),
        mask=torch.from_numpy(tokens.mask)[None],
        turn=torch.tensor([tokens.turn]),
    )

    loyalty = torch.zeros(1, MAX_PIECES)
    loyalty[0, 5] = 1.0  # トークン5が寝返り駒だとする
    with torch.no_grad():
        base = model(**inputs)
        handicapped = model(**inputs, loyalty=loyalty)

    ratio = handicapped.desire[0, 5] / base.desire[0, 5].clamp(min=1e-8)
    assert torch.allclose(ratio, torch.full_like(ratio, 1 - KAPPA_LOYALTY), atol=1e-5)
    # 他の駒は不変
    assert torch.allclose(handicapped.desire[0, 6], base.desire[0, 6], atol=1e-6)
