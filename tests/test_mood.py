"""感情GRU (model/mood.py) の性質を検証する (DESIGN.md §3(9) [A])。

学習の良し悪しではなく、イベント特徴の意味論 (誰に・どれだけ届くか) と
モジュールの不変条件 (形状・値域・チェックポイント互換) を見るテスト。
"""

from __future__ import annotations

import cshogi
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch は dependency-groups の train 側")

from kokoro_shogi.config import FeatureFlags, ModelConfig  # noqa: E402
from kokoro_shogi.core.piece_state import PieceIdTracker  # noqa: E402
from kokoro_shogi.core.squares import str_to_sq  # noqa: E402
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer  # noqa: E402
from kokoro_shogi.model.mood import (  # noqa: E402
    ALLY_LOST,
    CAPTURED,
    ENEMY_LOST,
    KING_IN_CHECK,
    MOVED,
    NUM_EVENT_FEATURES,
    THREATENED,
    WAS_CAPTURED,
    MoodGRU,
    MoodProjection,
    build_event_features,
)
from kokoro_shogi.model.policy import KokoroPolicy  # noqa: E402

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4, d_mood=8)


def play(moves_usi: list[str]) -> tuple[cshogi.Board, PieceIdTracker, object]:
    """USIの手を順に適用し、(盤面, tracker, 最後のMoveRecord) を返す。"""
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    record = None
    for usi in moves_usi:
        move = board.move_from_usi(usi)
        record = tracker.apply_move(board, move)
        board.push(move)
    return board, tracker, record


def token_index(tracker: PieceIdTracker, piece_id: str) -> int:
    """piece_id 昇順 (トークンの並び) での位置。"""
    return sorted(tracker.states).index(piece_id)


def test_initial_position_has_no_move_events() -> None:
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    features = build_event_features(board, tracker, None)

    assert features.shape == (MAX_PIECES, NUM_EVENT_FEATURES)
    # 初期局面は手が無いので、動いた/取った/取られた/被取距離減衰は全て0
    for axis in (MOVED, CAPTURED, WAS_CAPTURED, ALLY_LOST, ENEMY_LOST, KING_IN_CHECK):
        assert features[:, axis].sum() == 0.0


def test_moved_flag_lands_on_the_moving_piece() -> None:
    board, tracker, record = play(["7g7f"])
    features = build_event_features(board, tracker, record)

    index = token_index(tracker, record.piece_id)
    assert features[index, MOVED] == 1.0
    assert features[:, MOVED].sum() == 1.0  # 動いたのは1枚だけ


def test_capture_event_decays_with_distance() -> None:
    # 角交換: ▲7六歩 △3四歩 ▲2二角成 (2二の角を取る)
    board, tracker, record = play(["7g7f", "3c3d", "8h2b+"])
    assert record.capture and record.captured_piece_id is not None
    features = build_event_features(board, tracker, record)

    # 取られた駒 (後手の角) 本人には was_captured が立ち、距離減衰は載らない
    captured_index = token_index(tracker, record.captured_piece_id)
    assert features[captured_index, WAS_CAPTURED] == 1.0
    assert features[captured_index, ALLY_LOST] == 0.0

    # 味方 (後手) は事件現場 2二 に近いほど強く動揺する
    capture_square = str_to_sq(record.to_square)
    states = sorted(tracker.states.values(), key=lambda item: item.piece_id)
    losses = {
        state.piece_id: features[i, ALLY_LOST]
        for i, state in enumerate(states)
        if state.owner == 1 and not state.in_hand
    }
    near = losses["L11_gen0_0001"]   # 1一の香 (現場の隣)
    far = losses["L91_gen0_0037"]    # 9一の香 (盤の反対側)
    assert near > far > 0.0

    # 取った側 (先手) には enemy_lost として届き、ally_lost とは混ざらない
    mover_side = [
        features[i, ENEMY_LOST]
        for i, state in enumerate(states)
        if state.owner == 0 and not state.in_hand and state.piece_id != record.piece_id
    ]
    assert max(mover_side) > 0.0
    assert all(
        features[i, ALLY_LOST] == 0.0 for i, state in enumerate(states) if state.owner == 0
    )


def test_hand_pieces_are_isolated_from_board_events() -> None:
    board, tracker, record = play(["7g7f", "3c3d", "8h2b+"])
    features = build_event_features(board, tracker, record)

    states = sorted(tracker.states.values(), key=lambda item: item.piece_id)
    for index, state in enumerate(states):
        if state.in_hand:
            # 持ち駒に届くのは was_captured だけ (盤上の脅威や距離減衰は0)
            others = [
                axis for axis in range(NUM_EVENT_FEATURES) if axis != WAS_CAPTURED
            ]
            assert features[index, others].sum() == 0.0


def test_check_event_reaches_only_the_checked_side() -> None:
    # ▲7六歩 △3四歩 ▲2二角成 △同銀 ▲3三角打 (3三→4二→5一 の筋で後手玉に王手)
    board, tracker, record = play(["7g7f", "3c3d", "8h2b+", "3a2b", "B*3c"])
    assert board.is_check()
    features = build_event_features(board, tracker, record)

    states = sorted(tracker.states.values(), key=lambda item: item.piece_id)
    for index, state in enumerate(states):
        if state.in_hand:
            continue
        expected = 1.0 if state.owner == 1 else 0.0  # 王手されているのは後手
        assert features[index, KING_IN_CHECK] == expected


def test_threatened_axis_is_normalized() -> None:
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    features = build_event_features(board, tracker, None)
    assert features[:, THREATENED].min() >= 0.0
    assert features[:, THREATENED].max() <= 1.0


def test_gru_updates_shape_and_reacts_to_events() -> None:
    gru = MoodGRU(SMALL)
    state = gru.initial_state(batch=2, tokens=MAX_PIECES)
    events = torch.zeros(2, MAX_PIECES, NUM_EVENT_FEATURES)
    events[0, 3, CAPTURED] = 1.0

    updated = gru(events, state)
    assert updated.shape == (2, MAX_PIECES, SMALL.d_mood)
    # イベントが起きた駒と起きていない駒で状態が分かれる
    assert not torch.allclose(updated[0, 3], updated[0, 4])
    # 同じ入力なら決定的
    assert torch.allclose(updated, gru(events, state))


def test_projection_ranges_match_interface() -> None:
    projection = MoodProjection(SMALL)
    mood = torch.randn(3, MAX_PIECES, SMALL.d_mood) * 5
    out = projection(mood)

    assert out.shape == (3, MAX_PIECES, 3)
    fear, aggression, valence = out[..., 0], out[..., 1], out[..., 2]
    assert fear.min() >= 0.0 and fear.max() <= 1.0
    assert aggression.min() >= 0.0 and aggression.max() <= 1.0
    assert valence.min() >= -1.0 and valence.max() <= 1.0


def test_flag_off_keeps_checkpoint_compatibility() -> None:
    """mood フラグOFFの state_dict に mood 関連キーが混ざらない (Phase 1-2 互換)。"""
    model = KokoroPolicy(SMALL, FeatureFlags(), head="desire")
    assert not any("mood" in key for key in model.state_dict())


def test_flag_on_zero_init_matches_flag_off_output() -> None:
    """mood ONでも W_m が零初期化なので、同じ重みならフラグOFFと同じ出力から始まる。"""
    torch.manual_seed(0)
    base = KokoroPolicy(SMALL, FeatureFlags(), head="desire")
    torch.manual_seed(0)
    with_mood = KokoroPolicy(SMALL, FeatureFlags(mood=True), head="desire")
    # フラグOFF側の重みを流し込む。personality.project は mood 分だけ入力次元が
    # 広がる (16→16+8) ので除外し、trunk の合流点だけを検証する
    state = {
        key: value
        for key, value in base.state_dict().items()
        if not key.startswith("personality.project")
    }
    missing, unexpected = with_mood.load_state_dict(state, strict=False)
    assert not unexpected and all("mood" in key or "personality" in key for key in missing)

    tokenizer = PieceTokenizer()
    board = cshogi.Board()
    tokens = tokenizer.tokenize(board)
    as_long = lambda a: torch.from_numpy(a.astype("int64"))[None]
    inputs = dict(
        species=as_long(tokens.species),
        position=as_long(tokens.position),
        owner=as_long(tokens.owner),
        promoted=as_long(tokens.promoted),
        mask=torch.from_numpy(tokens.mask)[None],
        turn=torch.tensor([tokens.turn]),
    )
    mood = torch.randn(1, MAX_PIECES, SMALL.d_mood)

    with torch.no_grad():
        plain = base(**inputs)
        moody = with_mood(**inputs, mood=mood)
    # trunk 側の合流 (W_m=0) は完全一致。personality 側は mood 行が乱数初期化なので
    # logits は変わり得るが、trunk 出力 (hidden) が一致していれば合流点は正しい
    assert torch.allclose(plain.hidden, moody.hidden, atol=1e-6)
