"""人間 vs AI セッション (server/game_session.py) の状態機械を偽エンジンで検証する。

INTERFACE.md §4 の遷移と「不正な move_request には legal_moves を再送」を確かめる。
モデルは使わず、偽エンジンは合法手をランダムに選び中身の薄い state_update を返す。
"""

from __future__ import annotations

import random

import cshogi
import pytest

from kokoro_shogi.core.pieces import BLACK, WHITE
from kokoro_shogi.core.squares import sq_to_str
from kokoro_shogi.logging.jsonl import (
    CareerMessage,
    LegalMove,
    LegalMovesMessage,
    StateUpdate,
    to_json_line,
    validate_line,
)
from kokoro_shogi.server.game_session import (
    GameSession,
    Phase,
    find_move,
    legal_move_of,
    legal_moves_message,
)


class RandomEngine:
    """合法手を乱択し、盤面だけ入った state_update を返す偽エンジン。"""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.started = 0
        self.ai_moved_flags: list[bool] = []

    def start(self) -> None:
        self.started += 1

    def step(self, board, tracker, ply, record, *, ai_moved):
        self.ai_moved_flags.append(ai_moved)
        moves = list(board.legal_moves)
        move = self.rng.choice(moves) if moves else None
        last = None
        if record is not None:
            last = {
                "from": record.from_square, "to": record.to_square,
                "piece_id": record.piece_id, "capture": record.capture,
                "promote": record.promote, "drop": record.drop,
            }
        state = StateUpdate(ply=ply, sfen=board.sfen(), last_move=last)
        return state, move


def start(session: GameSession):
    return session.handle({"schema": "1.0", "type": "game_control", "command": "start"})


def request(session: GameSession, move: dict):
    return session.handle({"schema": "1.0", "type": "move_request", "move": move})


def test_start_sends_initial_state_and_legal_moves_for_black_human():
    session = GameSession(RandomEngine())
    out = start(session)
    assert [type(m) for m in out] == [StateUpdate, LegalMovesMessage]
    assert out[0].ply == 0 and out[0].last_move is None
    assert len(out[1].moves) == 30  # 平手初期局面の先手合法手
    assert session.phase is Phase.HUMAN_TURN


def test_white_human_gets_ai_opening_first():
    session = GameSession(RandomEngine(), human=WHITE)
    out = start(session)
    assert [type(m) for m in out] == [StateUpdate, StateUpdate, LegalMovesMessage]
    assert out[1].ply == 1 and out[1].last_move is not None
    assert session.board.turn == WHITE


def test_valid_move_yields_human_state_ai_state_and_legal_moves():
    engine = RandomEngine()
    session = GameSession(engine)
    start(session)
    out = request(session, {"from": "77", "to": "76", "promote": False})
    assert [type(m) for m in out] == [StateUpdate, StateUpdate, LegalMovesMessage]
    assert out[0].ply == 1 and out[0].last_move.from_ == "77" and out[0].last_move.to == "76"
    assert out[1].ply == 2
    # 会議ログの主は AI の手だけ: 人間の手のあと False、AI の手のあと True
    assert engine.ai_moved_flags == [False, False, True]
    assert session.phase is Phase.HUMAN_TURN
    # 送るものは全部 Schema 1.0 として往復できる
    for message in out:
        validate_line(to_json_line(message))


def test_illegal_move_resends_legal_moves_without_advancing():
    session = GameSession(RandomEngine())
    start(session)
    out = request(session, {"from": "77", "to": "75", "promote": False})  # 歩は2マス進めない
    assert len(out) == 1 and isinstance(out[0], LegalMovesMessage)
    assert session.ply == 0 and session.phase is Phase.HUMAN_TURN


def test_move_request_outside_human_turn_is_ignored():
    session = GameSession(RandomEngine())
    assert request(session, {"from": "77", "to": "76", "promote": False}) == []
    assert session.phase is Phase.IDLE


def test_unknown_or_broken_messages_are_ignored():
    session = GameSession(RandomEngine())
    assert session.handle({"type": "nonsense"}) == []
    assert session.handle({"schema": "1.0", "type": "move_request"}) == []
    assert session.handle({"schema": "1.0", "type": "state_update", "ply": 0, "sfen": "x"}) == []


def test_resign_reports_career_and_returns_to_idle():
    session = GameSession(RandomEngine())
    start(session)
    out = session.handle({"schema": "1.0", "type": "game_control", "command": "resign"})
    assert len(out) == 1 and isinstance(out[0], CareerMessage)
    assert len(out[0].pieces) == 40 and all(p.games == 1 for p in out[0].pieces)
    # 人間 (先手) が投了したので AI (後手) の勝ち
    assert out[0].result.winner == "white"
    assert out[0].result.reason == "resign"
    assert session.phase is Phase.IDLE
    # 待機中の resign / reset は何も返さない
    assert session.handle({"schema": "1.0", "type": "game_control", "command": "resign"}) == []
    assert session.handle({"schema": "1.0", "type": "game_control", "command": "reset"}) == []


def test_reset_discards_game_without_career():
    session = GameSession(RandomEngine())
    start(session)
    assert session.handle({"schema": "1.0", "type": "game_control", "command": "reset"}) == []
    assert session.phase is Phase.IDLE and session.ledger.games == {}


def test_checkmate_by_human_ends_game_with_career():
    # 後手玉 5一、先手 金 6三・飛 5九・玉 1九。6三金→5二 で詰み (飛が金を支える)
    session = GameSession(RandomEngine(), initial_sfen="4k4/9/3G5/9/9/9/9/9/4R3K b - 1")
    start(session)
    assert session.phase is Phase.HUMAN_TURN
    out = request(session, {"from": "63", "to": "52", "promote": False})
    assert [type(m) for m in out] == [StateUpdate, CareerMessage]
    assert out[0].ply == 1 and session.board.is_game_over()
    assert session.phase is Phase.IDLE
    # 盤上の4駒 (k, G, R, K) が全員生存で career に載る
    assert len(out[1].pieces) == 4 and all(p.survival_rate == 1.0 for p in out[1].pieces)
    assert out[1].result.winner == "black"
    assert out[1].result.human == BLACK
    assert out[1].result.reason == "checkmate"
    # ワイヤ形式を往復しても result が保たれる
    assert validate_line(to_json_line(out[1])).result == out[1].result


def test_max_plies_ends_game_as_draw():
    session = GameSession(RandomEngine(seed=1), max_plies=4)
    start(session)
    legal = session._last_legal.moves
    first = legal[0]
    out = request(
        session,
        {"from": first.from_, "to": first.to, "promote": first.promote,
         **({"drop_species": first.drop_species} if first.drop_species else {})},
    )
    # ply 2 で AI が指し、次の人間手番へ
    assert session.phase is Phase.HUMAN_TURN
    second = session._last_legal.moves[0]
    out = request(
        session,
        {"from": second.from_, "to": second.to, "promote": second.promote,
         **({"drop_species": second.drop_species} if second.drop_species else {})},
    )
    assert isinstance(out[-1], CareerMessage)
    assert session.ply == 4 and session.phase is Phase.IDLE
    assert out[-1].result.winner == "draw"
    assert out[-1].result.reason == "max_plies"


def test_legal_move_wire_format_matches_interface_examples():
    board = cshogi.Board()
    message = legal_moves_message(board)
    encoded = to_json_line(message)
    assert '"from":"77","to":"76","promote":false' in encoded
    assert "drop_species" not in encoded  # 初期局面に打つ手はない

    # 打つ手: from "00" + drop_species、promote は落ちる
    board = cshogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
    drops = [legal_move_of(m) for m in board.legal_moves if cshogi.move_is_drop(m)]
    assert drops and drops[0].from_ == "00" and drops[0].drop_species == "FU"
    line = to_json_line(LegalMovesMessage(moves=drops[:1]))
    assert '"from":"00"' in line and '"drop_species":"FU"' in line and "promote" not in line


def test_find_move_distinguishes_promotion_and_drop():
    board = cshogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
    drop_wire = legal_move_of(next(m for m in board.legal_moves if cshogi.move_is_drop(m)))
    drop = find_move(board, drop_wire)
    assert drop is not None and cshogi.move_is_drop(drop)
    # 打つ手の from "00" を通常の手として問い合わせても一致しない
    assert find_move(board, LegalMove(**{"from": "00"}, to=drop_wire.to)) is None

    # 成れる局面: 先手歩 2三 → 2二 は成/不成の2通りで別の手
    board = cshogi.Board("4k4/9/7P1/9/9/9/9/9/4K4 b - 1")
    plain = find_move(board, LegalMove(**{"from": "23"}, to="22", promote=False))
    promoted = find_move(board, LegalMove(**{"from": "23"}, to="22", promote=True))
    assert plain is not None and promoted is not None and plain != promoted
    assert cshogi.move_is_promotion(promoted) and not cshogi.move_is_promotion(plain)
    assert sq_to_str(cshogi.move_to(promoted)) == "22"


@pytest.mark.parametrize("human", [BLACK, WHITE])
def test_full_random_game_terminates(human):
    """乱択同士で 1 局が必ず終わる (状態機械が詰まらない)。"""
    session = GameSession(RandomEngine(seed=7), human=human, max_plies=200)
    start(session)
    rng = random.Random(3)
    steps = 0
    while session.phase is Phase.HUMAN_TURN and steps < 300:
        pick = rng.choice(session._last_legal.moves)
        payload = {"from": pick.from_, "to": pick.to}
        if pick.drop_species:
            payload["drop_species"] = pick.drop_species
        else:
            payload["promote"] = pick.promote
        out = request(session, payload)
        assert out
        steps += 1
    assert session.phase is Phase.IDLE
    assert session.ledger.message().pieces


def test_move_request_from_unity_jsonutility_with_empty_drop_species():
    """Unity の JsonUtility は null の string を "" にして送る。通常の手として受理する。"""
    session = GameSession(RandomEngine())
    start(session)
    out = request(session, {"from": "77", "to": "76", "promote": False, "drop_species": ""})
    assert [type(m) for m in out] == [StateUpdate, StateUpdate, LegalMovesMessage]
    assert out[0].last_move.from_ == "77" and out[0].last_move.to == "76"


@pytest.mark.parametrize("human", [BLACK, WHITE])
def test_ai_checkmate_reports_winner_after_final_state(human):
    """AI が詰ませた場合、人間が先手でも後手でも勝者の表示が逆転しない。

    `winner` は盤の先後 (black/white) で、人間がどちら側かは `human` で別に伝わる。
    学習済みモデルを使わず、詰ませる手を固定する。
    """

    class MatingEngine(RandomEngine):
        def step(self, board, tracker, ply, record, *, ai_moved):
            state, move = super().step(
                board, tracker, ply, record, ai_moved=ai_moved,
            )
            if ply == 0:
                move = board.move_from_usi("6g5h" if human == BLACK else "6c5b")
                assert board.is_legal(move)
            return state, move

    sfen = (
        "4r3k/9/9/9/9/9/3g5/9/4K4 w - 1" if human == BLACK
        else "4k4/9/3G5/9/9/9/9/9/4R3K b - 1"
    )
    session = GameSession(MatingEngine(), human=human, initial_sfen=sfen)
    out = start(session)
    assert [type(m) for m in out] == [StateUpdate, StateUpdate, CareerMessage]
    assert session.board.is_game_over()
    assert session.phase is Phase.IDLE
    result = validate_line(to_json_line(out[-1])).result
    assert result.winner == ("white" if human == BLACK else "black")
    assert result.human == human
    assert result.reason == "checkmate"
    # 終局後の着手は受け付けない
    assert request(session, {"from": "59", "to": "49"}) == []


def test_check_with_escape_does_not_report_result():
    """王手でも逃げ道があれば終局にしない (結果画面を誤って出さない)。"""
    session = GameSession(
        RandomEngine(), human=WHITE,
        initial_sfen="4k4/9/9/9/9/9/9/9/4R3K w - 1",
    )
    out = start(session)
    assert session.board.is_check() and not session.board.is_game_over()
    assert [type(m) for m in out] == [StateUpdate, LegalMovesMessage]
    assert out[-1].moves and session.phase is Phase.HUMAN_TURN


def test_career_without_result_omits_the_key():
    """起動時の成績配信など、終局でない career には result キー自体が出ない。

    `result: null` が出ると Unity 側が終局と誤認しかねないため
    (INTERFACE.md §5、`CareerMessage._omit_absent_result`)。
    """
    session = GameSession(RandomEngine())
    start(session)
    plain = session.ledger.message()
    assert plain.result is None
    encoded = to_json_line(plain)
    assert '"result"' not in encoded
    assert validate_line(encoded).result is None
