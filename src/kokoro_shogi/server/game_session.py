"""人間 vs AI のライブ対局セッション (INTERFACE.md §4 の状態機械、AI 側)。

WebSocket 層 (server.py) は JSON を運ぶだけなので、ここが「受けた move_request を
盤に適用し、モデルで応手し、state_update / legal_moves / career を返す」本体。
通信には依存せず、`GameSession.handle(dict) -> list[メッセージ]` の純粋な関数として
書いてある (テストは偽エンジンで回せる。実モデルは scripts/play_server.py が差す)。

状態機械 (INTERFACE.md §4。これ以外の遷移は実装しない)::

    [待機] --start--> [人間手番] --move_request--> [AI思考中]
                         ↑                              |
                         └──── state_update 送信 ───────┘
    任意の状態で resign / reset → [待機]

- 合法手の判定は AI 側の責務。人間手番の開始時に legal_moves を配る
- 不正な move_request は **エラーにせず legal_moves を再送** する
- 終局 (詰み・千日手・手数上限・投了) では career を配って [待機] に戻る

2026-09-06 実装 (週5 「server v1」の積み残し)。参照: docs/INTERFACE.md §2, §4, §5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import cshogi

from kokoro_shogi.core.piece_state import MoveRecord, PieceIdTracker, PieceState
from kokoro_shogi.core.pieces import BLACK, HAND_INDEX_TO_SPECIES, WHITE
from kokoro_shogi.core.squares import sq_to_str
from kokoro_shogi.logging.jsonl import (
    CareerMessage,
    CareerMvp,
    CareerPiece,
    GameControl,
    LegalMove,
    LegalMovesMessage,
    MoveRequest,
    StateUpdate,
    _Message,
    validate_obj,
)

#: 手数上限。ここまで決着しなければ引き分け扱いで終局する
DEFAULT_MAX_PLIES = 320


class Engine(Protocol):
    """1局面を評価して state_update と (手番側の) 指し手を返すもの。

    `step` は **指した後の局面** `board` に対して呼ばれる (初期局面は record=None)。
    戻り値の指し手は `board.turn` 側のもので、AI 手番のときだけセッションが採用する。
    合法手が無ければ None。`ai_moved` は直前の手を AI が指したかどうか
    (会議ログ・実況を state_update に載せるのは AI の手だけ)。
    """

    def start(self) -> None: ...

    def step(
        self,
        board: cshogi.Board,
        tracker: PieceIdTracker,
        ply: int,
        record: MoveRecord | None,
        *,
        ai_moved: bool,
    ) -> tuple[StateUpdate, int | None]: ...


class Phase(StrEnum):
    IDLE = "idle"
    HUMAN_TURN = "human_turn"
    AI_THINKING = "ai_thinking"


# --- 合法手 ⇄ ワイヤ形式 -------------------------------------------------------


def legal_move_of(move: int) -> LegalMove:
    """cshogi の move → INTERFACE.md §4 の legal_moves 要素。"""
    to_square = sq_to_str(cshogi.move_to(move))
    if cshogi.move_is_drop(move):
        return LegalMove(
            **{"from": "00"},
            to=to_square,
            drop_species=HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)],
        )
    return LegalMove(
        **{"from": sq_to_str(cshogi.move_from(move))},
        to=to_square,
        promote=bool(cshogi.move_is_promotion(move)),
    )


def legal_moves_message(board: cshogi.Board) -> LegalMovesMessage:
    """手番側の合法手一覧。"""
    return LegalMovesMessage(moves=[legal_move_of(move) for move in board.legal_moves])


def find_move(board: cshogi.Board, requested: LegalMove) -> int | None:
    """move_request の手を合法手から探す。無ければ None (打ちは成りを見ない)。"""
    for move in board.legal_moves:
        candidate = legal_move_of(move)
        if candidate.from_ != requested.from_ or candidate.to != requested.to:
            continue
        if candidate.drop_species is not None:
            if candidate.drop_species == requested.drop_species:
                return move
        elif candidate.promote == requested.promote and requested.drop_species is None:
            return move
    return None


# --- career (セッション内集計) --------------------------------------------------


@dataclass
class CareerLedger:
    """このプロセスで指した対局のキャリア集計 (INTERFACE.md §5)。

    永続化 (persist/store.py) とは独立にセッション内で数える。終局時に配る
    career は「今日この相手と指した分」で、Unity のキャリア UI の確認には足りる。
    MVP は Phase 5 の功績配分が本来の出所なので、ここでは成り+生存の簡易指標。
    """

    games: dict[str, int] = field(default_factory=dict)
    survivals: dict[str, int] = field(default_factory=dict)
    promotions: dict[str, int] = field(default_factory=dict)
    species: dict[str, str] = field(default_factory=dict)
    mvp_count: dict[str, int] = field(default_factory=dict)
    last_mvp: CareerMvp | None = None

    def record(self, states: dict[str, PieceState], promoted: dict[str, int]) -> None:
        best_id, best_score = None, -1.0
        for state in states.values():
            pid = state.piece_id
            self.games[pid] = self.games.get(pid, 0) + 1
            self.species[pid] = state.species
            if not state.in_hand:  # 終局時に盤上に残っていれば生存
                self.survivals[pid] = self.survivals.get(pid, 0) + 1
            self.promotions[pid] = self.promotions.get(pid, 0) + promoted.get(pid, 0)
            score = promoted.get(pid, 0) + (0.0 if state.in_hand else 1.0)
            if score > best_score:
                best_id, best_score = pid, score
        if best_id is not None:
            self.mvp_count[best_id] = self.mvp_count.get(best_id, 0) + 1
            self.last_mvp = CareerMvp(piece_id=best_id, contribution=round(best_score, 3))

    def message(self) -> CareerMessage:
        pieces = [
            CareerPiece(
                piece_id=pid,
                species=self.species[pid],
                games=count,
                survival_rate=round(self.survivals.get(pid, 0) / count, 3),
                promotions=self.promotions.get(pid, 0),
                mvp_count=self.mvp_count.get(pid, 0),
            )
            for pid, count in sorted(self.games.items())
        ]
        return CareerMessage(pieces=pieces, last_game_mvp=self.last_mvp)


# --- セッション ----------------------------------------------------------------


class GameSession:
    """1 クライアントぶんの対局状態機械。

    `human` は人間の手番 (BLACK=先手 / WHITE=後手)。後手なら start 直後に AI が
    初手を指してから legal_moves を配る。`initial_sfen` はテスト用 (持ち駒なし限定)。
    """

    def __init__(
        self,
        engine: Engine,
        *,
        human: int = BLACK,
        max_plies: int = DEFAULT_MAX_PLIES,
        initial_sfen: str | None = None,
    ) -> None:
        if human not in (BLACK, WHITE):
            raise ValueError(f"human は BLACK(0) か WHITE(1): {human}")
        self.engine = engine
        self.human = human
        self.max_plies = max_plies
        self.initial_sfen = initial_sfen
        self.ledger = CareerLedger()
        self.phase = Phase.IDLE
        self.board = cshogi.Board()
        self.tracker: PieceIdTracker | None = None
        self.ply = 0
        self._promoted: dict[str, int] = {}
        #: 直近に配った legal_moves (再送用)
        self._last_legal: LegalMovesMessage | None = None

    @property
    def ai(self) -> int:
        return WHITE if self.human == BLACK else BLACK

    # --- 入口 -------------------------------------------------------------------

    def handle(self, data: dict) -> list[_Message]:
        """受信 1 件を処理して、送り返すメッセージ列を返す。

        壊れた JSON や未知の type は無視する (空リスト)。AI→Unity 専用の種別
        (state_update / legal_moves / career) が来ても同様に無視。
        """
        try:
            message = validate_obj(data)
        except ValueError:
            return []
        if isinstance(message, GameControl):
            return self._on_control(message)
        if isinstance(message, MoveRequest):
            return self._on_move_request(message)
        return []

    # --- game_control -----------------------------------------------------------

    def _on_control(self, message: GameControl) -> list[_Message]:
        if message.command == "start":
            return self._start()
        if message.command == "resign":
            out: list[_Message] = []
            if self.phase is not Phase.IDLE and self.tracker is not None:
                self.ledger.record(self.tracker.states, self._promoted)
                out.append(self.ledger.message())
            self._to_idle()
            return out
        self._to_idle()  # reset
        return []

    def _to_idle(self) -> None:
        self.phase = Phase.IDLE
        self._last_legal = None

    def _start(self) -> list[_Message]:
        self.board = cshogi.Board(self.initial_sfen) if self.initial_sfen else cshogi.Board()
        self.tracker = PieceIdTracker(self.board)
        self.ply = 0
        self._promoted = {}
        self.engine.start()
        out: list[_Message] = []
        state, move = self.engine.step(self.board, self.tracker, 0, None, ai_moved=False)
        out.append(state)
        if self._finished(move):
            return out + self._finish()
        if self.board.turn == self.ai:
            out.extend(self._ai_reply(move))
        else:
            out.append(self._human_turn())
        return out

    # --- move_request -----------------------------------------------------------

    def _on_move_request(self, message: MoveRequest) -> list[_Message]:
        if self.phase is not Phase.HUMAN_TURN or self.tracker is None:
            return []  # 待機中・AI思考中の手は捨てる (状態機械外の遷移は実装しない)
        move = find_move(self.board, message.move)
        if move is None:
            # 不正な手はエラー扱いにせず legal_moves を再送する (INTERFACE.md §4)
            return [self._last_legal or legal_moves_message(self.board)]
        self.phase = Phase.AI_THINKING
        out: list[_Message] = []
        state, reply = self._apply(move, ai_moved=False)
        out.append(state)
        if self._finished(reply):
            return out + self._finish()
        out.extend(self._ai_reply(reply))
        return out

    # --- 内部 -------------------------------------------------------------------

    def _apply(self, move: int, *, ai_moved: bool) -> tuple[StateUpdate, int | None]:
        """1手を進めてエンジンに次局面を見せる。"""
        assert self.tracker is not None
        record = self.tracker.apply_move(self.board, move)
        if record.promote:
            self._promoted[record.piece_id] = self._promoted.get(record.piece_id, 0) + 1
        self.board.push(move)
        self.ply += 1
        return self.engine.step(self.board, self.tracker, self.ply, record, ai_moved=ai_moved)

    def _ai_reply(self, move: int | None) -> list[_Message]:
        """AI が `move` を指し、その局面の state_update と人間の legal_moves を返す。"""
        assert move is not None
        self.phase = Phase.AI_THINKING
        state, next_move = self._apply(move, ai_moved=True)
        out: list[_Message] = [state]
        if self._finished(next_move):
            return out + self._finish()
        out.append(self._human_turn())
        return out

    def _human_turn(self) -> LegalMovesMessage:
        self.phase = Phase.HUMAN_TURN
        self._last_legal = legal_moves_message(self.board)
        return self._last_legal

    def _finished(self, move: int | None) -> bool:
        """終局判定。`move` はエンジンが返した手 (None なら合法手なし)。"""
        return (
            move is None
            or self.board.is_game_over()
            or self.ply >= self.max_plies
            or self.board.is_draw() == cshogi.REPETITION_DRAW
        )

    def _finish(self) -> list[_Message]:
        assert self.tracker is not None
        self.ledger.record(self.tracker.states, self._promoted)
        self._to_idle()
        return [self.ledger.message()]


__all__ = [
    "DEFAULT_MAX_PLIES",
    "CareerLedger",
    "Engine",
    "GameSession",
    "Phase",
    "find_move",
    "legal_move_of",
    "legal_moves_message",
]
