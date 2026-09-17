"""人間 vs AI のライブ対局サーバー (INTERFACE.md §4)。

Unity (WebSocket クライアント) から `game_control` / `move_request` を受け、学習済み
モデルで応手して `state_update` / `legal_moves` / `career` を返す。状態機械は
`kokoro_shogi.server.game_session.GameSession`、通信は `kokoro_shogi.server.server`
(JSON を運ぶだけ) で、このスクリプトはその 2 つをモデルで繋ぐ。

- 推論は既定で CPU (1 手 1 秒未満)。GPU 実験と同居しても干渉しない
- 接続してきたクライアントごとに独立した `GameSession` を持つ (同時に複数人と指せる)
- 既定モデルは checkpoints/league_E2b_grace/league.pt (リポジトリ同梱、mood / relations /
  council 込み)。ppo2.pt などローカルのチェックポイントも `--checkpoint` で渡せる。
  `--checkpoint` に Phase 2 の方策 (desire_lambda*.pt) を渡すと mood 等はヒューリスティック

使い方::

    uv run python scripts/play_server.py                       # 人間が先手、port 8765
    uv run python scripts/play_server.py --human white --tau 0  # 人間が後手、AI は argmax
    uv run python scripts/play_server.py --selfcheck            # 接続せずに乱択で 1 局回して終了

Unity 側の手順: 接続 → `{"schema":"1.0","type":"game_control","command":"start"}` を送る →
届いた `legal_moves` から選んで `move_request` を送る → `state_update` ×2 + 次の `legal_moves`。
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path

import cshogi
import numpy as np
import torch
from export_model_jsonl import (
    ModelRunner,
    build_action_map,
    build_state_update,
    council_rounds_from_output,
    sample_move,
)

from kokoro_shogi.config import REPO_ROOT
from kokoro_shogi.core.effects import piece_effect_matrix
from kokoro_shogi.core.piece_state import MoveRecord, PieceIdTracker
from kokoro_shogi.core.pieces import BLACK, WHITE
from kokoro_shogi.core.tokenizer import MAX_PIECES
from kokoro_shogi.logging.jsonl import CouncilRound, StateUpdate, to_json_line
from kokoro_shogi.model.mood import build_event_features
from kokoro_shogi.model.relations import initial_relations, update_relations
from kokoro_shogi.server import server as ws
from kokoro_shogi.server.game_session import DEFAULT_MAX_PLIES, GameSession, Phase
from kokoro_shogi.viz.narrator import TemplateNarrator

#: リポジトリに同梱している共有モデル (README「学習済みモデル」)。clone 直後でも存在する
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "league_E2b_grace" / "league.pt"
#: 対局時の既定温度 (export_model_jsonl の 0.25 は多様なサンプル生成用)
DEFAULT_TAU = 0.1
#: 受信キューを覗く間隔 (秒)。server.py の receive_any は非ブロッキング
POLL_INTERVAL = 0.02


class ModelEngine:
    """`GameSession` に差す実モデルのエンジン (game_session.Engine プロトコル)。

    1 局のあいだ感情 GRU の状態 m、関係状態 R、直前の会議ログを持ち回る。
    時刻合わせは export_model_jsonl.generate_game と同じ:
    m^(t) = GRU(u_ev(直前の手), m^(t-1)) を、その手の後の局面で更新する。
    """

    def __init__(self, runner: ModelRunner, *, tau: float, seed: int) -> None:
        self.runner = runner
        self.tau = tau
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.narrator = TemplateNarrator()
        self.mood_state: torch.Tensor | None = None
        self.relation_state: torch.Tensor | None = None
        self.prev_council: list[CouncilRound] = []

    def start(self) -> None:
        self.mood_state = None
        self.relation_state = None
        self.prev_council = []

    def step(
        self,
        board: cshogi.Board,
        tracker: PieceIdTracker,
        ply: int,
        record: MoveRecord | None,
        *,
        ai_moved: bool,
    ) -> tuple[StateUpdate, int | None]:
        runner = self.runner
        if runner.gru is not None:
            if self.mood_state is None:
                self.mood_state = runner.gru.initial_state(1, MAX_PIECES, device=runner.device)
            events = build_event_features(board, tracker, record)
            with torch.no_grad():
                self.mood_state = runner.gru(
                    torch.from_numpy(events)[None].to(runner.device), self.mood_state
                )
        tokens = runner.tokenizer.tokenize(board, tracker)
        effect = piece_effect_matrix(board, tokens.squares()).astype(np.int64)
        if runner.relations:
            if self.relation_state is None:
                self.relation_state = initial_relations(1, MAX_PIECES, device=runner.device)
            self.relation_state = update_relations(
                self.relation_state, torch.from_numpy(effect)[None].to(runner.device)
            )
        output, legal = runner.forward(board, tokens, self.mood_state, self.relation_state, effect)

        # 会議ログと実況は AI が指した手のぶんだけ載せる (人間の手に議事録はない)
        council: list[CouncilRound] = []
        narration = ""
        if ai_moved and record is not None and self.prev_council:
            council = self.prev_council
            meta = {s.piece_id: (s.species, s.owner) for s in tracker.states.values()}
            narration = self.narrator.narrate(council, record.piece_id, meta)
        state = build_state_update(
            runner, board, tracker, ply, record, output, legal, tokens,
            self.mood_state, self.relation_state, council, narration,
        )

        move: int | None = None
        self.prev_council = []
        if legal.any() and not board.is_game_over():
            move = sample_move(output, build_action_map(board, tokens), self.tau, self.generator)
            states = sorted(tracker.states.values(), key=lambda item: item.piece_id)
            self.prev_council = council_rounds_from_output(output, states)
        return state, move


def load_runner(checkpoint: Path, device: torch.device) -> ModelRunner:
    """感情GRU入り (mood_gru を持つ) なら mood_checkpoint として、そうでなければ方策だけ読む。"""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "mood_gru" in state:
        return ModelRunner(checkpoint, device, mood_checkpoint=checkpoint)
    return ModelRunner(checkpoint, device)


def as_wire(message) -> dict:
    """Schema 1.0 のメッセージ → server.send_client に渡す dict。"""
    return json.loads(to_json_line(message))


def selfcheck(make_engine, human: int, max_plies: int) -> None:
    """接続せずに、人間側を乱択で代行して 1 局回す (モデルと配線の疎通確認)。"""
    session = GameSession(make_engine(), human=human, max_plies=max_plies)
    rng = random.Random(0)
    started = time.perf_counter()
    out = session.handle({"schema": "1.0", "type": "game_control", "command": "start"})
    steps = 0
    while session.phase is Phase.HUMAN_TURN:
        pick = rng.choice(session._last_legal.moves)
        move = {"from": pick.from_, "to": pick.to}
        if pick.drop_species:
            move["drop_species"] = pick.drop_species
        else:
            move["promote"] = pick.promote
        out = session.handle({"schema": "1.0", "type": "move_request", "move": move})
        steps += 1
        for message in out:
            if isinstance(message, StateUpdate) and message.narration:
                print(f"  ply {message.ply:3d} eval {message.eval:+.3f}  {message.narration}")
    elapsed = time.perf_counter() - started
    career = out[-1]
    print(
        f"selfcheck: {session.ply} 手で終局 (人間の手 {steps} 回, {elapsed:.1f}s,"
        f" {elapsed / max(session.ply, 1):.2f}s/手) / career {len(career.pieces)} 駒"
        f" / MVP {career.last_game_mvp.piece_id if career.last_game_mvp else '-'}"
    )


def serve(make_engine, *, host: str, port: int, human: int, max_plies: int) -> None:
    sessions: dict[str, GameSession] = {}
    running = True

    def stop(*_args) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ws.start_server(host, port)
    side = "先手" if human == BLACK else "後手"
    print(f"listening ws://{host}:{port}  (人間: {side}) Ctrl-C で終了")
    try:
        while running:
            busy = False
            while (event := ws.get_server_event()) is not None:
                busy = True
                kind, ip = event.get("event"), event.get("client_ip")
                print(f"[{event.get('time', '')}] {kind} {ip or ''}")
                if kind == "client_connected" and ip:
                    session = sessions.setdefault(
                        ip, GameSession(make_engine(), human=human, max_plies=max_plies)
                    )
                    # 起動時 (接続時) にも career を配る (INTERFACE.md §4)
                    ws.send_client(ip, as_wire(session.ledger.message()))
            while (received := ws.receive_any()) is not None:
                busy = True
                ip, data = received["client_ip"], received["data"]
                session = sessions.setdefault(
                    ip, GameSession(make_engine(), human=human, max_plies=max_plies)
                )
                if not isinstance(data, dict):
                    continue
                started = time.perf_counter()
                replies = session.handle(data)
                for message in replies:
                    ws.send_client(ip, as_wire(message))
                kinds = ",".join(type(m).__name__ for m in replies) or "-"
                elapsed = time.perf_counter() - started
                print(
                    f"{ip} {data.get('type')}({data.get('command', '')}) ply={session.ply}"
                    f" phase={session.phase.value} -> {kinds} ({elapsed:.2f}s)"
                )
            if not busy:
                time.sleep(POLL_INTERVAL)
    finally:
        ws.end_server()
        print("server stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="ppo2.pt / league.pt など感情GRU入り、または desire_lambda*.pt (Phase 2 方策)",
    )
    parser.add_argument("--human", choices=["black", "white"], default="black", help="人間の手番")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="AI の温度 (0 で argmax)")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--host", default=ws.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=ws.DEFAULT_PORT)
    parser.add_argument(
        "--device", default="cpu", help="推論デバイス (既定 cpu。GPU 実験と同居するため)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selfcheck", action="store_true", help="接続せず乱択相手に 1 局回して終了"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    runner = load_runner(args.checkpoint, device)
    print(
        f"checkpoint: {args.checkpoint.name} / device: {device} / tau: {args.tau}"
        f" / mood: {'感情GRU' if runner.gru is not None else 'ヒューリスティック'}"
        f" / relations: {'r_ij状態' if runner.relations else 'ヒューリスティック'}"
        f" / council: {'ON' if runner.council else 'OFF'}"
    )
    human = BLACK if args.human == "black" else WHITE

    def make_engine() -> ModelEngine:
        return ModelEngine(runner, tau=args.tau, seed=args.seed)

    if args.selfcheck:
        selfcheck(make_engine, human, args.max_plies)
        return
    serve(make_engine, host=args.host, port=args.port, human=human, max_plies=args.max_plies)


if __name__ == "__main__":
    main()
