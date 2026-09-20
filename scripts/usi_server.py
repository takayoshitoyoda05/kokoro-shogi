"""USI (Universal Shogi Interface) エンジンアダプタ。

将棋所 / ShogiGUI / ShogiHome などの GUI から、学習済みモデルを 1 つの USI エンジンとして
使えるようにする。stdin から 1 行ずつコマンドを読み、stdout に応答を書くだけの薄い層で、
モデル側の部品は `scripts/play_server.py` (人間 vs AI の WebSocket 対局) と同じものを使う。

- 探索はしない。`go` を受けたら forward を 1 回して `bestmove` を即答する
  (`ponder` / `stop` / `go mate` / `multipv` は未対応。`stop` は無視して問題ない)
- USI の `position` は毎回**初手からの全手順**を送ってくる。このモデルは駒の同一性
  (`PieceIdTracker`)・感情 m・関係 R が履歴に依存するので、局面は初手から再生する。
  直前の手順の前方一致なら差分だけ進める (prefix キャッシュ)
- 感情・関係の時刻合わせは `play_server.ModelEngine.step` と同じ:
  「指した後の局面」で、その手のイベントで m を更新し、その局面の利きで R を更新してから forward
- `info string` に会議の議事録と実況を流す (GUI の思考ログに駒たちの議論が出る)
- `score cp` は価値ヘッド V∈[-1,1] を勝率に直して cp = -600·ln(1/w − 1) で写した参考値

使い方 (GUI にはこのコマンドをエンジンとして登録する)::

    uv run python scripts/usi_server.py                       # 既定: league_E2b_grace, culture1
    uv run python scripts/usi_server.py --culture culture3 --tau 0
    printf 'usi\\nisready\\nposition startpos moves 7g7f\\ngo byoyomi 1000\\nquit\\n' \\
        | uv run python scripts/usi_server.py

GUI の `setoption name Culture value cultureN` / `Tau` / `MateCheck` は `isready` の時点で反映する。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cshogi
import numpy as np
import torch
from export_model_jsonl import (
    ModelRunner,
    build_action_map,
    council_rounds_from_output,
    sample_move,
)
from play_server import DEFAULT_CHECKPOINT, DEFAULT_TAU, apply_culture

from kokoro_shogi.core.effects import piece_effect_matrix
from kokoro_shogi.core.piece_state import MoveRecord, PieceIdTracker
from kokoro_shogi.core.tokenizer import MAX_PIECES
from kokoro_shogi.model.mood import build_event_features
from kokoro_shogi.model.relations import initial_relations, update_relations
from kokoro_shogi.viz.narrator import SPECIES_JA, TemplateNarrator

ENGINE_NAME = "Kokoro-Shogi"
ENGINE_AUTHOR = "Taka"
#: 既定の文化。E2b の保存時の culture4 (0.617) より culture1 (0.692) が強い (README の表)
DEFAULT_CULTURE = "culture1"
#: 勝率 → cp の慣用スケール (エンジンごとに違う参考値。GUI の評価値グラフ用)
CP_SCALE = 600.0
CP_LIMIT = 30000


def send(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def value_to_cp(value: float) -> int:
    win = min(max((value + 1.0) / 2.0, 1e-6), 1.0 - 1e-6)
    return int(max(-CP_LIMIT, min(CP_LIMIT, -CP_SCALE * math.log(1.0 / win - 1.0))))


class Position:
    """`position` コマンドの再生器。tracker / 感情 / 関係 / 利きを手順に沿って保つ。"""

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self.sfen: str | None = None
        self.moves: list[int] = []
        self.reset(None)

    def reset(self, sfen: str | None) -> None:
        self.board = cshogi.Board()
        if sfen:
            self.board.set_sfen(sfen)
        # 持ち駒のある局面からは駒の同一性を復元できない (piece_state.PieceIdTracker)。
        # その場合は tracker なし = 感情・関係・会議ログを持たない「記憶なし」で指す
        try:
            self.tracker: PieceIdTracker | None = PieceIdTracker(self.board)
        except ValueError:
            self.tracker = None
            send("info string 持ち駒のある局面から開始: 駒の記憶 (感情・関係) なしで指します")
        runner = self.runner
        self.mood = (
            runner.gru.initial_state(1, MAX_PIECES, device=runner.device)
            if runner.gru is not None else None
        )
        self.relation = (
            initial_relations(1, MAX_PIECES, device=runner.device) if runner.relations else None
        )
        self.sfen = sfen
        self.moves = []
        self._observe(None)

    def set(self, sfen: str | None, moves: list[int]) -> None:
        if sfen != self.sfen or moves[: len(self.moves)] != self.moves:
            self.reset(sfen)
        for move in moves[len(self.moves):]:
            self.push(move)

    def push(self, move: int) -> None:
        record = self.tracker.apply_move(self.board, move) if self.tracker else None
        self.board.push(move)
        self.moves.append(move)
        self._observe(record)

    def _observe(self, record: MoveRecord | None) -> None:
        """play_server.ModelEngine.step と同じ順で m, R を進め、tokens / effect を用意する。"""
        runner = self.runner
        if runner.gru is not None and self.tracker is not None:
            events = build_event_features(self.board, self.tracker, record)
            with torch.no_grad():
                self.mood = runner.gru(
                    torch.from_numpy(events)[None].to(runner.device), self.mood
                )
        self.tokens = runner.tokenizer.tokenize(self.board, self.tracker)
        self.effect = piece_effect_matrix(self.board, self.tokens.squares()).astype(np.int64)
        if runner.relations and self.tracker is not None:
            self.relation = update_relations(
                self.relation, torch.from_numpy(self.effect)[None].to(runner.device)
            )


class Engine:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.cultures: dict = self.state.get("cultures", {})
        self.options = {
            "Culture": DEFAULT_CULTURE if DEFAULT_CULTURE in self.cultures else None,
            "Tau": DEFAULT_TAU,
            "MateCheck": True,
            "Seed": 0,
        }
        self.runner: ModelRunner | None = None
        self.position: Position | None = None
        self.narrator = TemplateNarrator()

    # --- ハンドシェイク -------------------------------------------------------

    def usi(self) -> None:
        send(f"id name {ENGINE_NAME}")
        send(f"id author {ENGINE_AUTHOR}")
        if self.cultures:
            current = self.options["Culture"] or next(iter(self.cultures))
            variants = " ".join(f"var {name}" for name in self.cultures)
            send(f"option name Culture type combo default {current} {variants}")
        send(f"option name Tau type string default {self.options['Tau']}")
        send("option name MateCheck type check default true")
        send(f"option name Seed type spin default {self.options['Seed']} min 0 max 1000000")
        send("usiok")

    def setoption(self, args: list[str]) -> None:
        # setoption name <id> [value <x>]
        if "name" not in args:
            return
        name = args[args.index("name") + 1]
        value = args[args.index("value") + 1] if "value" in args else None
        if name == "Culture" and value in self.cultures:
            self.options["Culture"] = value
        elif name == "Tau" and value is not None:
            self.options["Tau"] = float(value)
        elif name == "MateCheck" and value is not None:
            self.options["MateCheck"] = value.lower() == "true"
        elif name == "Seed" and value is not None:
            self.options["Seed"] = int(value)

    def isready(self) -> None:
        if self.runner is None:
            self.runner = (
                ModelRunner(self.checkpoint, self.device, mood_checkpoint=self.checkpoint)
                if "mood_gru" in self.state else ModelRunner(self.checkpoint, self.device)
            )
        label = apply_culture(self.runner, self.cultures, self.options["Culture"])
        self.generator = torch.Generator().manual_seed(int(self.options["Seed"]))
        self.position = Position(self.runner)
        send(
            f"info string {self.checkpoint.name} / culture: {label} / tau: {self.options['Tau']}"
            f" / mood: {'GRU' if self.runner.gru is not None else 'heuristic'}"
            f" / relations: {'on' if self.runner.relations else 'off'}"
            f" / council: {'on' if self.runner.council else 'off'}"
        )
        send("readyok")

    # --- 局面と着手 -------------------------------------------------------------

    def set_position(self, args: list[str]) -> None:
        assert self.position is not None, "isready より前に position が来た"
        sfen: str | None = None
        moves: list[str] = []
        if args and args[0] == "startpos":
            rest = args[1:]
        elif args and args[0] == "sfen":
            rest = args[1:]
            fields = []
            while rest and rest[0] != "moves":
                fields.append(rest.pop(0))
            sfen = " ".join(fields)
        else:
            rest = args
        if rest and rest[0] == "moves":
            moves = rest[1:]
        # USI 文字列 → cshogi の手。初手から順に読むため一時盤で変換する
        board = cshogi.Board()
        if sfen:
            board.set_sfen(sfen)
        parsed: list[int] = []
        for usi in moves:
            move = board.move_from_usi(usi)
            board.push(move)
            parsed.append(move)
        self.position.set(sfen, parsed)

    def go(self) -> None:
        assert self.position is not None and self.runner is not None
        pos, runner = self.position, self.runner
        board = pos.board
        if board.is_game_over():
            send("bestmove resign")
            return

        if self.options["MateCheck"]:
            mate = board.mate_move_in_1ply()
            if mate:
                send(f"info depth 1 score mate 1 pv {cshogi.move_to_usi(mate)}")
                send("info string 詰みを発見したので議論を打ち切った")
                send(f"bestmove {cshogi.move_to_usi(mate)}")
                return

        output, legal = runner.forward(board, pos.tokens, pos.mood, pos.relation, pos.effect)
        if not legal.any():
            send("bestmove resign")
            return
        action_map = build_action_map(board, pos.tokens)
        move = sample_move(output, action_map, float(self.options["Tau"]), self.generator)
        usi = cshogi.move_to_usi(move)

        council = []
        if pos.tracker is not None:
            states = sorted(pos.tracker.states.values(), key=lambda item: item.piece_id)
            council = council_rounds_from_output(output, states)
        if council:
            meta = {s.piece_id: (s.species, s.owner) for s in states}
            inverse = {v: k for k, v in action_map.items()}
            chosen = states[inverse[move][0]].piece_id if move in inverse else None
            last = council[-1].proposals[:3]
            claims = " / ".join(
                f"{'▲' if meta[p.piece_id][1] == 0 else '△'}"
                f"{SPECIES_JA.get(meta[p.piece_id][0], '?')}"
                f"「{p.move}」bid={p.bid:.2f}"
                for p in last
            )
            send(f"info string 会議R{len(council)}: {claims}")
            narration = self.narrator.narrate(council, chosen, meta)
            if narration:
                send(f"info string {narration}")
        send(f"info depth 1 nodes 1 score cp {value_to_cp(float(output.value[0]))} pv {usi}")
        send(f"bestmove {usi}")

    # --- メインループ -----------------------------------------------------------

    def run(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            command, *args = line.split()
            try:
                self.dispatch(command, args)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 — 何が来てもプロトコルは続ける
                send(f"info string error: {type(error).__name__}: {error}")
                if command == "go":
                    send("bestmove resign")
            if command == "quit":
                break

    def dispatch(self, command: str, args: list[str]) -> None:
            if command == "usi":
                self.usi()
            elif command == "isready":
                self.isready()
            elif command == "setoption":
                self.setoption(args)
            elif command == "usinewgame":
                if self.position is not None:
                    self.position.reset(None)
                self.generator = torch.Generator().manual_seed(int(self.options["Seed"]))
            elif command == "position":
                self.set_position(args)
            elif command == "go":
                self.go()
            # quit はループ側で処理。stop / ponderhit / gameover / debug は即答型なので無視してよい


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--culture", default=None, help=f"既定 {DEFAULT_CULTURE} (league.pt のとき)"
    )
    parser.add_argument("--tau", type=float, default=None, help=f"既定 {DEFAULT_TAU}。0 で argmax")
    parser.add_argument("--no-mate-check", action="store_true", help="1手詰チェックを切る")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    engine = Engine(args.checkpoint, torch.device(args.device))
    if args.culture is not None:
        engine.options["Culture"] = args.culture
    if args.tau is not None:
        engine.options["Tau"] = args.tau
    if args.no_mate_check:
        engine.options["MateCheck"] = False
    engine.run()


if __name__ == "__main__":
    main()
