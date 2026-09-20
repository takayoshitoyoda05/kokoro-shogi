"""固定した外部エンジン (やねうら王 + Háo、ノード/深さ制限) を物差しにして、
学習済みモデルの絶対的な強さを測る。

2026-09-19 まで強さの数字は全て自己相対 (対 ppo2 勝率・はしご Elo) だった。このスクリプトは
**時間とともに動かない基準**を 1 つ立てる。本モデルは探索なし (1 手読み) なので、相手は
`--engine-depth` / `--engine-nodes` で強さを絞る。同じ設定で測り続けることに意味がある。

    uv run python scripts/eval_vs_engine.py --games 40 --engine-depth 1 \\
        --out checkpoints/vs_engine_d1.json
    uv run python scripts/eval_vs_engine.py --games 100 --engine-nodes 2000 \\
        --culture culture1 --tau 0 --out checkpoints/vs_engine_n2000.json

モデル側は usi_server.Position で感情・関係・会議を持ち回り、指し手は play_server と同じ
(既定 culture1 / argmax / 1 手詰チェック)。先後は交互。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cshogi
import torch
from engine_teacher import DEFAULT_ENGINE, DEFAULT_EVAL_DIR, FV_SCALE
from export_model_jsonl import build_action_map, sample_move
from play_server import DEFAULT_CHECKPOINT, DEFAULT_CULTURE, load_runner
from usi_server import Position


class UsiPlayer:
    """外部エンジンを 1 プロセス持ち、手順を渡して bestmove をもらう。"""

    def __init__(self, path: Path, eval_dir: Path, *, depth: int | None, nodes: int | None,
                 threads: int, hash_mb: int) -> None:
        self.depth, self.nodes = depth, nodes
        self.proc = subprocess.Popen(
            [str(path)], cwd=path.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("usi")
        self._wait("usiok")
        for name, value in (("EvalDir", str(eval_dir)), ("FV_SCALE", FV_SCALE),
                            ("Threads", threads), ("USI_Hash", hash_mb), ("BookFile", "no_book"),
                            ("NetworkDelay", 0), ("NetworkDelay2", 0), ("MinimumThinkingTime", 1)):
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait("readyok")

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str) -> str:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(token):
                return line.strip()

    def new_game(self) -> None:
        self._send("usinewgame")

    def move(self, moves_usi: list[str]) -> str:
        self._send("position startpos" + (" moves " + " ".join(moves_usi) if moves_usi else ""))
        self._send(f"go depth {self.depth}" if self.depth else f"go nodes {self.nodes}")
        return self._wait("bestmove").split()[1]

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--culture", default=None,
        help=f"league.pt の文化 (省略時 {DEFAULT_CULTURE}。文化の無いモデルはそのまま)",
    )
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--no-mate-check", action="store_true")
    parser.add_argument(
        "--council-rounds", type=int, default=None,
        help="会議のラウンド数 R を上書きする (重み共有なので推論時に自由に変えられる)。"
        " 省略時はモデル既定の R=2",
    )
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument(
        "--openings", type=Path, default=None,
        help="序盤ブック (1 行 = USI 手順)。局 i は手順 i//2 から (先後で同じ序盤を 1 回ずつ)。"
        " 無いと毎局平手初期局面からで、argmax × 決定論エンジンでは同じ棋譜が繰り返される",
    )
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--engine-depth", type=int, default=None)
    parser.add_argument("--engine-nodes", type=int, default=None)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.engine_depth or args.engine_nodes):
        parser.error("--engine-depth か --engine-nodes が要る")

    device = torch.device(args.device)
    if device.type == "cpu":
        # 1 手ごとの小さな forward に全コアを使うと、並走する学習 (自己対戦の env 側) を餓死させる
        torch.set_num_threads(2)
    runner, culture = load_runner(args.checkpoint, device, culture=args.culture)
    engine = UsiPlayer(
        args.engine, args.eval_dir, depth=args.engine_depth, nodes=args.engine_nodes,
        threads=args.engine_threads, hash_mb=64,
    )
    generator = torch.Generator().manual_seed(args.seed)
    label = f"depth {args.engine_depth}" if args.engine_depth else f"nodes {args.engine_nodes}"
    mate_label = "off" if args.no_mate_check else "on"
    print(f"model: {args.checkpoint.name} / {culture} / tau {args.tau} / mate1 {mate_label}"
          f"  vs  engine: {args.engine.name}+Háo {label}", flush=True)

    openings: list[list[str]] = []
    if args.openings is not None:
        openings = [
            line.split() for line in args.openings.read_text().splitlines() if line.strip()
        ]
    score = 0.0
    results = []
    started = time.perf_counter()
    for game in range(args.games):
        model_side = game % 2  # 偶数局は先手
        pos = Position(runner)
        engine.new_game()
        moves_usi: list[str] = []
        if openings:
            for usi in openings[(game // 2) % len(openings)]:
                move = pos.board.move_from_usi(usi)
                moves_usi.append(usi)
                pos.push(move)
        opening_plies = len(moves_usi)
        winner = None
        reason = "max_plies"
        for _ply in range(args.max_plies):
            board = pos.board
            if board.is_game_over():
                winner = 1 - int(board.turn)
                reason = "mate"
                break
            if int(board.turn) == model_side:
                mate = 0 if args.no_mate_check else board.mate_move_in_1ply()
                if mate:
                    move = int(mate)
                else:
                    output, legal = runner.forward(
                        board, pos.tokens, pos.mood, pos.relation, pos.effect,
                        rounds=args.council_rounds,
                    )
                    if not legal.any():
                        winner = 1 - model_side
                        reason = "no_legal"
                        break
                    move = sample_move(
                        output, build_action_map(board, pos.tokens), args.tau, generator
                    )
            else:
                usi = engine.move(moves_usi)
                if usi in ("resign", "win"):
                    winner = model_side if usi == "resign" else 1 - model_side
                    reason = usi
                    break
                move = board.move_from_usi(usi)
            moves_usi.append(cshogi.move_to_usi(move))
            pos.push(move)
        if winner is None:
            result = 0.5
        else:
            result = 1.0 if winner == model_side else 0.0
        score += result
        results.append({"game": game, "model_side": model_side, "result": result,
                        "plies": len(moves_usi), "opening_plies": opening_plies,
                        "reason": reason})
        side = "先手" if model_side == 0 else "後手"
        mark = "勝" if result == 1 else "負" if result == 0 else "分"
        print(f"  game {game + 1}/{args.games}: {side} {mark} ({len(moves_usi)}手, {reason})"
              f" 累計 {score / (game + 1):.3f}", flush=True)
    engine.close()
    summary = {"checkpoint": args.checkpoint.name, "culture": culture, "tau": args.tau,
               "mate_check": not args.no_mate_check, "engine": args.engine.name, "eval": "Hao",
               "engine_limit": label, "games": args.games, "win_rate": score / args.games,
               "council_rounds": args.council_rounds,
               "openings": str(args.openings) if args.openings else None,
               "seconds": round(time.perf_counter() - started), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"== 勝率 {score / args.games:.3f} ({args.games} 局, {summary['seconds']}s) → {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
