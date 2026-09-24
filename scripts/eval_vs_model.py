"""凍結したチェックポイント (アンカー) を相手に、学習済みモデルの相対的な強さを測る。

対 Háo (eval_vs_engine.py) は絶対基準だが、全モデルが勝率 0.02〜0.23 に潰れて感度が足りず、
`--engine-nodes` で相手を弱めることもできなかった (2026-09-20、nodes=1 でも 0.230)。
このスクリプトは**動かさないアンカー** (既定 phase37_e3.pt = 全系列の出発点) と対戦させ、
勝率 0.5 付近で差を検出する。絶対値は測れないので対 Háo と**併記**する。

    uv run python scripts/eval_vs_model.py --checkpoint checkpoints/stage1_fate.pt \\
        --games 100 --openings data/teacher/openings_fg.txt \\
        --out checkpoints/anchor/stage1_fate.json

両者とも play_server と同じ指し方 (argmax / 1 手詰チェック)。序盤ブックと先後は
eval_vs_engine と同じ配り方なので、同じ序盤集で測った結果どうしは対応あり検定できる。
出力の `model_side` / `result` は --checkpoint 側から見た値。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cshogi
import torch
from export_model_jsonl import build_action_map, sample_move
from play_server import DEFAULT_CHECKPOINT, load_runner
from usi_server import Position

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR = REPO_ROOT / "checkpoints" / "phase37_e3.pt"


def pick_move(pos: Position, runner, *, mate_check: bool, tau: float, generator, rounds):
    """play_server / eval_vs_engine と同じ指し方。合法手が無ければ None。"""
    board = pos.board
    if mate_check:
        mate = board.mate_move_in_1ply()
        if mate:
            return int(mate)
    output, legal = runner.forward(
        board, pos.tokens, pos.mood, pos.relation, pos.effect, rounds=rounds
    )
    if not legal.any():
        return None
    return sample_move(output, build_action_map(board, pos.tokens), tau, generator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--culture", default=None, help="--checkpoint 側の文化 (league.pt 用)")
    parser.add_argument("--anchor-culture", default=None)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--no-mate-check", action="store_true")
    parser.add_argument("--council-rounds", type=int, default=None, help="--checkpoint 側の R")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--openings", type=Path, default=None)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(2)  # 並走する学習を餓死させない
    runner_a, culture_a = load_runner(args.checkpoint, device, culture=args.culture)
    runner_b, culture_b = load_runner(args.anchor, device, culture=args.anchor_culture)
    generator = torch.Generator().manual_seed(args.seed)
    mate_check = not args.no_mate_check
    print(f"model: {args.checkpoint.name} / {culture_a}"
          f"  vs  anchor: {args.anchor.name} / {culture_b}"
          f"  (tau {args.tau} / mate1 {'on' if mate_check else 'off'})", flush=True)

    openings: list[list[str]] = []
    if args.openings is not None:
        openings = [
            line.split() for line in args.openings.read_text().splitlines() if line.strip()
        ]
    score = 0.0
    results = []
    started = time.perf_counter()
    for game in range(args.games):
        model_side = game % 2
        pos_a, pos_b = Position(runner_a), Position(runner_b)
        moves_usi: list[str] = []
        if openings:
            for usi in openings[(game // 2) % len(openings)]:
                move = pos_a.board.move_from_usi(usi)
                moves_usi.append(usi)
                pos_a.push(move)
                pos_b.push(move)
        opening_plies = len(moves_usi)
        winner = None
        reason = "max_plies"
        for _ply in range(args.max_plies):
            board = pos_a.board
            if board.is_game_over():
                winner = 1 - int(board.turn)
                reason = "mate"
                break
            a_to_move = int(board.turn) == model_side
            if a_to_move:
                move = pick_move(pos_a, runner_a, mate_check=mate_check, tau=args.tau,
                                 generator=generator, rounds=args.council_rounds)
            else:
                move = pick_move(pos_b, runner_b, mate_check=mate_check, tau=args.tau,
                                 generator=generator, rounds=None)
            if move is None:
                winner = (1 - model_side) if a_to_move else model_side
                reason = "no_legal"
                break
            moves_usi.append(cshogi.move_to_usi(move))
            pos_a.push(move)
            pos_b.push(move)
        result = 0.5 if winner is None else (1.0 if winner == model_side else 0.0)
        score += result
        results.append({"game": game, "model_side": model_side, "result": result,
                        "plies": len(moves_usi), "opening_plies": opening_plies,
                        "reason": reason})
        side = "先手" if model_side == 0 else "後手"
        mark = "勝" if result == 1 else "負" if result == 0 else "分"
        print(f"  game {game + 1}/{args.games}: {side} {mark} ({len(moves_usi)}手, {reason})"
              f" 累計 {score / (game + 1):.3f}", flush=True)
    summary = {"checkpoint": args.checkpoint.name, "culture": culture_a,
               "anchor": args.anchor.name, "anchor_culture": culture_b, "tau": args.tau,
               "mate_check": mate_check, "games": args.games, "win_rate": score / args.games,
               "council_rounds": args.council_rounds,
               "openings": str(args.openings) if args.openings else None,
               "distinct_plies": len({r["plies"] for r in results}),
               "seconds": round(time.perf_counter() - started), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"== 対アンカー勝率 {score / args.games:.3f} ({args.games} 局, 異なる手数 "
          f"{summary['distinct_plies']}, {summary['seconds']}s) → {args.out}", flush=True)


if __name__ == "__main__":
    main()
