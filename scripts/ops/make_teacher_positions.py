"""floodgate 棋譜から、エンジン教師ラベル用の局面リスト (1 行 1 SFEN) を作る。

蒸留と同じ採用条件 (両者レート 3000+、決着局、50〜400 手、平手開始) の棋譜を再生し、
各局からランダムな位相で `stride` 手ごとに局面を拾う。盤面+手番+持ち駒で重複を除く。

    uv run python scripts/ops/make_teacher_positions.py --stride 10 --limit 2000000 \\
        --out data/teacher/positions_fg.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cshogi

from kokoro_shogi.config import REPO_ROOT
from kokoro_shogi.data.floodgate import iter_csa_files, load_games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csa-root", type=Path, default=REPO_ROOT / "data" / "floodgate" / "csa")
    parser.add_argument("--stride", type=int, default=10, help="1 局から何手ごとに拾うか")
    parser.add_argument("--min-ply", type=int, default=4, help="序盤の定跡的な数手は飛ばす")
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    written = games = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for record in load_games(iter_csa_files(args.csa_root)):
            games += 1
            board = cshogi.Board()
            offset = rng.randrange(args.stride)
            for ply, move in enumerate(record.moves):
                due = ply >= args.min_ply and (ply - offset) % args.stride == 0
                if due and not board.is_game_over():
                    sfen = board.sfen()
                    key = " ".join(sfen.split(" ")[:3])  # 手数を除いて重複判定
                    if key not in seen:
                        seen.add(key)
                        sink.write(sfen + "\n")
                        written += 1
                        if written >= args.limit:
                            break
                board.push(move)
            if written >= args.limit:
                break
            if games % 10000 == 0:
                print(f"  {games} 局 / {written} 局面", flush=True)
    print(f"完了: {games} 局から {written} 局面 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
