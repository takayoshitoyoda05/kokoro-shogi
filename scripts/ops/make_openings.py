"""floodgate 棋譜から評価用の序盤ブック (1 行 = USI の手順) を作る。

対局評価の決定論対策。argmax の方策 × 決定論的エンジンでは同じ棋譜が繰り返され、
100 局回しても実質 2 局しか測れていなかった (2026-09-19 の第1弾)。局ごとに違う序盤から始めて
標本を増やす。手順で持つのは PieceIdTracker / 感情 / 関係が履歴依存のため
(SFEN では初期化できない)。

    uv run python scripts/ops/make_openings.py --count 500 --min-plies 10 --max-plies 24 \\
        --out data/teacher/openings_fg.txt
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
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--min-plies", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    lines: list[str] = []
    for record in load_games(iter_csa_files(args.csa_root)):
        if rng.random() > 0.05:  # 棋譜順の偏り (同じ日の同じ対局者) を薄める
            continue
        plies = rng.randint(args.min_plies, args.max_plies)
        if len(record.moves) <= plies + 20:
            continue
        board = cshogi.Board()
        usi = []
        for move in record.moves[:plies]:
            usi.append(cshogi.move_to_usi(move))
            board.push(move)
        if board.is_game_over():
            continue
        line = " ".join(usi)
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= args.count:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} 手順 → {args.out}")


if __name__ == "__main__":
    main()
