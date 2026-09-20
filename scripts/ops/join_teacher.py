"""エンジン教師ラベル (engine_teacher.py の JSONL) を既存の npz シャードに結合する。

シャードの各局面をトークンから盤面へ戻し、SFEN (盤面+手番+持ち駒) をキーに教師を引く。
教師のある局面だけ値が入り、無い局面は NaN / -1 で埋める。出力はシャードと同名の
`*.teacher.npz` (シャード本体は変更しない)。

    uv run python scripts/ops/join_teacher.py --teacher data/teacher/hao_d10_fg.jsonl \\
        --shards data/shards/full/2024 --limit-shards 2

列:
- teacher_value  (N,) float32   手番側から見た評価値 → [-1,1]。cp は 2σ(cp/600)−1、mate は ±1
- teacher_cp     (N,) float32   生の cp (mate は ±30000)。診断用
- teacher_actions (N, K) int64  MultiPV の手 (action index、無い所は -1)
- teacher_cps    (N, K) float32 各手の cp (soft target 用、無い所は NaN)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cshogi
import numpy as np

from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES, SPECIES_ORDER, base_species
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.core.tokenizer import MAX_PIECES
from kokoro_shogi.data.dataset import ShardDataset, action_index, tokens_to_board

#: 勝率変換のスケール (cp)。慣用値で、Háo の FV_SCALE とは別物
CP_SCALE = 600.0
MATE_CP = 30000.0


def cp_to_value(cp: float | None, mate: int | None) -> tuple[float, float]:
    """(value ∈ [-1,1], 生 cp)。mate は手数に関係なく ±1。"""
    if mate is not None:
        sign = 1.0 if mate > 0 else -1.0
        return sign, sign * MATE_CP
    assert cp is not None
    return 2.0 / (1.0 + math.exp(-cp / CP_SCALE)) - 1.0, float(cp)


def usi_to_action(
    board: cshogi.Board, usi: str, species: np.ndarray, position: np.ndarray,
    owner: np.ndarray, mask: np.ndarray,
) -> int:
    """USI の手 → action index。

    打ちは piece_id 最小の持ち駒トークン (dataset.legal_move_mask と同じ規則)。
    """
    move = board.move_from_usi(usi)
    to_square = cshogi.move_to(move)
    promote = int(cshogi.move_is_promotion(move))
    turn = int(board.turn)
    if cshogi.move_is_drop(move):
        held = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
        for index in range(MAX_PIECES):
            in_hand = mask[index] and int(position[index]) >= NUM_SQUARES
            if in_hand and int(owner[index]) == turn:
                if base_species(SPECIES_ORDER[int(species[index])]) == held:
                    return action_index(index, to_square, promote)
        return -1
    source = cshogi.move_from(move)
    for index in range(MAX_PIECES):
        if mask[index] and int(position[index]) == source:
            return action_index(index, to_square, promote)
    return -1


def load_teacher(path: Path) -> dict[str, dict]:
    table: dict[str, dict] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            table[" ".join(record["sfen"].split(" ")[:3])] = record
    return table


def join_shard(shard: Path, teacher: dict[str, dict], k: int) -> tuple[int, int]:
    dataset = ShardDataset([shard], with_legal=False, with_effect=False)
    n = len(dataset)
    value = np.full(n, np.nan, dtype=np.float32)
    cp_raw = np.full(n, np.nan, dtype=np.float32)
    actions = np.full((n, k), -1, dtype=np.int64)
    cps = np.full((n, k), np.nan, dtype=np.float32)
    hits = 0
    for i in range(n):
        p = dataset[i]
        board = tokens_to_board(p.species, p.position, p.owner, p.promoted, p.mask, p.turn)
        record = teacher.get(" ".join(board.sfen().split(" ")[:3]))
        if record is None or not record["pv"]:
            continue
        hits += 1
        top = record["pv"][0]
        value[i], cp_raw[i] = cp_to_value(top["cp"], top["mate"])
        for j, entry in enumerate(record["pv"][:k]):
            if entry["move"] is None:
                continue
            action = usi_to_action(board, entry["move"], p.species, p.position, p.owner, p.mask)
            if action < 0:
                continue
            actions[i, j] = action
            _, cps[i, j] = cp_to_value(entry["cp"], entry["mate"])
    out = shard.with_suffix(".teacher.npz")
    np.savez(
        out, teacher_value=value, teacher_cp=cp_raw, teacher_actions=actions, teacher_cps=cps
    )
    return n, hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument(
        "--shards", type=Path, nargs="+", required=True, help="シャードのディレクトリ"
    )
    parser.add_argument("--k", type=int, default=4, help="MultiPV の上位何手を残すか")
    parser.add_argument("--limit-shards", type=int, default=None)
    args = parser.parse_args()

    teacher = load_teacher(args.teacher)
    print(f"教師 {len(teacher)} 局面", flush=True)
    paths = sorted(
        p for d in args.shards for p in Path(d).glob("shard_*.npz") if ".teacher" not in p.name
    )
    if args.limit_shards:
        paths = paths[: args.limit_shards]
    total = total_hits = 0
    for path in paths:
        n, hits = join_shard(path, teacher, args.k)
        total += n
        total_hits += hits
        print(f"  {path.parent.name}/{path.name}: {hits}/{n} ({hits / n:.1%})", flush=True)
    print(f"完了: {total_hits}/{total} 局面に教師 ({total_hits / max(total, 1):.1%})", flush=True)


if __name__ == "__main__":
    main()
