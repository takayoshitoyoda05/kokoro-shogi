"""|g| 上位手が捨て駒になっているかの定性確認 (DESIGN.md §4a / Phase 2 ADR用)。

DESIGN.md は「|自由項 g| が捨て駒検出器になる」と予想している。λ_g掃引で学習した
モデルを読み、検証局面ごとに **合法手の中で |g| が最大の手** を取り出して、
それが「損な交換」かどうかを数える::

    uv run python scripts/analyze_free_term.py \\
        --checkpoint checkpoints/desire_lambda0.1.pt --positions 2000

「損な交換」の判定 (簡易プロキシ): その手を指したあと相手が移動先のマスを
合法手で取り返せて、かつ動かした駒の価値 > 取った駒の価値 (取っていなければ0)。
歩の突き捨て・駒のタダ捨て・高い駒で安い駒を取る手がここに入る。

比較のベースラインとして、同じ局面から**一様ランダムに選んだ合法手**で同じ率を
測る。|g| 上位手の率がランダムより明確に高ければ「|g| は捨て駒に反応している」。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cshogi
import numpy as np
import torch

from kokoro_shogi.config import load_config
from kokoro_shogi.core.pieces import SPECIES_ORDER, base_species
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.data.dataset import (
    NUM_PROMOTE,
    ShardDataset,
    collate,
    decode_action,
    find_shards,
    tokens_to_board,
)
from kokoro_shogi.model.policy import KokoroPolicy

#: 駒の価値 (cshogi の piece type % 16 → 概算値。定性判定用なので粗くてよい)
PIECE_VALUES = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 99,
                9: 7, 10: 6, 11: 6, 12: 6, 13: 10, 14: 12}


def piece_value(code: int) -> int:
    return PIECE_VALUES.get(code % 16, 0) if code else 0


def find_cshogi_move(board: cshogi.Board, position: np.ndarray, species: np.ndarray,
                     owner: np.ndarray, mask: np.ndarray,
                     token: int, to_square: int, promote: int) -> int | None:
    """(token, to, promote) に対応する cshogi の合法手を探す (legal_move_mask と同じ規則)。"""
    from_square = int(position[token])
    for move in board.legal_moves:
        if cshogi.move_to(move) != to_square:
            continue
        if int(cshogi.move_is_promotion(move)) != promote:
            continue
        if cshogi.move_is_drop(move):
            if from_square < NUM_SQUARES:
                continue
            held = base_species(SPECIES_ORDER[int(species[token])])
            from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES
            if HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)] == held:
                return move
        elif cshogi.move_from(move) == from_square:
            return move
    return None


def is_losing_exchange(board: cshogi.Board, move: int) -> bool:
    """指したあと相手に取り返されて、駒の価値で損をする手か。"""
    to_square = cshogi.move_to(move)
    captured = piece_value(board.piece(to_square))
    if cshogi.move_is_drop(move):
        mover = PIECE_VALUES.get(cshogi.move_drop_hand_piece(move) + 1, 1)
    else:
        mover = piece_value(board.piece(cshogi.move_from(move)))

    board.push(move)
    try:
        recapture = any(
            cshogi.move_to(reply) == to_square and bool(board.piece(to_square))
            for reply in board.legal_moves
        )
    finally:
        board.pop()
    return recapture and mover > captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/desire_lambda0.1.pt"))
    parser.add_argument("--shard-dir", type=Path, default=Path("data/shards/2024"))
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config()
    model = KokoroPolicy.from_config(config, head="desire").to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"checkpoint: {args.checkpoint} (lambda_g={state.get('lambda_g')})")

    # 検証は先頭シャードから取る (distill.split_shards と同じ向き)
    shards = find_shards(args.shard_dir)[:1]
    dataset = ShardDataset(shards, max_positions=args.positions)
    rng = random.Random(config.seed)

    top_g_losing = 0
    random_losing = 0
    total = 0
    examples: list[str] = []
    #: 教師手の |g|: 捨て駒 (損な交換) だった手と、そうでない手で分けて集める
    teacher_g_losing: list[float] = []
    teacher_g_normal: list[float] = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4
    )
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model.forward_batch(moved)
        # (B, N, 162) を合法手だけ残して argmax
        g_abs = output.free_term.abs()
        legal = moved["legal"].flatten(start_dim=2).view_as(g_abs)
        g_abs = torch.where(legal, g_abs, torch.zeros_like(g_abs))
        flat_index = g_abs.flatten(start_dim=1).argmax(dim=1).cpu().numpy()
        g_max = g_abs.flatten(start_dim=1).max(dim=1).values.cpu().numpy()

        for row in range(len(flat_index)):
            if g_max[row] <= 0:
                continue
            species = batch["species"][row].numpy()
            position = batch["position"][row].numpy()
            owner = batch["owner"][row].numpy()
            promoted = batch["promoted"][row].numpy()
            piece_mask = batch["mask"][row].numpy()
            turn = int(batch["turn"][row])

            board = tokens_to_board(species, position, owner, promoted, piece_mask, turn)
            token, to_square, promote = decode_action(int(flat_index[row]))
            move = find_cshogi_move(board, position, species, owner, piece_mask,
                                    token, to_square, promote)
            if move is None:
                continue

            legal_list = list(board.legal_moves)
            random_move = rng.choice(legal_list)

            total += 1
            if is_losing_exchange(board, move):
                top_g_losing += 1
                if len(examples) < args.examples:
                    examples.append(
                        f"{cshogi.move_to_usi(move)}  |g|={g_max[row]:.2f}  {board.sfen()}"
                    )
            if is_losing_exchange(board, random_move):
                random_losing += 1

            # 教師手 (実際に指された手) の |g|。DESIGN.md の主張は
            # 「指された捨て駒は欲求で説明できず g が背負う」なので、こちらが本命
            t_token, t_to, t_promote = decode_action(int(batch["action"][row]))
            teacher_move = find_cshogi_move(board, position, species, owner, piece_mask,
                                            t_token, t_to, t_promote)
            if teacher_move is not None:
                g_teacher = float(
                    output.free_term[row, t_token, t_to * NUM_PROMOTE + t_promote].abs()
                )
                if is_losing_exchange(board, teacher_move):
                    teacher_g_losing.append(g_teacher)
                else:
                    teacher_g_normal.append(g_teacher)

    print(f"\n局面数: {total}")
    print(f"|g|最大の手が損な交換:   {top_g_losing / total * 100:6.2f}%  ({top_g_losing})")
    print(f"ランダム合法手が損な交換: {random_losing / total * 100:6.2f}%  ({random_losing})")
    if teacher_g_losing and teacher_g_normal:
        mean_losing = float(np.mean(teacher_g_losing))
        mean_normal = float(np.mean(teacher_g_normal))
        print(f"教師手の平均|g|  捨て駒 ({len(teacher_g_losing)}件): {mean_losing:.4f}"
              f" / 通常手 ({len(teacher_g_normal)}件): {mean_normal:.4f}"
              f"  比 {mean_losing / max(mean_normal, 1e-9):.2f}倍")
    print("\n例 (|g|最大かつ損な交換と判定された手):")
    for line in examples:
        print("  " + line)


if __name__ == "__main__":
    main()
