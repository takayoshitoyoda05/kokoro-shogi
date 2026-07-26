"""棋譜を学習用シャードへ一括変換する (Phase 0 の出口)。

ダウンロード済みの CSA 棋譜を読み、1局面あたり次を npz シャードにまとめる:

- 駒トークン (species / position / owner / promoted / mask) … core/tokenizer.py
- 教師手を **方策ヘッドの形** に分解したもの (move_token / move_to / move_promote)
  DESIGN.md §3(3) の $u_{i,a} = [h_i \\| E_{pos}[dst(a)] \\| \\rho(a)]$ に対応する
- 勝敗 z (手番側視点) … value ヘッドの教師
- 欲求ラベル6軸 (uint8化) … 補助損失 $L_{desire}$ の教師

利き行列は **保存しない**。トークンから盤面を復元できるので学習時に計算する方が
ディスクを食わない (1局面あたり 40×40 バイト = 1.6KB になってしまう)。

Phase 0 の Gate はこのスクリプトのスループット計測 (1M局面変換ベンチ) を兼ねる。

使い方::

    uv run python scripts/make_labels.py --csa-dir data/floodgate/csa/2025-01-15
    uv run python scripts/make_labels.py --csa-dir data/floodgate/csa/2024 --limit-positions 1000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cshogi
import numpy as np

from kokoro_shogi.config import REPO_ROOT
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer
from kokoro_shogi.data.floodgate import (
    DEFAULT_MIN_RATING,
    GameRecord,
    iter_csa_files,
    load_games,
)
from kokoro_shogi.data.labels import NUM_AXES, compute_desire_labels

DEFAULT_CSA_DIR = REPO_ROOT / "data" / "floodgate" / "csa"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "shards"
DEFAULT_SHARD_SIZE = 100_000


class ShardBuilder:
    """局面を貯めて、一定数ごとに npz として書き出す。

    **シャードの切れ目は必ず局の境界にする** (`flush_if_full` を1局終わるごとに
    呼ぶ)。1局が2つのシャードに分かれると、Phase 3 の感情GRU [A] のように
    「1局を通した系列」を必要とする学習でシャードを跨いだ連結が要る。
    そのぶんシャードの局面数は `shard_size` ちょうどにはならない。
    """

    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._shard_index = 0
        self._buffers: dict[str, list[np.ndarray | int | float]] = {}
        self.total = 0

    def add(self, **columns: np.ndarray | int | float) -> None:
        for key, value in columns.items():
            self._buffers.setdefault(key, []).append(value)
        self.total += 1

    def flush_if_full(self) -> Path | None:
        """溜まっていれば書き出す。局の切れ目で呼ぶこと。"""
        if len(self._buffers.get("move_to", ())) >= self.shard_size:
            return self.flush()
        return None

    def flush(self) -> Path | None:
        if not self._buffers or not self._buffers.get("move_to"):
            return None

        path = self.out_dir / f"shard_{self._shard_index:04d}.npz"
        np.savez(path, **{key: np.asarray(value) for key, value in self._buffers.items()})
        self._buffers.clear()
        self._shard_index += 1
        return path


def encode_game(
    record: GameRecord, tokenizer: PieceTokenizer, builder: ShardBuilder, game_index: int
) -> int:
    """1局を局面ごとに分解してシャードへ積む。積んだ局面数を返す。"""
    game_labels = compute_desire_labels(list(record.moves))

    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    order = {piece_id: index for index, piece_id in enumerate(game_labels.piece_ids)}

    added = 0
    for ply, move in enumerate(record.moves):
        tokens = tokenizer.tokenize(board, tracker)

        if cshogi.move_is_drop(move):
            # 打ちは「持ち駒のどれを打つか」なので、動く駒トークンは駒台側にいる
            from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES

            species = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
            candidates = [
                state
                for state in tracker.pieces_in_hand(board.turn)
                if state.base_species == species
            ]
            moving_id = candidates[0].piece_id
        else:
            moving_id = tracker.piece_id_at(cshogi.move_from(move))  # type: ignore[assignment]

        builder.add(
            species=tokens.species,
            position=tokens.position,
            owner=tokens.owner,
            promoted=tokens.promoted,
            mask=tokens.mask,
            turn=np.int8(tokens.turn),
            move=np.int32(move),
            move_token=np.int8(order[moving_id]),
            move_to=np.int8(cshogi.move_to(move)),
            move_promote=np.bool_(cshogi.move_is_promotion(move)),
            result=np.int8(record.result_at(ply)),
            labels=np.rint(game_labels.labels[ply] * 255).astype(np.uint8),
            game_index=np.int32(game_index),
        )
        added += 1

        tracker.apply_move(board, move)
        board.push(move)

    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csa-dir", type=Path, default=DEFAULT_CSA_DIR, help="CSA棋譜のディレクトリ"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="シャードの出力先")
    parser.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="1シャードの局面数"
    )
    parser.add_argument(
        "--min-rating", type=float, default=DEFAULT_MIN_RATING, help="両者の最低レート"
    )
    parser.add_argument("--limit-games", type=int, default=None, help="使う棋譜数の上限")
    parser.add_argument("--limit-positions", type=int, default=None, help="作る局面数の上限")
    parser.add_argument(
        "--dry-run", action="store_true", help="書き出さずスループットだけ測る (ベンチ用)"
    )
    args = parser.parse_args()

    tokenizer = PieceTokenizer(MAX_PIECES)
    builder = ShardBuilder(args.out_dir, args.shard_size)

    games = load_games(
        iter_csa_files(args.csa_dir), min_rating=args.min_rating, limit=args.limit_games
    )

    started = time.perf_counter()
    game_count = 0
    for game_index, record in enumerate(games):
        encode_game(record, tokenizer, builder, game_index)
        game_count += 1
        if not args.dry_run:
            builder.flush_if_full()  # 局の境界でだけシャードを切る
        if args.limit_positions and builder.total >= args.limit_positions:
            break
        if game_count % 200 == 0:
            rate = builder.total / (time.perf_counter() - started)
            print(f"  {game_count} 局 / {builder.total} 局面 ({rate:.0f} 局面/秒)")

    if args.dry_run:
        builder._buffers.clear()  # noqa: SLF001 - ベンチ用に書き出しだけ捨てる
    else:
        builder.flush()

    elapsed = time.perf_counter() - started
    rate = builder.total / elapsed if elapsed else 0.0
    print(
        f"完了: {game_count} 局 / {builder.total} 局面 / {elapsed:.1f} 秒 "
        f"({rate:.0f} 局面/秒)"
    )
    if rate:
        print(f"1M局面あたりの見込み: {1_000_000 / rate / 60:.1f} 分")
    print(f"1局面あたり: トークン {MAX_PIECES * 4} B + 欲求ラベル {MAX_PIECES * NUM_AXES} B")


if __name__ == "__main__":
    main()
