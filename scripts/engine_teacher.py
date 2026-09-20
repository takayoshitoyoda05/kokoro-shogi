"""外部エンジン (やねうら王 + NNUE) で局面に教師ラベルを付ける。

2026-09-19 の決定「学習・評価時の外部エンジン利用は可」に基づく。

木を作らないのは**推論時**の話で、学習の教師にはエンジンの探索結果を使う (Ruoss et al. 2024 流)。
現行のやねうら王には gensfen (学習部) が無いので、USI を直接叩いて MultiPV と評価値を集める。

入力: 1 行 1 SFEN のテキスト。出力: 1 行 1 JSON::

    {"sfen": "...", "depth": 8, "best": "7g7f",
     "pv": [{"move": "7g7f", "cp": 45, "mate": null}, ...],   # MultiPV 順 (1 位が先頭)
     "nodes": 12345, "time_ms": 30}

**評価値は USI の慣例どおり手番側から見た値** (先手固定ではない)。cp と mate はどちらか一方。

使い方::

    uv run python scripts/engine_teacher.py --random 20 --out /tmp/x.jsonl          # 動作確認
    uv run python scripts/engine_teacher.py --positions sfens.txt --depth 8 --multipv 4 \\
        --workers 16 --out data/teacher/hao_d8.jsonl                                  # 本番

`--out` が既にあれば済みの SFEN は飛ばす (途中再開できる)。ワーカーごとにエンジンを 1 プロセス
(Threads=1) 持つので、`--workers` は CPU スレッド数まで。GPU は使わないので学習と並走できる。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import subprocess
import time
from pathlib import Path

from kokoro_shogi.config import REPO_ROOT

DEFAULT_ENGINE = REPO_ROOT / "data" / "engines" / "bin" / "YaneuraOu-by-gcc"
DEFAULT_EVAL_DIR = REPO_ROOT / "data" / "engines" / "eval_hao" / "eval"
#: Háo の配布ページが推奨する値 (やねうら王の既定 16 ではない)
FV_SCALE = 20


class Engine:
    """USI エンジン 1 プロセス。`analyse` で 1 局面ぶんの MultiPV を返す。"""

    def __init__(self, path: Path, eval_dir: Path, *, multipv: int, hash_mb: int) -> None:
        self.proc = subprocess.Popen(
            [str(path)], cwd=path.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.send("usi")
        self.wait_for("usiok")
        for name, value in (
            ("EvalDir", str(eval_dir)), ("FV_SCALE", FV_SCALE), ("Threads", 1),
            ("USI_Hash", hash_mb), ("MultiPV", multipv), ("BookFile", "no_book"),
            ("NetworkDelay", 0), ("NetworkDelay2", 0), ("MinimumThinkingTime", 1),
        ):
            self.send(f"setoption name {name} value {value}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait_for(self, token: str) -> list[str]:
        assert self.proc.stdout is not None
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(token):
                return lines

    def analyse(self, sfen: str, *, depth: int | None, nodes: int | None) -> dict:
        self.send(f"position sfen {sfen}")
        go = f"go depth {depth}" if depth else f"go nodes {nodes}"
        started = time.perf_counter()
        self.send(go)
        lines = self.wait_for("bestmove")
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        best = lines[-1].split()[1]
        by_rank: dict[int, dict] = {}
        last_nodes = 0
        for line in lines:
            if not line.startswith("info ") or " pv " not in line:
                continue
            tokens = line.split()
            fields: dict[str, str] = {}
            for i, tok in enumerate(tokens):
                if tok in ("depth", "seldepth", "multipv", "nodes", "time", "nps", "hashfull"):
                    fields[tok] = tokens[i + 1]
                elif tok == "score":
                    fields["score_kind"], fields["score"] = tokens[i + 1], tokens[i + 2]
            pv = tokens[tokens.index("pv") + 1 :]
            rank = int(fields.get("multipv", 1))
            by_rank[rank] = {
                "move": pv[0] if pv else None,
                "cp": int(fields["score"]) if fields.get("score_kind") == "cp" else None,
                "mate": int(fields["score"]) if fields.get("score_kind") == "mate" else None,
                "depth": int(fields.get("depth", 0)),
            }
            last_nodes = int(fields.get("nodes", last_nodes))
        return {
            "sfen": sfen,
            "depth": depth,
            "nodes_limit": nodes,
            "best": best,
            "pv": [by_rank[r] for r in sorted(by_rank)],
            "nodes": last_nodes,
            "time_ms": elapsed_ms,
        }

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


_ENGINE: Engine | None = None


def _init(path: str, eval_dir: str, multipv: int, hash_mb: int) -> None:
    global _ENGINE
    _ENGINE = Engine(Path(path), Path(eval_dir), multipv=multipv, hash_mb=hash_mb)


def _work(job: tuple[str, int | None, int | None]) -> dict:
    assert _ENGINE is not None
    sfen, depth, nodes = job
    return _ENGINE.analyse(sfen, depth=depth, nodes=nodes)


def random_positions(count: int, *, min_plies: int, max_plies: int, seed: int) -> list[str]:
    """平手から乱択で進めた局面 (動作確認用)。終局した局は捨てる。"""
    import cshogi

    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < count:
        board = cshogi.Board()
        for _ in range(rng.randint(min_plies, max_plies)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            out.append(board.sfen())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--positions", type=Path, help="1 行 1 SFEN")
    parser.add_argument(
        "--random", type=int, default=0, help="乱択局面を N 個作って使う (動作確認)"
    )
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument(
        "--nodes", type=int, default=None, help="指定すると depth の代わりにノード数で切る"
    )
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.random:
        sfens = random_positions(args.random, min_plies=8, max_plies=40, seed=args.seed)
    elif args.positions:
        sfens = [s.strip() for s in args.positions.read_text().splitlines() if s.strip()]
    else:
        parser.error("--positions か --random が要る")

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sfen"])
    todo = [s for s in sfens if s not in done]
    print(f"局面 {len(sfens)} (済 {len(done)}, 残り {len(todo)}) / depth {args.depth}"
          f" / multipv {args.multipv} / workers {args.workers}", flush=True)
    if not todo:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    depth = None if args.nodes else args.depth
    jobs = [(s, depth, args.nodes) for s in todo]
    started = time.perf_counter()
    written = 0
    with args.out.open("a", encoding="utf-8") as sink, mp.Pool(
        args.workers, initializer=_init,
        initargs=(str(args.engine), str(args.eval_dir), args.multipv, args.hash_mb),
    ) as pool:
        for record in pool.imap_unordered(_work, jobs, chunksize=4):
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0 or written == len(jobs):
                rate = written / (time.perf_counter() - started)
                print(f"  {written}/{len(jobs)}  {rate:.1f} 局面/秒", flush=True)
    print(f"完了: {written} 局面 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
