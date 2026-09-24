"""大きな検証セットで複数モデルの「棋譜の真似の上手さ」を比べる (2026-09-24)。

動機: 学習時の val は 32 局 (約 4,000 局面) しかなく、一致率で ±0.8pt、policy 損失で ±0.03 の
誤差がある。e3 から 1 epoch 追加したモデルが 3 本とも外部強さで e3 を下回った (0.200 / 0.155 /
0.125 vs 0.220) が、その原因が

  H1 過学習 (覚え込みで新しい局面に弱くなった) → 大きな val で val 損失が悪化しているはず
  H3 目的のずれ (真似は上手くなったが強さに繋がらない) → val 損失は改善しているはず

のどちらかを、32 局の val では判定できない。どのモデルも学習に使っていない 2026 年の
floodgate 棋譜 (学習は 2023〜2025 年) を 1,000 局使って測り直す。

同じ局を同じ順で全モデルに通すので、基準モデルとの差は**バッチ単位の対応あり**で誤差を出す。

    uv run python scripts/ops/eval_val.py --shard-dir data/shards/val2026 \\
        --models phase37_e3 phase37_e4 stage1_control --reference phase37_e3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kokoro_shogi.config import load_config
from kokoro_shogi.data.dataset import find_shards
from kokoro_shogi.data.sequence import SequenceDataset, collate_sequences
from kokoro_shogi.model.mood import MoodGRU
from kokoro_shogi.train.mood_distill import build_mood_policy, run_epoch

REPO = Path(__file__).resolve().parents[2]


def evaluate(name: str, loader: DataLoader, config, device: torch.device) -> list[dict]:
    """1 モデルを全バッチに通し、バッチごとの指標を返す (対応あり比較のため)。"""
    path = REPO / "checkpoints" / f"{name}.pt"
    policy = build_mood_policy(config, path, device, mood=True, relations=True, council=True)
    gru = MoodGRU(config.model).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "mood_gru" in ckpt:
        gru.load_state_dict(ckpt["mood_gru"])
    policy.eval()
    gru.eval()
    rows = []
    for batch in loader:
        m = run_epoch(policy, gru, [batch], config, device, tbptt=16)
        rows.append(
            {"n": m.positions, "acc": m.accuracy, "pol": m.policy_loss, "val": m.value_loss}
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    means = {k: sum(r[k] * r["n"] for r in rows) / n for k in ("acc", "pol", "val")}
    return means | {"positions": n}


def paired(rows: list[dict], ref: list[dict], key: str) -> tuple[float, float]:
    """バッチ単位の対応ありの差 (重み = 局面数) とその標準誤差。"""
    w = [r["n"] for r in rows]
    d = [a[key] - b[key] for a, b in zip(rows, ref, strict=True)]
    total = sum(w)
    mean = sum(wi * di for wi, di in zip(w, d, strict=True)) / total
    var = sum(wi * wi * (di - mean) ** 2 for wi, di in zip(w, d, strict=True)) / total**2
    return mean, math.sqrt(var)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shard-dir", type=Path, default=REPO / "data" / "shards" / "val2026")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--reference", default="phase37_e3")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--games-per-batch", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=REPO / "checkpoints" / "bigval")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config()
    # 比較の軸は policy / value だけにそろえる (教師・運命・転生の損失は切る)
    config = replace(config, loss=replace(
        config.loss, c_soft=0.0, teacher_value_weight=0.0, lambda_fate=0.0, lambda_rebirth=0.0))
    dataset = SequenceDataset(find_shards(args.shard_dir), max_games=args.max_games)
    loader = DataLoader(dataset, batch_size=args.games_per_batch, shuffle=False,
                        collate_fn=collate_sequences, num_workers=args.workers)
    print(f"検証セット: {len(dataset)} 局 / {args.shard_dir} / device {device}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for name in dict.fromkeys([args.reference, *args.models]):
        started = time.perf_counter()
        results[name] = evaluate(name, loader, config, device)
        s = summarize(results[name])
        payload = {"summary": s, "batches": results[name]}
        (args.out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1))
        print(f"  {name:<22} 一致率 {s['acc']*100:6.2f}%  policy {s['pol']:.4f}"
              f"  value {s['val']:.4f}"
              f"  ({s['positions']:,} 局面, {time.perf_counter()-started:.0f}s)", flush=True)

    ref = results[args.reference]
    print(f"\n=== {args.reference} との差 (対応あり、±は 1 SE) ===")
    print(f"{'モデル':<22}{'一致率':>16}{'policy 損失':>20}{'value 損失':>20}")
    for name, rows in results.items():
        if name == args.reference:
            continue
        a, sa = paired(rows, ref, "acc")
        p, sp = paired(rows, ref, "pol")
        v, sv = paired(rows, ref, "val")
        print(f"{name:<22}{a*100:+8.2f}±{sa*100:.2f}pt{p:+11.4f}±{sp:.4f}{v:+11.4f}±{sv:.4f}")
    print("\n読み方: policy 損失が e3 より有意に悪化 (+) → H1 過学習 /"
          " 有意に改善 (−) なのに外部強さは落ちた → H3 目的のずれ")


if __name__ == "__main__":
    main()
