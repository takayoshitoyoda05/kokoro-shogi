"""$\\lambda_g$ 掃引でトレードオフ曲線を作る (DESIGN.md §4a 評価 / Phase 2 Gate)。

自由項 $g$ のL2罰則 $\\lambda_g$ を振ると、

- $\\lambda_g$ 小: $g$ が自由に働ける → **一致率は上がる**が、手の理由が
  欲求で説明できなくなる (説明率が下がる)
- $\\lambda_g$ 大: $g$ が抑えられる → 手が $\\langle w_i, d_i(a)\\rangle$ で
  説明できるようになるが、**大局観を捨てるので一致率が下がる**

この2軸を結んだ曲線が「棋力 vs 解釈可能性」のトレードオフで、DESIGN.md が
論文の主図と呼んでいるもの。10週の縮退版では3点で足りる (TEAM_PLAN §3)。

使い方 (GPUマシンで実行する)::

    uv run python scripts/sweep_lambda_g.py --shard-dir data/shards/2024 \\
        --epochs 5 --batch-size 512 --device cuda --lambdas 0.0 0.01 0.1 1.0

結果は `checkpoints/lambda_g_sweep.json` に残る。曲線の形と、
$|g|$ 上位手が捨て駒になっているかの定性確認をADRへ書くこと。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from kokoro_shogi.config import load_config, set_global_seed
from kokoro_shogi.train.distill import (
    DEFAULT_OUT_DIR,
    DEFAULT_SHARD_DIR,
    build_loaders,
    build_model,
    build_scheduler,
    evaluate,
    resolve_device,
    train_epoch,
)

#: 縮退版の既定 (TEAM_PLAN §3: 10週内は3点)
DEFAULT_LAMBDAS = (0.0, 0.01, 0.1)


def run_one(lambda_g: float, args: argparse.Namespace) -> dict[str, Any]:
    """1つの $\\lambda_g$ で学習し、最終エポックの一致率と説明率を返す。"""
    base = load_config()
    set_global_seed(base.seed)
    # lambda_g 以外は完全に同じ設定にする (曲線の各点を比較可能にするため)
    config = replace(base, loss=replace(base.loss, lambda_g=lambda_g))
    device = resolve_device(args.device)

    train_loader, val_loader = build_loaders(
        args.shard_dir,
        batch_size=args.batch_size,
        max_positions=args.max_positions,
        workers=args.workers,
        val_ratio=args.val_ratio,
    )

    model = build_model("kokoro", config, head="desire").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(total_steps * args.warmup_ratio),
        total_steps=total_steps,
    )

    print(f"\n=== lambda_g = {lambda_g} ===")
    metrics = None
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_epoch(model, train_loader, optimizer, config, device, scheduler=scheduler)
        metrics = evaluate(model, val_loader, config, device)
        print(f"  epoch {epoch}: val {metrics.format()}  ({time.perf_counter() - started:.1f}秒)")

    torch.save(
        {"model": model.state_dict(), "lambda_g": lambda_g},
        args.out_dir / f"desire_lambda{lambda_g}.pt",
    )
    return {"lambda_g": lambda_g, **asdict(metrics)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS),
        help="掃引する lambda_g の値",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = [run_one(value, args) for value in args.lambdas]

    print("\n=== トレードオフ曲線 (DESIGN.md §4a) ===")
    print(f"{'lambda_g':>10} {'一致率':>10} {'説明率':>10} {'g²':>10}")
    for point in points:
        print(
            f"{point['lambda_g']:>10.4g} {point['accuracy'] * 100:>9.2f}% "
            f"{point['explained_ratio'] * 100:>9.1f}% {point['free_term_penalty']:>10.4f}"
        )

    output = args.out_dir / "lambda_g_sweep.json"
    output.write_text(
        json.dumps({"points": points}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{output} に保存しました。")


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_LAMBDAS", "run_one"]
