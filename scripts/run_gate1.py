"""Phase 1 の Gate1 を判定する (DESIGN.md §6 Phase 1)。

Gate1 = **駒トークンTransformerの一致率が、CNNベースライン比 -2%以内**。

2つのモデルを**同一の設定・同一の train/val 分割**で学習し、検証一致率を比べる。
`distill.py` を2回叩いても同じことはできるが、片方だけ引数を変えてしまうと
比較が無効になるため、Gate の判定はこのスクリプトから行う。

使い方 (GPUマシン gmk02 で実行する)::

    uv run python scripts/run_gate1.py --shard-dir data/shards/2024 --epochs 10 \\
        --batch-size 1024 --device cuda

結果は `checkpoints/gate1.json` に残る。ADR
`docs/decisions/2026-07-26-phase1-policy-and-baseline.md` へ転記すること。
"""

from __future__ import annotations

import argparse
import json
import time
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

#: Gate1 の合格ライン (CNNベースラインからの許容低下幅)
TOLERANCE = 0.02


def run_one(kind: str, args: argparse.Namespace) -> dict[str, Any]:
    """1モデルを学習して、各エポックの検証一致率を返す。"""
    config = load_config()
    # 両モデルで初期化・シャッフル順を揃える
    set_global_seed(config.seed)
    device = resolve_device(args.device)

    train_loader, val_loader = build_loaders(
        args.shard_dir,
        batch_size=args.batch_size,
        max_positions=args.max_positions,
        workers=args.workers,
        val_ratio=args.val_ratio,
    )

    model = build_model(
        kind, config, channels=args.cnn_channels, blocks=args.cnn_blocks
    ).to(device)
    parameters = sum(tensor.numel() for tensor in model.parameters())
    print(f"\n=== {kind} (params {parameters / 1e6:.2f}M, device {device}) ===")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(total_steps * args.warmup_ratio),
        total_steps=total_steps,
    )

    history = []
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, optimizer, config, device, scheduler=scheduler
        )
        val_metrics = evaluate(model, val_loader, config, device)
        elapsed = time.perf_counter() - started

        print(f"  epoch {epoch}: train {train_metrics.format()}")
        print(f"           val   {val_metrics.format()}  ({elapsed:.1f}秒)")

        history.append({"epoch": epoch, "val_accuracy": val_metrics.accuracy})
        if val_metrics.accuracy > best:
            best = val_metrics.accuracy
            torch.save(
                {"model": model.state_dict(), "kind": kind, "epoch": epoch},
                args.out_dir / f"{kind}_gate1_best.pt",
            )

    return {
        "kind": kind,
        "params": parameters,
        "best_val_accuracy": best,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cnn-channels", type=int, default=192)
    parser.add_argument("--cnn-blocks", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_one("cnn", args)
    kokoro = run_one("kokoro", args)

    gap = kokoro["best_val_accuracy"] - baseline["best_val_accuracy"]
    passed = gap >= -TOLERANCE

    print("\n=== Gate1 ===")
    print(f"CNNベースライン 一致率: {baseline['best_val_accuracy'] * 100:.2f}%")
    print(f"駒トークン      一致率: {kokoro['best_val_accuracy'] * 100:.2f}%")
    print(f"差: {gap * 100:+.2f}%  (合格ライン {-TOLERANCE * 100:+.0f}%)")
    print("判定: " + ("通過" if passed else "不通過"))
    if not passed:
        print(
            "  → DESIGN.md §9 の対策: 利きバイアス強化 / "
            "マス+駒ハイブリッドトークン化への後退を検討する"
        )

    result = {
        "tolerance": TOLERANCE,
        "gap": gap,
        "passed": passed,
        "baseline": baseline,
        "kokoro": kokoro,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    output = args.out_dir / "gate1.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{output} に保存しました。")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()


__all__ = ["TOLERANCE", "run_one"]
