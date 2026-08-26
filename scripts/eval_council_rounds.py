"""会議ラウンド数 R=0..4 の棋力比較 (Phase 4 Gate、DESIGN.md §6)。

会議 [D] は trunk 最終2層の重み共有再適用なので、**同じ重みのまま** R を変えて
評価できる (Universal Transformer の性質)。R=0 は会議OFF相当。

使い方::

    uv run python scripts/eval_council_rounds.py --checkpoint checkpoints/phase34.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kokoro_shogi.config import REPO_ROOT, load_config
from kokoro_shogi.data.dataset import find_shards
from kokoro_shogi.data.sequence import SequenceDataset, collate_sequences
from kokoro_shogi.model.mood import MoodGRU
from kokoro_shogi.train.distill import resolve_device
from kokoro_shogi.train.mood_distill import build_mood_policy, run_epoch

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "phase34.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shard-dir", type=Path, default=REPO_ROOT / "data" / "shards" / "2024")
    parser.add_argument("--val-games", type=int, default=64, help="学習時と同じ先頭N局")
    parser.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = load_config()

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    features = state.get("features", {})
    policy = build_mood_policy(
        config, None, device,
        relations=features.get("relations", False),
        council=features.get("council", True),
    )
    policy.load_state_dict(state["model"])
    gru = MoodGRU(config.model).to(device)
    gru.load_state_dict(state["mood_gru"])

    dataset = SequenceDataset(find_shards(args.shard_dir), max_games=args.val_games)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, collate_fn=collate_sequences, num_workers=4
    )

    results = []
    print(f"{'R':>3} {'一致率':>10} {'loss':>10} {'説明率':>10}")
    for rounds in args.rounds:
        metrics = run_epoch(policy, gru, loader, config, device, tbptt=1, rounds=rounds)
        print(
            f"{rounds:>3} {metrics.accuracy * 100:>9.2f}% {metrics.loss:>10.4f} "
            f"{metrics.explained_ratio * 100:>9.1f}%"
        )
        results.append({"rounds": rounds, "accuracy": metrics.accuracy, "loss": metrics.loss,
                        "explained_ratio": metrics.explained_ratio})

    out = args.checkpoint.with_name("council_rounds_eval.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{out.name} に保存しました。")


if __name__ == "__main__":
    main()
