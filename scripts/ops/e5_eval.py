"""E5 評価: 文化プール PPO vs 自己プール PPO (対照) vs ppo2 の総当たり。

uv run python scripts/ops/e5_eval.py --games 80 --out checkpoints/e5_eval.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from kokoro_shogi.config import load_config, set_global_seed
from kokoro_shogi.model.mood import MoodGRU
from kokoro_shogi.model.policy import KokoroPolicy
from kokoro_shogi.train.distill import resolve_device
from kokoro_shogi.train.selfplay_ppo import play_match

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=80)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument(
    "--models", nargs="+",
    default=["checkpoints/ppo2.pt", "checkpoints/ppo_e5_culture.pt", "checkpoints/ppo_e5_control.pt"],
)
args = parser.parse_args()

config = load_config()
set_global_seed(config.seed)
device = resolve_device(None)
models = {}
gru = None
for path in args.models:
    state = torch.load(path, map_location=device, weights_only=True)
    saved = state.get("features", {})
    flags = replace(
        config.features, mood=True,
        relations=bool(saved.get("relations", False)), council=bool(saved.get("council", False)),
    )
    m = KokoroPolicy(config.model, flags, tau=config.loss.tau, head="desire").to(device)
    m.load_state_dict(state["model"])
    m.eval()
    models[Path(path).stem] = m
    if gru is None:
        gru = MoodGRU(config.model).to(device)
        gru.load_state_dict(state["mood_gru"])
        gru.eval()

results = []
for k, (a, b) in enumerate(itertools.combinations(models, 2)):
    started = time.perf_counter()
    wr = play_match(models[a], models[b], gru, args.games, device, seed=config.seed + 7 * k)
    results.append({"a": a, "b": b, "win_rate_a": wr, "games": args.games})
    print(f"{a} vs {b}: {wr:.3f} ({time.perf_counter() - started:.0f}秒)")
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
