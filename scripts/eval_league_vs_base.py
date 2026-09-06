"""リーグ最終文化 vs 基点モデルの対局評価 (淘汰は絶対強さを上げたか)。

各リーグの league.pt について、(共有 trunk + 文化 θ_sp) を challenger、
基点チェックポイント (例: ppo2.pt) を incumbent として play_match で勝率を測る。
trunk の漂流と文化 θ_sp の効果を分けるため、「trunk + 基点 θ_sp」も 1 行測る。

使い方::

    uv run python scripts/eval_league_vs_base.py --base checkpoints/ppo2.pt \\
        --leagues checkpoints/league_run2_6x80 checkpoints/league_E1_control \\
        --games 40 --out checkpoints/league_vs_base.json
"""

from __future__ import annotations

import argparse
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


def build(state: dict, config, device: torch.device) -> KokoroPolicy:
    saved = state.get("features", {})
    flags = replace(
        config.features,
        mood=True,
        relations=bool(saved.get("relations", False)),
        council=bool(saved.get("council", False)),
    )
    model = KokoroPolicy(config.model, flags, tau=config.loss.tau, head="desire").to(device)
    model.load_state_dict(state["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--leagues", type=Path, nargs="+", required=True, help="league.pt を含むディレクトリ"
    )
    parser.add_argument("--games", type=int, default=40, help="文化ごとの対局数 (先後交互)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config.seed)
    device = resolve_device(args.device)

    base_state = torch.load(args.base, map_location=device, weights_only=True)
    base = build(base_state, config, device)
    base_theta = base.personality.theta_species.weight.detach().clone()
    gru = MoodGRU(config.model).to(device)
    gru.load_state_dict(base_state["mood_gru"])
    gru.eval()

    results: dict[str, dict] = {}
    for league_dir in args.leagues:
        state = torch.load(league_dir / "league.pt", map_location=device, weights_only=True)
        model = build(state, config, device)
        rows: dict[str, float] = {}
        # trunk 漂流だけの効果: 文化 θ_sp を基点 θ_sp に戻して測る
        model.personality.theta_species.weight.data.copy_(base_theta)
        started = time.perf_counter()
        rows["trunk+base_theta"] = play_match(
            model, base, gru, args.games, device,
            tau=args.tau, max_plies=args.max_plies, seed=config.seed,
        )
        elapsed = time.perf_counter() - started
        print(f"{league_dir.name} trunk+base: {rows['trunk+base_theta']:.3f} ({elapsed:.0f}秒)")
        for k, (name, culture) in enumerate(state["cultures"].items(), start=1):
            model.personality.theta_species.weight.data.copy_(culture["theta_sp"].to(device))
            started = time.perf_counter()
            rows[name] = play_match(
                model, base, gru, args.games, device,
                tau=args.tau, max_plies=args.max_plies, seed=config.seed + k,
            )
            elapsed = time.perf_counter() - started
            print(f"{league_dir.name} {name}: {rows[name]:.3f} ({elapsed:.0f}秒)")
        cultures = [v for k, v in rows.items() if k != "trunk+base_theta"]
        rows["cultures_mean"] = sum(cultures) / len(cultures)
        results[league_dir.name] = rows
        args.out.write_text(
            json.dumps(
                {"base": args.base.name, "games": args.games, "tau": args.tau, "results": results},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        del model
    for name, rows in results.items():
        print(
            f"== {name}: 文化平均 {rows['cultures_mean']:.3f}"
            f" / trunk のみ {rows['trunk+base_theta']:.3f}"
        )


if __name__ == "__main__":
    main()
