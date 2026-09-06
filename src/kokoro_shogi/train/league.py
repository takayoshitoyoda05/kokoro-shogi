"""文化リーグ (PBT) [F] (DESIGN.md §4d)。

文化 $k$ = 駒種性格セット $\\Phi_k = \\{\\theta^{sp}_c\\}_c$ + ハイパー $(\\tau, \\lambda_g)_k$。
trunk・ヘッドは全文化で共有し (AlphaStarリーグの共有バックボーンに相当)、
文化ごとに $\\Phi_k$ を差し替えて自己対戦PPOを回す。リーグ節ごとに:

1. 各文化が自分の $\\Phi_k$ で数イテレーションのPPO (train/selfplay_ppo.py を流用)
2. 総当たり対戦で勝率 $\\mathrm{WR}_k$ を計測
3. 下位を淘汰: $\\Phi_k \\leftarrow \\Phi_{k^\\ast} + \\mathcal{N}(0, \\sigma^2_{PBT})$、
   $(\\tau, \\lambda_g)_k \\leftarrow (\\tau, \\lambda_g)_{k^\\ast} \\cdot e^{\\mathcal{N}(0, 0.2^2)}$
4. 文化分化の観測: $\\|\\Phi_k - \\Phi_{k'}\\|$ の世代推移 (DESIGN.md §8)

割り切り (縮退版): optimizer は全文化で共有 (モーメントが文化間で混ざるが、
節あたりの更新が少ないので許容)。本将棋のまま回す (ユーザー決定)。

使い方::

    uv run python -m kokoro_shogi.train.league --cultures 4 --generations 5

2026-09-04 追加: ``--grace`` (新生の猶予), ``--newborn-ppo-iters`` (新生のウォームアップ),
``--selection random`` (対照実験), ``--fix-hypers`` (ハイパー固定)。既定値では従来と同じ挙動。
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from kokoro_shogi.config import Config, load_config, set_global_seed
from kokoro_shogi.model.mood import MoodGRU
from kokoro_shogi.model.policy import KokoroPolicy
from kokoro_shogi.train.distill import DEFAULT_OUT_DIR, resolve_device
from kokoro_shogi.train.selfplay_ppo import collect_games, play_match, ppo_update

DEFAULT_CHECKPOINT = DEFAULT_OUT_DIR / "phase34.pt"

#: Φ への変異ノイズは重みの標準偏差に対する相対値で決める
SIGMA_PBT_RELATIVE = 0.05
#: ハイパー摂動の対数正規スケール (DESIGN.md §4d の 0.2)
HYPER_JITTER = 0.2


@dataclass
class Culture:
    """1つの文化圏。Φ (θ_sp 埋め込み) と探索ハイパーを持つ。"""

    name: str
    theta_sp: torch.Tensor  # (NUM_SPECIES, d_theta)、CPUに保持
    tau: float
    lambda_g: float
    win_rate: float = 0.5
    age: int = 0  # 置換 (誕生) からの世代数。猶予期間とウォームアップの判定に使う

    def config_for(self, base: Config) -> Config:
        return replace(base, loss=replace(base.loss, lambda_g=self.lambda_g))


def swap_in(model: KokoroPolicy, culture: Culture) -> None:
    model.personality.theta_species.weight.data.copy_(
        culture.theta_sp.to(model.personality.theta_species.weight.device)
    )


def swap_out(model: KokoroPolicy, culture: Culture) -> None:
    culture.theta_sp = model.personality.theta_species.weight.detach().cpu().clone()


def culture_distances(cultures: list[Culture]) -> dict:
    """$\\|\\Phi_k - \\Phi_{k'}\\|_F$ の全ペアと平均 (文化分化の観測量)。"""
    pairs = {}
    values = []
    for i, a in enumerate(cultures):
        for b in cultures[i + 1 :]:
            distance = float(torch.norm(a.theta_sp - b.theta_sp))
            pairs[f"{a.name}-{b.name}"] = round(distance, 4)
            values.append(distance)
    return {"mean": float(np.mean(values)) if values else 0.0, "pairs": pairs}


def round_robin(
    model: KokoroPolicy,
    rival: KokoroPolicy,
    gru: MoodGRU,
    cultures: list[Culture],
    device: torch.device,
    *,
    games_per_pair: int,
    max_plies: int,
    seed: int,
) -> None:
    """総当たりで win_rate を更新する。"""
    points = {culture.name: 0.0 for culture in cultures}
    matches = {culture.name: 0 for culture in cultures}
    for i, a in enumerate(cultures):
        for j, b in enumerate(cultures[i + 1 :], start=i + 1):
            swap_in(model, a)
            swap_in(rival, b)
            score = play_match(
                model, rival, gru, games_per_pair, device,
                max_plies=max_plies, seed=seed + i * 100 + j,
            )
            points[a.name] += score * games_per_pair
            points[b.name] += (1 - score) * games_per_pair
            matches[a.name] += games_per_pair
            matches[b.name] += games_per_pair
    for culture in cultures:
        if matches[culture.name]:
            culture.win_rate = points[culture.name] / matches[culture.name]


def select_and_mutate(
    cultures: list[Culture],
    rng: np.random.Generator,
    *,
    replaced: int,
    sigma_relative: float = SIGMA_PBT_RELATIVE,
    grace: int = 0,
    selection: str = "fitness",
    fix_hypers: bool = False,
) -> tuple[list[str], str]:
    """下位 `replaced` 文化を、最良文化の複製+変異で置き換える (PBT)。

    `sigma_relative` は変異ノイズの std を最良文化 θ_sp の std に対する比で与える
    (既定 0.05。2026-09-04 の 4文化×50世代では複製が続くと集団が均質化したため、
    分化の再生を速める目的で引き上げられるようにした)。

    `grace`: 誕生から `grace` 世代未満の文化は淘汰候補から外す (新生の猶予)。
    2026-09-04 の観測では、複製+変異で生まれた文化はコピー元より約10ポイント弱く、
    次の世代で再び最下位になって置き換えられる「回転ドア」が run1 で 21/49、
    run2 で 9/13 の世代に起きた。全員が猶予中なら最年長から選ぶ。
    `selection`: "fitness" (勝率順) か "random" (淘汰先・コピー元とも一様乱択;
    淘汰の効果を分離する対照実験用)。
    `fix_hypers`: True なら (τ, λ_g) を摂動せず親の値をそのまま継ぐ。

    戻り値は (置き換えた文化名, コピー元の文化名)。
    """
    if selection == "random":
        order = list(rng.permutation(len(cultures)))
        best = cultures[order[0]]
        eligible = [cultures[i] for i in order[1:]]
    elif selection == "fitness":
        ranked = sorted(cultures, key=lambda culture: culture.win_rate, reverse=True)
        best = ranked[0]
        eligible = [c for c in ranked[1:] if c.age >= grace]
        if len(eligible) < replaced:  # 全員猶予中なら最年長 (= 最も長く生き残った下位) を対象
            eligible = sorted(ranked[1:], key=lambda culture: culture.age)  # 末尾が最年長
    else:
        raise ValueError(f"unknown selection: {selection}")
    sigma = sigma_relative * float(best.theta_sp.std().clamp(min=1e-6))
    renewed = []
    for culture in eligible[-replaced:]:
        noise = torch.from_numpy(
            rng.normal(0.0, sigma, size=tuple(best.theta_sp.shape)).astype(np.float32)
        )
        culture.theta_sp = best.theta_sp.clone() + noise
        if fix_hypers:
            culture.tau, culture.lambda_g = best.tau, best.lambda_g
        else:
            culture.tau = float(best.tau * np.exp(rng.normal(0.0, HYPER_JITTER)))
            culture.lambda_g = float(best.lambda_g * np.exp(rng.normal(0.0, HYPER_JITTER)))
        culture.age = 0
        renewed.append(culture.name)
    return renewed, best.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cultures", type=int, default=4)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--ppo-iters", type=int, default=2, help="1節あたり各文化のPPO回数")
    parser.add_argument("--games-per-iter", type=int, default=8)
    parser.add_argument("--games-per-pair", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--replaced", type=int, default=1, help="1節で淘汰される文化数")
    parser.add_argument(
        "--mutation-sigma", type=float, default=SIGMA_PBT_RELATIVE,
        help="複製時の変異ノイズ std (最良文化 θ_sp の std に対する比)",
    )
    parser.add_argument(
        "--grace", type=int, default=0,
        help="誕生から何世代は淘汰対象外にするか (新生の回転ドア対策、0 で無効)",
    )
    parser.add_argument(
        "--newborn-ppo-iters", type=int, default=None,
        help="誕生直後 (age 0) の文化だけに使うPPO回数 (既定: --ppo-iters と同じ)",
    )
    parser.add_argument(
        "--selection", choices=("fitness", "random"), default="fitness",
        help="淘汰方式。random は淘汰先とコピー元を一様乱択する対照実験用",
    )
    parser.add_argument(
        "--fix-hypers", action="store_true",
        help="(τ, λ_g) を初期値 τ=0.8, λ_g=設定値に固定し摂動しない (変数を θ_sp だけにする)",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    newborn_ppo_iters = args.ppo_iters if args.newborn_ppo_iters is None else args.newborn_ppo_iters

    config = load_config()
    set_global_seed(config.seed)
    device = resolve_device(args.device)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved = state.get("features", {})
    flags = replace(
        config.features,
        mood=True,
        relations=bool(saved.get("relations", False)),
        council=bool(saved.get("council", False)),
        league=True,
    )
    model = KokoroPolicy(config.model, flags, tau=config.loss.tau, head="desire").to(device)
    model.load_state_dict(state["model"])
    rival = copy.deepcopy(model)  # 総当たり用の相手側 (Φだけ差し替える)
    for parameter in rival.parameters():
        parameter.requires_grad_(False)
    gru = MoodGRU(config.model).to(device)
    gru.load_state_dict(state["mood_gru"])
    gru.eval()

    base_theta = model.personality.theta_species.weight.detach().cpu()
    init_sigma = SIGMA_PBT_RELATIVE * float(base_theta.std().clamp(min=1e-6))
    cultures = [
        Culture(
            name=f"culture{k}",
            theta_sp=base_theta.clone()
            + torch.from_numpy(
                rng.normal(0.0, init_sigma, size=tuple(base_theta.shape)).astype(np.float32)
            ),
            tau=0.8 if args.fix_hypers else float(0.8 * np.exp(rng.normal(0.0, HYPER_JITTER))),
            lambda_g=(
                config.loss.lambda_g if args.fix_hypers
                else float(config.loss.lambda_g * np.exp(rng.normal(0.0, HYPER_JITTER)))
            ),
        )
        for k in range(args.cultures)
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    print(f"warm start: {args.checkpoint.name} / {args.cultures}文化 / device: {device}")

    history = []
    for generation in range(1, args.generations + 1):
        started = time.perf_counter()
        for culture in cultures:
            swap_in(model, culture)
            culture_config = culture.config_for(config)
            iters = newborn_ppo_iters if culture.age == 0 else args.ppo_iters
            for _ in range(iters):
                steps, _stats = collect_games(
                    model, gru, args.games_per_iter, device,
                    tau=culture.tau, max_plies=args.max_plies, generator=generator,
                )
                if steps:
                    ppo_update(
                        model, optimizer, steps, culture_config, device,
                        epochs=1, minibatch=256, generator=generator,
                    )
            swap_out(model, culture)

        round_robin(
            model, rival, gru, cultures, device,
            games_per_pair=args.games_per_pair, max_plies=args.max_plies,
            seed=config.seed + generation * 1000,
        )
        ages = {c.name: c.age for c in cultures}  # 淘汰前の年齢 (この世代の評価時点)
        renewed, parent = select_and_mutate(
            cultures, rng, replaced=args.replaced, sigma_relative=args.mutation_sigma,
            grace=args.grace, selection=args.selection, fix_hypers=args.fix_hypers,
        )
        for culture in cultures:
            if culture.name not in renewed:
                culture.age += 1
        distances = culture_distances(cultures)
        elapsed = time.perf_counter() - started

        entry = {
            "generation": generation,
            "win_rates": {c.name: round(c.win_rate, 3) for c in cultures},
            "hypers": {c.name: {"tau": round(c.tau, 3), "lambda_g": round(c.lambda_g, 4)} for c in cultures},
            "renewed": renewed,
            "parent": parent,
            "ages": ages,
            "selection": args.selection,
            "phi_distance": distances,
            "seconds": round(elapsed, 1),
        }
        history.append(entry)
        print(
            f"gen {generation}: WR {entry['win_rates']} / 淘汰 {renewed} (親 {parent})"
            f" / ||Φ-Φ'||平均 {distances['mean']:.4f} ({elapsed:.0f}秒)"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "mood_gru": gru.state_dict(),
                "mood_projection": state.get("mood_projection", {}),
                "features": saved,
                "cultures": {
                    c.name: {
                        "theta_sp": c.theta_sp, "tau": c.tau, "lambda_g": c.lambda_g, "age": c.age,
                    }
                    for c in cultures
                },
            },
            args.out_dir / "league.pt",
        )
        (args.out_dir / "league_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()


__all__ = ["Culture", "culture_distances", "round_robin", "select_and_mutate", "swap_in", "swap_out"]
