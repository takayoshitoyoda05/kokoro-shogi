"""感情GRU [A] を通した系列蒸留 (Phase 3、DESIGN.md §3(9) + §4a)。

蒸留損失は distill.py と同じ統一損失。違いは局面を独立に流すのではなく、
**1局を時系列で流しながら感情状態 $m_i$ を引き回す**こと:

$$m^{(t)} = \\mathrm{GRU}(u^{ev}(t),\\, m^{(t-1)}), \\qquad
  \\pi, V, d = \\mathrm{Policy}(s_t;\\, m^{(t)})$$

勾配は truncated BPTT (既定16手) で切る。1局丸ごと逆伝播すると150手ぶんの
計算グラフが載って VRAM が持たないため。

Phase 2 チェックポイント (desire_lambda*.pt) からのウォームスタートを前提にする。
trunk の合流点 $W_m$ は零初期化 (model/trunk.py) なので、学習開始時点の方策は
Phase 2 と完全に同じところから、感情を使うと損失が下がる方向にだけ動く。
`PersonalityWeights.project` だけは入力次元が広がるため再初期化される。

MoodProjection (Unityへ送る3軸への射影) はここでは学習しない (教師が無い)。
学習後に `scripts/fit_mood_projection.py` でヒューリスティック mood への回帰で
アンカーする (ADR 2026-08-25)。

使い方::

    uv run python -m kokoro_shogi.train.mood_distill --max-games 400 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from kokoro_shogi.config import Config, load_config, set_global_seed
from kokoro_shogi.core.tokenizer import MAX_PIECES
from kokoro_shogi.data.dataset import find_shards
from kokoro_shogi.data.sequence import SequenceDataset, collate_sequences
from kokoro_shogi.model.mood import MoodGRU, MoodProjection
from kokoro_shogi.model.policy import KokoroPolicy, PolicyOutput
from kokoro_shogi.model.relations import initial_relations, update_relations
from kokoro_shogi.train.distill import (
    _AVERAGED,
    DEFAULT_OUT_DIR,
    DEFAULT_SHARD_DIR,
    Metrics,
    compute_loss_tensors,
    move_batch,
    resolve_device,
)

DEFAULT_WARM_START = DEFAULT_OUT_DIR / "desire_lambda0.1.pt"

#: forward だけに使う入力 (mood は別に載せる)
_STEP_KEYS = (
    "species",
    "position",
    "owner",
    "promoted",
    "mask",
    "turn",
    "effect",
    "legal",
    "action",
    "result",
    "labels",
)


def build_mood_policy(
    config: Config,
    warm_start: Path | None,
    device: torch.device,
    *,
    mood: bool = True,
    relations: bool = False,
    council: bool = False,
) -> KokoroPolicy:
    """機能フラグ組合せの KokoroPolicy を Phase 2 の重みから立ち上げる。

    `mood=False` はアブレーション (§8: A/C/D の全組合せ表) の計測用。
    """
    flags = replace(config.features, mood=mood, relations=relations, council=council)
    model = KokoroPolicy(config.model, flags, tau=config.loss.tau, head="desire").to(device)

    if warm_start is not None:
        loaded = torch.load(warm_start, map_location=device, weights_only=True)["model"]
        own = model.state_dict()
        # 形が合わないキーだけ捨てる (personality.project は mood の連結で
        # 入力次元が変わることがある。mood 学習済みチェックポイントからなら残る)
        state = {k: v for k, v in loaded.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        allowed = ("mood", "personality", "relation", "proposal")
        assert all(any(word in k for word in allowed) for k in missing), missing
    return model


def run_epoch(
    policy: KokoroPolicy,
    gru: MoodGRU,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    *,
    tbptt: int,
    optimizer: torch.optim.Optimizer | None = None,
    rounds: int | None = None,
    amp: bool = False,
) -> Metrics:
    """1エポック。optimizer が None なら評価 (勾配なし)。

    `rounds` は会議 [D] のラウンド数の上書き (評価時の R=0..4 比較用)。
    None ならモデル既定 (DEFAULT_ROUNDS)。
    `amp` は bfloat16 autocast (Ampere以降のGPU向け。GradScaler不要)。
    """
    training = optimizer is not None
    policy.train(training)
    gru.train(training)
    total = Metrics()

    with torch.set_grad_enabled(training):
        for batch in loader:
            # 有効手のマスクはCPU側で先に読む。GPU転送後に毎手 `.any()` /
            # `.nonzero()` すると1手ごとにGPU同期が入り、それだけで
            # メインプロセスが1コア張り付く (実測)。
            steps_np = batch["steps"].numpy().astype(bool)
            plies_active = steps_np.any(axis=0)
            t_end = int(plies_active.sum())  # 詰め物は末尾のみなので有効手数=先頭の連続区間
            all_active = steps_np.all(axis=0)

            batch = move_batch(batch, device)
            games, _plies = steps_np.shape
            mood = (
                gru.initial_state(games, MAX_PIECES, device=device)
                if policy.features.mood
                else None
            )
            relation = (
                initial_relations(games, MAX_PIECES, device=device)
                if policy.features.relations
                else None
            )

            # 集計はテンソルのまま積み、バッチ末尾で一度だけ同期する
            sums = {name: torch.zeros((), device=device) for name in _AVERAGED}
            positions_total = 0

            window: list[Tensor] = []
            for t in range(t_end):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    if mood is not None:
                        mood = gru(batch["events"][:, t], mood)
                    if relation is not None:
                        relation = update_relations(relation, batch["effect"][:, t])

                    if all_active[t]:
                        step = {key: batch[key][:, t] for key in _STEP_KEYS}
                        if mood is not None:
                            step["mood"] = mood
                        if relation is not None:
                            step["relation"] = relation
                    else:
                        rows = torch.as_tensor(
                            np.flatnonzero(steps_np[:, t]), device=device
                        )
                        step = {key: batch[key][rows, t] for key in _STEP_KEYS}
                        if mood is not None:
                            step["mood"] = mood[rows]
                        if relation is not None:
                            step["relation"] = relation[rows]
                    output = policy.forward_batch(step, rounds=rounds)
                # 損失は fp32 で計算する (binary_cross_entropy は autocast 禁止、
                # かつ log の数値安定性のため)。matmul 群は上の autocast 内で
                # bf16 実行済みなので速度効果は保たれる
                if amp:
                    output = _float_output(output)
                loss, parts, positions = compute_loss_tensors(output, step, config)
                for name in _AVERAGED:
                    sums[name] += parts[name] * positions
                positions_total += positions

                if training:
                    window.append(loss)
                    if len(window) >= tbptt:
                        _flush_window(window, policy, gru, optimizer)
                        if mood is not None:
                            mood = mood.detach()

            if training and window:
                _flush_window(window, policy, gru, optimizer)

            if positions_total:
                total.update(
                    Metrics(
                        **{
                            name: float(sums[name]) / positions_total
                            for name in _AVERAGED
                        },
                        positions=positions_total,
                    )
                )

    return total


def _float_output(output: PolicyOutput) -> PolicyOutput:
    """bf16 autocast の forward 出力を fp32 に揃える (損失計算用)。"""
    converted = {}
    for field in fields(output):
        value = getattr(output, field.name)
        converted[field.name] = value.float() if isinstance(value, Tensor) else value
    return PolicyOutput(**converted)


def _flush_window(
    window: list[Tensor], policy: nn.Module, gru: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    loss = torch.stack(window).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group["params"]], max_norm=1.0
    )
    optimizer.step()
    window.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_START)
    parser.add_argument("--relations", action="store_true", help="関係性 [C] を有効化")
    parser.add_argument("--council", action="store_true", help="会議 [D] を有効化")
    parser.add_argument(
        "--no-mood", action="store_true", help="感情 [A] を無効化 (アブレーション計測用)"
    )
    parser.add_argument(
        "--out-name", default="mood_distill", help="チェックポイントのファイル名 (拡張子なし)"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--games-per-batch", type=int, default=16)
    parser.add_argument("--tbptt", type=int, default=16, help="逆伝播を切る手数")
    parser.add_argument("--lr", type=float, default=2e-4, help="ウォームスタート前提の控えめな値")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-games", type=int, default=None, help="使う対局数の上限")
    parser.add_argument("--val-games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="bfloat16 autocast で学習する (VRAM節約と高速化。valは常にfp32)",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config.seed)
    device = resolve_device(args.device)

    dataset = SequenceDataset(find_shards(args.shard_dir), max_games=args.max_games)
    if len(dataset) <= args.val_games:
        raise SystemExit(f"対局数が足りません: {len(dataset)} 局 (val {args.val_games} 局)")
    val_set = torch.utils.data.Subset(dataset, range(args.val_games))
    train_set = torch.utils.data.Subset(dataset, range(args.val_games, len(dataset)))

    loader_args = dict(
        batch_size=args.games_per_batch,
        collate_fn=collate_sequences,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    policy = build_mood_policy(
        config, args.warm_start, device,
        mood=not args.no_mood, relations=args.relations, council=args.council,
    )
    gru = MoodGRU(config.model).to(device)
    projection = MoodProjection(config.model).to(device)  # 未学習のまま同梱 (後で回帰)
    # ウォームスタート元に学習済みGRU/射影があれば引き継ぐ
    # (イベント特徴の次元が変わった場合は形が合わないので新規初期化)
    if args.warm_start is not None and args.warm_start.exists():
        warm = torch.load(args.warm_start, map_location=device, weights_only=True)
        for module, key in ((gru, "mood_gru"), (projection, "mood_projection")):
            if key in warm:
                try:
                    module.load_state_dict(warm[key])
                except RuntimeError:
                    print(f"{key}: 形が合わないため新規初期化 (イベント特徴の変更)")
    optimizer = torch.optim.AdamW(
        [*policy.parameters(), *gru.parameters()], lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"train {len(train_set)}局 / val {len(val_set)}局 / device {device} / "
        f"warm start {args.warm_start.name}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            policy,
            gru,
            train_loader,
            config,
            device,
            tbptt=args.tbptt,
            optimizer=optimizer,
            amp=args.amp,
        )
        # val は数値の比較可能性を保つため常に fp32 (32局なので速度影響は無視できる)
        val_metrics = run_epoch(policy, gru, val_loader, config, device, tbptt=args.tbptt)
        elapsed = time.perf_counter() - started
        print(f"epoch {epoch}: train {train_metrics.format()}")
        print(f"          val   {val_metrics.format()}  ({elapsed:.1f}秒)")
        history.append(
            {"epoch": epoch, "train": asdict(train_metrics), "val": asdict(val_metrics)}
        )

        torch.save(
            {
                "model": policy.state_dict(),
                "mood_gru": gru.state_dict(),
                "mood_projection": projection.state_dict(),
                "warm_start": str(args.warm_start),
                "features": {
                    "mood": not args.no_mood,
                    "relations": args.relations,
                    "council": args.council,
                },
            },
            args.out_dir / f"{args.out_name}.pt",
        )

    (args.out_dir / f"{args.out_name}_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()


__all__ = ["build_mood_policy", "run_epoch"]
