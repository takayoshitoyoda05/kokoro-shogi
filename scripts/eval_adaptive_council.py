"""会議ラウンドを局面ごとに打ち切る「適応的会議」の評価 (DESIGN.md §9 の対策の実装)。

固定 R の比較 (`eval_council_rounds.py`) では R=2 が最良で R=4 は劣化した。だが
「何ラウンド必要か」は局面によって違うはずで、DESIGN.md §9 は *bid分散が大きい局面のみ
R増* を対策として挙げている。ここではその停止則をオフラインで評価する。

推論本体は変えない。1局面につき R=0..R_max をすべて回して

- 各 R での一致 (argmax == 指し手)
- 停止判断に使える統計 (方策のマージン、bid の分散・首位差、argmax の変化)

を記録し、あとから閾値を掃引して「一致率 vs 平均ラウンド数」の曲線を作る。
これは test-time compute の配分問題そのもので、曲線が固定 R の点を上回れば
「難所だけ長く考える」ことに意味があると言える。

使い方::

    uv run python scripts/eval_adaptive_council.py --checkpoint checkpoints/phase34.pt --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kokoro_shogi.config import REPO_ROOT, load_config
from kokoro_shogi.data.dataset import find_shards
from kokoro_shogi.data.sequence import SequenceDataset, collate_sequences
from kokoro_shogi.model.mood import MoodGRU
from kokoro_shogi.model.relations import initial_relations, update_relations
from kokoro_shogi.train.distill import move_batch, resolve_device
from kokoro_shogi.train.mood_distill import _STEP_KEYS, build_mood_policy
from kokoro_shogi.train.selfplay_ppo import MAX_PIECES

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "phase34.pt"

#: 停止則が使う統計の名前 (R>=1 でのみ定義されるものを含む)
STATS = ("margin", "bid_std", "bid_gap", "changed")


def bid_stats(bid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """top-k の bid `(B, k)` から (標準偏差, 首位差) を返す。

    合法手が k 個に満たない局面では埋め合わせに $-10^9$ (非合法手のマスク値) が
    入るので、それを外して集計する。有効な提案が 1 つしか無ければ「揉める余地なし」
    として std=0 / gap=inf を返す。
    """
    valid = bid > -1e8
    count = valid.sum(dim=-1)
    safe = torch.where(valid, bid, torch.zeros_like(bid))
    mean = safe.sum(dim=-1) / count.clamp(min=1)
    deviation = torch.where(valid, (bid - mean[:, None]) ** 2, torch.zeros_like(bid))
    std = (deviation.sum(dim=-1) / count.clamp(min=1)).sqrt()

    ordered = torch.where(valid, bid, torch.full_like(bid, -float("inf")))
    ordered = ordered.sort(dim=-1, descending=True).values
    gap = ordered[:, 0] - ordered[:, 1]
    return std, torch.where(count >= 2, gap, torch.full_like(gap, float("inf")))


def collect(
    policy,
    gru,
    loader,
    device: torch.device,
    max_rounds: int,
) -> dict[str, np.ndarray]:
    """全局面について R=0..max_rounds の一致と停止統計を集める。

    戻り値の配列はすべて `(局面数, max_rounds + 1)`。`changed[:, 0]` は常に True
    (0ラウンド目は「前ラウンド」が無いので、まだ収束していない扱い)。
    """
    policy.eval()
    gru.eval()
    rounds_list = list(range(max_rounds + 1))
    columns: dict[str, list[np.ndarray]] = {"correct": [], **{name: [] for name in STATS}}

    with torch.no_grad():
        for batch in loader:
            steps_np = batch["steps"].numpy().astype(bool)
            t_end = int(steps_np.any(axis=0).sum())
            all_active = steps_np.all(axis=0)
            batch = move_batch(batch, device)
            games = steps_np.shape[0]

            mood = gru.initial_state(games, MAX_PIECES, device=device)
            relation = (
                initial_relations(games, MAX_PIECES, device=device)
                if policy.features.relations
                else None
            )

            for t in range(t_end):
                mood = gru(batch["events"][:, t], mood)
                if relation is not None:
                    relation = update_relations(relation, batch["effect"][:, t])

                if all_active[t]:
                    step = {key: batch[key][:, t] for key in _STEP_KEYS}
                    step_mood, step_relation = mood, relation
                else:
                    rows = torch.as_tensor(np.flatnonzero(steps_np[:, t]), device=device)
                    step = {key: batch[key][rows, t] for key in _STEP_KEYS}
                    step_mood = mood[rows]
                    step_relation = None if relation is None else relation[rows]
                step["mood"] = step_mood
                if step_relation is not None:
                    step["relation"] = step_relation

                action = step["action"]
                per_round = {name: [] for name in ("correct", *STATS)}
                previous_best: torch.Tensor | None = None
                for rounds in rounds_list:
                    output = policy.forward_batch(step, rounds=rounds)
                    log_probs = torch.log_softmax(output.logits, dim=-1)
                    top2 = log_probs.topk(2, dim=-1)
                    best = top2.indices[:, 0]

                    per_round["correct"].append(best == action)
                    per_round["margin"].append(top2.values[:, 0] - top2.values[:, 1])
                    per_round["changed"].append(
                        torch.ones_like(best, dtype=torch.bool)
                        if previous_best is None
                        else best != previous_best
                    )
                    if output.council:
                        std, gap = bid_stats(output.council[-1].bid.float())
                        per_round["bid_std"].append(std)
                        per_round["bid_gap"].append(gap)
                    else:  # R=0 は提案が無い。停止しない側 (=大きい値) に倒す
                        huge = torch.full_like(top2.values[:, 0], float("inf"))
                        per_round["bid_std"].append(huge)
                        per_round["bid_gap"].append(huge)
                    previous_best = best

                for name, values in per_round.items():
                    columns[name].append(torch.stack(values, dim=1).cpu().numpy())

    return {name: np.concatenate(chunks, axis=0) for name, chunks in columns.items()}


def simulate(correct: np.ndarray, stop: np.ndarray, min_rounds: int) -> tuple[float, float]:
    """停止則 `stop` (局面×ラウンドの bool) を適用したときの (一致率, 平均ラウンド数)。

    `stop[:, r]` が True の最小の `r >= min_rounds` で打ち切る。どこも True に
    ならなければ最終ラウンドまで回す。
    """
    positions, columns = correct.shape
    allowed = np.zeros_like(stop, dtype=bool)
    allowed[:, min_rounds:] = stop[:, min_rounds:]
    allowed[:, -1] = True  # 最終ラウンドは必ず停止
    chosen = allowed.argmax(axis=1)
    return float(correct[np.arange(positions), chosen].mean()), float(chosen.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shard-dir", type=Path, default=REPO_ROOT / "data" / "shards" / "2024")
    parser.add_argument("--val-games", type=int, default=64, help="学習時と同じ先頭N局")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
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
        dataset, batch_size=args.batch_size, collate_fn=collate_sequences,
        num_workers=args.workers,
    )

    print(f"{args.checkpoint.name} / {args.val_games}局 / device: {device}")
    data = collect(policy, gru, loader, device, args.max_rounds)
    correct = data["correct"]
    positions = correct.shape[0]

    fixed = [
        {"rule": "fixed", "threshold": None, "rounds": r,
         "accuracy": float(correct[:, r].mean()), "mean_rounds": float(r)}
        for r in range(args.max_rounds + 1)
    ]
    print(f"\n局面数 {positions}")
    print("固定ラウンド:")
    for entry in fixed:
        print(f"  R={entry['rounds']}  一致率 {entry['accuracy'] * 100:.2f}%")

    results = list(fixed)

    # ラウンドごとに argmax がどれだけ揺れるか。停止則が効かない理由の診断になる
    # (「一度収束しても次のラウンドで答えが変わる」なら早期打ち切りは成立しない)。
    changed_rate = [float(data["changed"][:, r].mean()) for r in range(1, args.max_rounds + 1)]
    print("\nargmax が前ラウンドから変わった割合:")
    for r, rate in enumerate(changed_rate, start=1):
        print(f"  R={r - 1}→{r}  {rate * 100:.1f}%")

    # 上限: 局面ごとに最良のラウンドを選べたとしたら何%か (完璧な停止則の天井)。
    oracle = float(correct.max(axis=1).mean())
    oracle_rounds = float(correct.argmax(axis=1).mean())
    capped = correct[:, : min(3, args.max_rounds + 1)]
    print(f"\nオラクル (局面ごとに最良Rを選べた場合): 一致率 {oracle * 100:.2f}%  "
          f"平均 {oracle_rounds:.2f} ラウンド")
    print(f"オラクル (R<=2 に限定):                一致率 {capped.max(axis=1).mean() * 100:.2f}%")
    results.append({"rule": "oracle", "threshold": None, "rounds": None,
                    "accuracy": oracle, "mean_rounds": oracle_rounds})
    results.append({"rule": "oracle_r2", "threshold": None, "rounds": None,
                    "accuracy": float(capped.max(axis=1).mean()),
                    "mean_rounds": float(capped.argmax(axis=1).mean())})

    # 「argmax が前ラウンドから変わらなくなったら止める」— 閾値なしの自然な停止則
    accuracy, mean_rounds = simulate(correct, ~data["changed"], min_rounds=1)
    results.append({"rule": "stable", "threshold": None, "rounds": None,
                    "accuracy": accuracy, "mean_rounds": mean_rounds})
    print(f"\n収束判定 (argmax が変わらなくなったら停止): "
          f"一致率 {accuracy * 100:.2f}%  平均 {mean_rounds:.2f} ラウンド")

    # 統計が閾値を下回ったら「もう揉める余地なし」として停止する
    for name, direction in (("margin", "above"), ("bid_std", "below"), ("bid_gap", "above")):
        values = data[name]
        finite = values[np.isfinite(values)]
        grid = np.quantile(finite, np.linspace(0.05, 0.95, 19))
        print(f"\n{name} で停止 ({'>=' if direction == 'above' else '<='} 閾値):")
        for threshold in grid:
            stop = values >= threshold if direction == "above" else values <= threshold
            accuracy, mean_rounds = simulate(correct, stop, min_rounds=1)
            results.append({"rule": name, "threshold": float(threshold), "rounds": None,
                            "accuracy": accuracy, "mean_rounds": mean_rounds})
            print(f"  閾値 {threshold:>9.3f}  一致率 {accuracy * 100:.2f}%  "
                  f"平均 {mean_rounds:.2f} ラウンド")

    out = args.out or args.checkpoint.with_name("adaptive_council_eval.json")
    out.write_text(
        json.dumps(
            {"positions": positions, "changed_rate": changed_rate, "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{out.name} に保存しました。")


if __name__ == "__main__":
    main()
