"""対戦結果を Bradley-Terry で 1 本の Elo 尺度にまとめる (2026-09-21)。

動機: 単一の相手 (Háo depth 1) に対する勝率では、全モデルが 0.02〜0.23 に潰れていて
400 局でも 0.045〜0.06 の差しか見えない (シード雑音と同水準)。強さの違う相手を複数用意し、
全対戦を同時に説明する Elo を最尤推定すれば、1 局あたりの情報量が増えて信頼区間が縮む。

    P(i が j に勝つ) = 1 / (1 + 10^((R_j - R_i) / 400))

外部エンジン Háo depth 1 を R = 0 に固定して尺度を留める (これがないと全体が平行移動する)。
引き分けは 0.5 勝として数える。信頼区間は Fisher 情報行列の対角から出す。

    uv run python scripts/ops/fit_elo.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCALE = 400.0 / math.log(10.0)  # Elo 1 点あたりの自然対数尺度


def node_name(data: dict, fallback: str) -> str:
    """ノード名 = チェックポイント名 + 既定から外れた推論設定。

    同じ .pt でも τ (温度) や R (会議ラウンド) や文化が違えば別のプレイヤーとして扱う。
    yardstick の json の stem は評価の別名 (e4c_iter100 = ppo_e4c_gru.pt) なので stem では
    識別せず、記録された checkpoint と設定から組み立てる。
    """
    name = Path(data.get("checkpoint", fallback)).stem
    if data.get("tau"):
        name += f"@tau{data['tau']}"
    if data.get("council_rounds"):
        name += f"@R{data['council_rounds']}"
    if data.get("culture") and data["culture"] not in ("-", None, "culture1"):
        name += f"@{data['culture']}"
    return name


def collect() -> tuple[list[str], dict[tuple[int, int], tuple[float, float]]]:
    """対戦表を集める。戻り値は (名前の一覧, {(i,j): (i の勝ち数, 対局数)})。"""
    names: list[str] = ["Hao_d1"]
    index: dict[str, int] = {"Hao_d1": 0}

    def node(name: str) -> int:
        if name not in index:
            index[name] = len(names)
            names.append(name)
        return index[name]

    table: dict[tuple[int, int], list[float]] = {}

    def add(a: int, b: int, wins: float, games: int) -> None:
        key = (a, b) if a < b else (b, a)
        entry = table.setdefault(key, [0.0, 0])
        entry[0] += wins if a < b else games - wins
        entry[1] += games

    # 対エンジン (Háo depth 1)。yardstick400 を優先し、無ければ yardstick_open
    seen: set[str] = set()
    for directory in ("yardstick400", "yardstick_open"):
        for path in sorted((REPO / "checkpoints" / directory).glob("*.json")):
            data = json.loads(path.read_text())
            if data.get("engine_limit") != "depth 1" or path.stem in seen:
                continue
            seen.add(path.stem)
            # ノード名はチェックポイントのファイル名で統一する
            # (json の stem は評価の別名のことがある。例: yardstick の e4c_iter100.json は
            # ppo_e4c_gru.pt を測ったもの。別名のままだと
            # 同じモデルが 2 ノードに割れ、片方は対 Háo だけ・片方は総当たりだけになって
            # 推定が壊れる)
            model = node_name(data, path.stem)
            add(node(model), 0, data["win_rate"] * data["games"], data["games"])

    # モデル同士 (ラダー)。ファイル名は <挑戦者>__vs__<相手>.json
    for directory in ("ladder", "anchor400", "anchor"):
        folder = REPO / "checkpoints" / directory
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text())
            if not data.get("anchor"):
                continue
            challenger = node_name(data, Path(data["checkpoint"]).stem)
            # アンカー側は既定設定 (argmax / R 既定) で走らせている
            opponent = Path(data["anchor"]).stem
            add(node(challenger), node(opponent), data["win_rate"] * data["games"], data["games"])

    return names, {k: (v[0], v[1]) for k, v in table.items()}


def fit(
    names: list[str], table: dict[tuple[int, int], tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """勾配法で対数尤度を最大化する。Háo (index 0) は 0 に固定。"""
    size = len(names)
    rating = np.zeros(size)
    for _ in range(4000):
        grad = np.zeros(size)
        for (i, j), (wins, games) in table.items():
            expected = games / (1.0 + math.exp((rating[j] - rating[i]) / SCALE))
            grad[i] += (wins - expected) / SCALE
            grad[j] -= (wins - expected) / SCALE
        grad[0] = 0.0  # 基準を固定
        rating += 8.0 * grad
    # Fisher 情報の対角から標準誤差
    info = np.zeros(size)
    for (i, j), (_, games) in table.items():
        p = 1.0 / (1.0 + math.exp((rating[j] - rating[i]) / SCALE))
        v = games * p * (1 - p) / SCALE**2
        info[i] += v
        info[j] += v
    stderr = np.where(info > 0, 1.0 / np.sqrt(np.maximum(info, 1e-12)), float("nan"))
    return rating, stderr


def main() -> None:
    names, table = collect()
    if len(table) < 2:
        print("対戦データが足りません。ラダーを先に走らせてください。")
        return
    rating, stderr = fit(names, table)
    print("=== Bradley-Terry Elo (Háo depth 1 = 0 に固定) ===")
    print(f"対戦ペア {len(table)} / 総局数 {sum(g for _, g in table.values()):,}\n")
    print(f"{'モデル':<24}{'Elo':>8}{'±95%':>9}   対戦相手数")
    order = sorted(range(len(names)), key=lambda k: -rating[k])
    degree = {k: 0 for k in range(len(names))}
    for (i, j), _ in table.items():
        degree[i] += 1
        degree[j] += 1
    for k in order:
        ci = 1.96 * stderr[k]
        mark = "  (基準)" if k == 0 else ""
        print(f"{names[k]:<24}{rating[k]:>8.0f}{ci:>9.0f}   {degree[k]:>3}{mark}")
    out = REPO / "checkpoints" / "elo_ladder.json"
    out.write_text(json.dumps(
        {"names": names, "elo": rating.tolist(), "stderr": stderr.tolist(),
         "pairs": {f"{names[i]}|{names[j]}": {"wins": w, "games": g}
                   for (i, j), (w, g) in table.items()}},
        ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
