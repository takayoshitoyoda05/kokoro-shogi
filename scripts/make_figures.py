"""発表・論文用の図を checkpoints/*.json から生成する (DESIGN.md §8 の計測項目)。

図はすべて計測結果の JSON から作る。手で数値を書き写さないので、実験を回し直せば
図もそのまま更新される。

使い方::

    uv sync --group viz   # matplotlib (学習には不要なので既定の同期対象外)
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from kokoro_shogi.config import REPO_ROOT  # noqa: E402

#: 検証済みカテゴリカル配色 (スロット順に使う。循環させない)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: WSL には日本語フォントが無いので Windows 側のものを借りる
FONT_CANDIDATES = (
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothM.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def setup_font() -> bool:
    """日本語フォントを登録する。見つからなければ False (呼び出し側で英語に落とす)。"""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            return True
    return False


def new_axes(width: float = 7.2, height: float = 4.2):
    figure, axes = plt.subplots(figsize=(width, height), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=MUTED, labelsize=9, length=0)
    return figure, axes


def finish(figure, axes, title: str, subtitle: str, out: Path) -> None:
    axes.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=18)
    axes.text(
        0.0, 1.03, subtitle, transform=axes.transAxes,
        color=INK_SECONDARY, fontsize=9.5, va="bottom",
    )
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"  {out.relative_to(REPO_ROOT)}")


def spread_labels(values: list[float], *, gap: float) -> list[float]:
    """昇順の値を、隣り合うラベルが `gap` 以上離れるように押し上げる。"""
    placed: list[float] = []
    for value in values:
        placed.append(value if not placed else max(value, placed[-1] + gap))
    return placed


def figure_lambda_tradeoff(checkpoints: Path, out_dir: Path) -> None:
    """説明率を上げても一致率がほとんど落ちないこと (本研究の主結果)。"""
    points = json.loads((checkpoints / "lambda_g_sweep.json").read_text())["points"]
    x = [p["explained_ratio"] * 100 for p in points]
    y = [p["accuracy"] * 100 for p in points]

    figure, axes = new_axes()
    axes.plot(x, y, color=SERIES[0], linewidth=2, marker="o", markersize=9,
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    for point, px, py in zip(points, x, y, strict=True):
        axes.annotate(
            f"λ$_g$={point['lambda_g']:g}", (px, py), textcoords="offset points",
            xytext=(0, 13), ha="center", fontsize=9.5, color=INK,
        )
    axes.annotate(
        f"この間の低下は {y[0] - y[-1]:.2f} ポイントだけ",
        (x[-1], y[-1]), textcoords="offset points", xytext=(-6, -30), ha="right",
        fontsize=10, color=SERIES[1], fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": SERIES[1], "linewidth": 1.2},
    )
    axes.set_xlabel("欲求による手スコアの説明率 (%)", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("指し手一致率 (%)", color=INK_SECONDARY, fontsize=10)
    axes.set_xlim(0, 108)
    axes.set_ylim(0, 42)  # 一致率は0が意味を持つ。切り詰めると小さな差を誇張してしまう
    finish(
        figure, axes,
        "解釈性はほぼ無料で手に入る",
        f"説明率 {x[0]:.0f}% → {x[-1]:.1f}% に対し、一致率の低下は {y[0] - y[-1]:.2f} ポイント",
        out_dir / "lambda_g_tradeoff.png",
    )


def figure_council_rounds(checkpoints: Path, out_dir: Path) -> None:
    """会議は2ラウンドが最良。そして局面ごとに選べれば大きな余地がある。"""
    data = json.loads((checkpoints / "adaptive_council_eval.json").read_text())
    fixed = [r for r in data["results"] if r["rule"] == "fixed"]
    oracle = next(r for r in data["results"] if r["rule"] == "oracle")
    rounds = [r["rounds"] for r in fixed]
    accuracy = [r["accuracy"] * 100 for r in fixed]
    best = max(range(len(accuracy)), key=accuracy.__getitem__)

    figure, axes = new_axes()
    colors = [SERIES[0] if i == best else "#b9d2f0" for i in range(len(rounds))]
    axes.bar(rounds, accuracy, color=colors, width=0.62, zorder=3)
    for r, value in zip(rounds, accuracy, strict=True):
        axes.text(r, value + 0.9, f"{value:.1f}%", ha="center", fontsize=9.5, color=INK)

    ceiling = oracle["accuracy"] * 100
    axes.axhline(ceiling, color=SERIES[1], linewidth=2, linestyle=(0, (5, 3)), zorder=4)
    axes.text(
        len(rounds) - 0.6, ceiling + 1.0,
        f"局面ごとに最良のラウンドを選べた場合  {ceiling:.1f}%",
        ha="right", fontsize=9.5, color=SERIES[1], fontweight="bold",
    )
    axes.set_xlabel("会議のラウンド数", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("指し手一致率 (%)", color=INK_SECONDARY, fontsize=10)
    axes.set_xticks(rounds)
    axes.set_ylim(0, ceiling + 7)
    finish(
        figure, axes,
        "議論は2ラウンドで頭打ち、しかし配分には余地がある",
        f"phase37 / {data['positions']:,}局面。3ラウンド以降は劣化するが、"
        f"上限とは {ceiling - max(accuracy):.1f} ポイントの差がある",
        out_dir / "council_rounds.png",
    )


def figure_ladder(checkpoints: Path, out_dir: Path) -> None:
    """自己対戦が単調に強くしたこと、そして頭打ちに達したこと。"""
    ladder = json.loads((checkpoints / "ladder_eval.json").read_text())
    labels = [row["challenger"].replace(".pt", "") for row in ladder]
    y = [row["win_rate"] * 100 for row in ladder]
    x = list(range(len(y)))

    figure, axes = new_axes()
    axes.axhline(50, color=AXIS, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    axes.text(len(x) - 0.1, 51.5, "互角", ha="right", fontsize=9, color=MUTED)
    axes.plot(x, y, color=SERIES[0], linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    axes.annotate(
        f"最後だけ勝ち越せず {y[-1]:.1f}%\n= 頭打ち", (x[-1], y[-1]),
        textcoords="offset points", xytext=(-16, -34), ha="right",
        fontsize=9.5, color=SERIES[1], fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": SERIES[1], "linewidth": 1.2},
    )
    axes.set_xticks(x)
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes.set_xlabel("PPO 世代 (挑戦者)", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("直前の世代に対する勝率 (%)", color=INK_SECONDARY, fontsize=10)
    axes.set_ylim(0, 100)
    finish(
        figure, axes,
        "各世代が前の世代に勝ち越し続け、最後に止まった",
        "各点は「その世代 vs 直前の世代」の勝率 (40〜80局)。50% を超えている間は改善している",
        out_dir / "selfplay_ladder.png",
    )


def figure_league_distance(checkpoints: Path, out_dir: Path) -> None:
    """淘汰の方式を変えても文化の分化は大きく変わらない (ネガティブ結果)。"""
    runs = (
        ("E1: 乱択淘汰", "league_E1_control"),
        ("E2: 適応度淘汰", "league_E2_run3"),
        ("E2b: +猶予", "league_E2b_grace"),
        ("E6: +ES交叉", "league_E6_crossover"),
    )
    window = 5

    figure, axes = new_axes(width=7.8)
    ends = []
    for slot, (label, run) in enumerate(runs):
        path = checkpoints / run / "league_history.json"
        if not path.exists():
            continue
        values = [g["phi_distance"]["mean"] for g in json.loads(path.read_text())]
        smoothed = [
            sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1])
            for i in range(len(values))
        ]
        generations = list(range(1, len(values) + 1))
        axes.plot(generations, smoothed, color=SERIES[slot], linewidth=2, zorder=3)
        ends.append([smoothed[-1], generations[-1], label, SERIES[slot]])

    for end, y_position in zip(
        sorted(ends), spread_labels([e[0] for e in sorted(ends)], gap=0.24), strict=True
    ):
        value, generation, label, color = end
        axes.annotate(
            label, (generation, value), xytext=(generation + 2.5, y_position),
            fontsize=9.5, color=color, fontweight="bold", va="center",
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8},
        )
    axes.set_xlabel("世代", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("文化間の性格距離  ‖Φ−Φ′‖ の平均", color=INK_SECONDARY, fontsize=10)
    axes.set_xlim(1, 66)
    axes.set_xticks([1, 10, 20, 30, 40, 50])
    axes.set_ylim(0, 4.0)  # 距離は0が意味を持つ (0 = 全文化が同一)
    finish(
        figure, axes,
        "どの淘汰方式でも分化は同じ帯に収まる",
        f"6文化×50世代、{window}世代の移動平均。50世代を通じて 2〜3 の範囲を出ず、"
        "ES交叉 (E6) だけが一貫して低い",
        out_dir / "league_distance.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    args = parser.parse_args()

    if not setup_font():
        print("警告: 日本語フォントが見つからないので文字化けします")

    print("生成した図:")
    figure_lambda_tradeoff(args.checkpoints, args.out_dir)
    figure_council_rounds(args.checkpoints, args.out_dir)
    figure_ladder(args.checkpoints, args.out_dir)
    figure_league_distance(args.checkpoints, args.out_dir)


if __name__ == "__main__":
    main()
