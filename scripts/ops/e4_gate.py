"""E4 自動ゲート: 100 iter 時点の履歴を判定する。exit 0 = 続行, 1 = 打ち切り, 2 = まだ。

判定 (すべて満たせば続行):
  - 100 iter ぶんの履歴があり、policy/value/anchor に NaN/inf がない
  - iter 100 の対開始時勝率 >= 0.40 (40 局、SE 0.08。崩壊していないこと)
  - GRU 漂流 Δgru / ||gru_0|| < 0.10
  - 直近 10 iter のエントロピー平均 > 0.50
"""
import json
import math
import sys
from pathlib import Path

import torch

hist_path = Path(sys.argv[1])
at = int(sys.argv[2]) if len(sys.argv) > 2 else 100
if not hist_path.exists():
    sys.exit(2)
hist = json.loads(hist_path.read_text())
if len(hist) < at:
    sys.exit(2)
rows = hist[:at]
state = torch.load("checkpoints/ppo2.pt", map_location="cpu", weights_only=True)
gru_norm = math.sqrt(sum(float((v.float() ** 2).sum()) for v in state["mood_gru"].values()))

bad = [r["iteration"] for r in rows
       if any(not math.isfinite(r.get(k, 0.0)) for k in ("policy", "value", "anchor", "gru_delta"))]
last = rows[-1]
wr = last.get("win_rate_vs_start")
delta_rel = last["gru_delta"] / gru_norm
h10 = sum(r["entropy"] for r in rows[-10:]) / 10
anchor10 = sum(r["anchor"] for r in rows[-10:]) / 10
checks = {
    "finite": not bad,
    "win_rate>=0.40": wr is not None and wr >= 0.40,
    "gru_delta_rel<0.10": delta_rel < 0.10,
    "entropy10>0.50": h10 > 0.50,
}
verdict = "PASS" if all(checks.values()) else "FAIL"
print(
    f"[E4-gate] {verdict} @iter{at}: WR {wr} / Δgru {last['gru_delta']:.3f} "
    f"({delta_rel:.3%} of ||gru0||={gru_norm:.2f}) / H10 {h10:.3f} / anchor10 {anchor10:.5f} / "
    f"non-finite {bad[:5]} / checks {checks}"
)
sys.exit(0 if verdict == "PASS" else 1)
