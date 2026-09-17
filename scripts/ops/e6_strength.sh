#!/bin/bash
# 9/11 承認: E6 (交叉) と E2b (単親) の最終文化を ppo2 と対戦させる。
# 既存 league_vs_base.json と同条件 (60局/文化, tau 0.1, max-plies 200) で並べられるようにする。
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
echo "[E6-strength] START $(date '+%m-%d %H:%M')"
uv run --no-sync python scripts/eval_league_vs_base.py \
  --base checkpoints/ppo2.pt \
  --leagues checkpoints/league_E6_crossover checkpoints/league_E2b_grace \
  --games 60 --tau 0.1 --max-plies 200 \
  --out checkpoints/league_vs_base_e6_e2b.json
rc=$?
if [ $rc -ne 0 ]; then echo "[E6-strength] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
echo "[E6-strength] DONE $(date '+%m-%d %H:%M')"
