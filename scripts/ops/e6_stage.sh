#!/bin/bash
# 9/11 ユーザー承認の実験計画「フル 6x50 / 夜を分けて 1 晩 1 本 / uniform のみ」。
# E6 = E2b と完全同条件 + ES 交叉 (uniform)。E2b との差分は --crossover uniform だけ。
# ログは checkpoints/g_chain.log (印: [E6-crossover] START/DONE/FAILED)
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
echo "[E6-crossover] START $(date '+%m-%d %H:%M')"
mkdir -p checkpoints/league_E6_crossover
uv run --no-sync python -m kokoro_shogi.train.league --checkpoint checkpoints/ppo2.pt \
  --out-dir checkpoints/league_E6_crossover --cultures 6 --generations 50 --games-per-pair 16 --max-plies 200 \
  --mutation-sigma 0.15 --grace 2 --fix-hypers --crossover uniform
rc=$?
if [ $rc -ne 0 ]; then echo "[E6-crossover] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
echo "[E6-crossover] DONE $(date '+%m-%d %H:%M')"
