#!/bin/bash
# 9/11 22:00 ユーザー指示「残っているものをすべて回して」。
# 承認済み計画の残り: A (変異のみ) → E7 (適応度EMA) → 両者の強さ評価。
# どちらも E2b と同条件 (6文化×50世代, pair 16, max-plies 200, σ0.15, grace 2, fix-hypers)。
# 差分は A が --selection drift、E7 が --fitness ema だけ。
# 9/11 のコード変更により、この 2 本からは league_history.json に draw_rate が入る。
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1

COMMON="--checkpoint checkpoints/ppo2.pt --cultures 6 --generations 50 --games-per-pair 16 \
  --max-plies 200 --mutation-sigma 0.15 --grace 2 --fix-hypers"

echo "[A-drift] START $(date '+%m-%d %H:%M')"
mkdir -p checkpoints/league_A_drift
uv run --no-sync python -m kokoro_shogi.train.league $COMMON \
  --out-dir checkpoints/league_A_drift --selection drift
rc=$?
if [ $rc -ne 0 ]; then echo "[A-drift] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
echo "[A-drift] DONE $(date '+%m-%d %H:%M')"

echo "[E7-ema] START $(date '+%m-%d %H:%M')"
mkdir -p checkpoints/league_E7_ema
uv run --no-sync python -m kokoro_shogi.train.league $COMMON \
  --out-dir checkpoints/league_E7_ema --fitness ema
rc=$?
if [ $rc -ne 0 ]; then echo "[E7-ema] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
echo "[E7-ema] DONE $(date '+%m-%d %H:%M')"

echo "[AE7-strength] START $(date '+%m-%d %H:%M')"
uv run --no-sync python scripts/eval_league_vs_base.py \
  --base checkpoints/ppo2.pt \
  --leagues checkpoints/league_A_drift checkpoints/league_E7_ema \
  --games 60 --tau 0.1 --max-plies 200 \
  --out checkpoints/league_vs_base_a_e7.json
rc=$?
if [ $rc -ne 0 ]; then echo "[AE7-strength] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
echo "[AE7-strength] DONE $(date '+%m-%d %H:%M')"
echo "[chain] ALL DONE $(date '+%m-%d %H:%M')"
