#!/bin/bash
# 9/7 13:10 ユーザー指示「始めて」: 停止した F チェーンの残り = E2b (grace のみ文化リーグ) を単独で実行。
# 設定は f_chain.sh の E2b と同一。ログは f_chain.log に追記 (印: [E2b-grace-only] START/DONE/FAILED, [chain] ALL DONE)
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
echo "[E2b-grace-only] START $(date '+%m-%d %H:%M')"
mkdir -p checkpoints/league_E2b_grace
uv run python -m kokoro_shogi.train.league --checkpoint checkpoints/ppo2.pt \
  --out-dir checkpoints/league_E2b_grace --cultures 6 --generations 50 --games-per-pair 16 --max-plies 200 \
  --mutation-sigma 0.15 --grace 2 --fix-hypers
rc=$?
if [ $rc -ne 0 ]; then echo "[E2b-grace-only] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
echo "[E2b-grace-only] DONE $(date '+%m-%d %H:%M')"
echo "[chain] ALL DONE $(date '+%m-%d %H:%M')"
