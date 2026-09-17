#!/bin/bash
# 9/4 承認の順番: E0 キャリア延長 → E1 対照リーグ → E2 改良リーグ (run3) → E3 忠誠キャリア。GPU は直列。
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
S=/home/toyod/projects/kokoro-shogi/checkpoints/ops
run() { # name, cmd...
  local name=$1; shift
  echo "[$name] START $(date '+%m-%d %H:%M')"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "[$name] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
  echo "[$name] DONE $(date '+%m-%d %H:%M')"
}
run E0-career bash $S/career_ext.sh
run E1-control uv run python -m kokoro_shogi.train.league --checkpoint checkpoints/ppo2.pt \
  --out-dir checkpoints/league_E1_control --cultures 6 --generations 50 --games-per-pair 16 --max-plies 200 \
  --mutation-sigma 0.15 --selection random --fix-hypers
run E2-run3 uv run python -m kokoro_shogi.train.league --checkpoint checkpoints/ppo2.pt \
  --out-dir checkpoints/league_E2_run3 --cultures 6 --generations 50 --games-per-pair 16 --max-plies 200 \
  --mutation-sigma 0.15 --grace 2 --newborn-ppo-iters 6 --fix-hypers
run E3-loyalty uv run python scripts/selfplay_career.py --checkpoint checkpoints/ppo2.pt --games 1000 \
  --seed 3000 --loyalty --db data/pieces_loyalty.sqlite3
echo "[chain] ALL DONE $(date '+%m-%d %H:%M')"
