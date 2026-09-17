#!/bin/bash
# E4: GRU 解凍 PPO (truncated BPTT 8 手 + anchor 1.0) を 1,200 iter。
# iter 100 で e4_gate.py が判定し、FAIL なら PPO を止めて exit 1 (f_chain は E2b へ続行)。
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
S=/home/toyod/projects/kokoro-shogi/checkpoints/ops
HIST=checkpoints/ppo_e4_gru_history.json
rm -f $HIST
uv run python -m kokoro_shogi.train.selfplay_ppo --checkpoint checkpoints/ppo2.pt --out-dir checkpoints \
  --out-name ppo_e4_gru --iterations 1200 --games-per-iter 64 --snapshot-every 100 --pool-size 12 \
  --eval-every 50 --eval-games 40 --train-gru --bptt 8 --anchor 1.0 &
PID=$!
while true; do
  sleep 120
  if ! kill -0 $PID 2>/dev/null; then wait $PID; rc=$?; echo "[E4] PPO がゲート前に終了 rc=$rc"; exit $rc; fi
  uv run python $S/e4_gate.py $HIST 100; g=$?
  if [ $g -eq 0 ]; then echo "[E4] ゲート通過、本番続行 $(date '+%m-%d %H:%M')"; break; fi
  if [ $g -eq 1 ]; then echo "[E4] ゲート不通過、PPO 停止 $(date '+%m-%d %H:%M')"; kill $PID; wait $PID 2>/dev/null; exit 1; fi
done
wait $PID; rc=$?
[ $rc -eq 0 ] && uv run python $S/e4_gate.py $HIST 1200
exit $rc
