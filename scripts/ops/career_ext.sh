#!/bin/bash
# キャリア延長: 100局刻みで 2,000 局までスナップ (9/4 決定)。完了済み目標はスキップ。
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
mkdir -p data/career_snap
for target in $(seq 600 100 2000); do
  have=$(uv run python -c "import sqlite3;print(sqlite3.connect('data/pieces.sqlite3').execute('select max(games) from pieces').fetchone()[0])" 2>/dev/null)
  n=$((target - have)); [ "$n" -le 0 ] && continue
  echo "[career-ext] $have -> $target ($n 局) $(date '+%H:%M')"
  uv run python scripts/selfplay_career.py --checkpoint checkpoints/ppo2.pt --games $n --seed $((1000 + target)) 2>&1 | grep -vi warning | grep -E "更新|Traceback|Error"
  test ${PIPESTATUS[0]} -eq 0 || { echo "[career-ext] FAILED at $target"; exit 1; }
  cp data/pieces.sqlite3 data/career_snap/pieces_$(printf %04d $target).sqlite3
done
echo "[career-ext] DONE $(date '+%H:%M')"
