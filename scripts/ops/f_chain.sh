#!/bin/bash
# 9/6 承認の混合案: BL → E5 (文化プール PPO) → E5c (対照: 自己プールのみ) → E5 評価
#                   → E4 (GRU 解凍、自動ゲート付き、e4_stage.sh が無ければ SKIP) → E2b (grace のみ)
cd /home/toyod/projects/kokoro-shogi; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
S=/home/toyod/projects/kokoro-shogi/checkpoints/ops
run() { # name, cmd...
  local name=$1; shift
  echo "[$name] START $(date '+%m-%d %H:%M')"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "[$name] FAILED rc=$rc $(date '+%m-%d %H:%M')"; exit 1; fi
  echo "[$name] DONE $(date '+%m-%d %H:%M')"
}
PPO_COMMON="--checkpoint checkpoints/ppo2.pt --out-dir checkpoints --iterations 1000 --games-per-iter 64 \
  --snapshot-every 100 --pool-size 12 --eval-every 50 --eval-games 40"

run BL uv run python scripts/eval_league_vs_base.py --base checkpoints/ppo2.pt \
  --leagues checkpoints/league_run2_6x80 checkpoints/league_E1_control checkpoints/league_E2_run3 \
  --games 60 --out checkpoints/league_vs_base.json
run E5-culture uv run python -m kokoro_shogi.train.selfplay_ppo $PPO_COMMON --out-name ppo_e5_culture \
  --culture-pool checkpoints/league_run2_6x80/league.pt checkpoints/league_E1_control/league.pt \
  checkpoints/league_E2_run3/league.pt
run E5-control uv run python -m kokoro_shogi.train.selfplay_ppo $PPO_COMMON --out-name ppo_e5_control
run E5-eval uv run python $S/e5_eval.py --games 80 --out checkpoints/e5_eval.json
if [ -x $S/e4_stage.sh ]; then
  echo "[E4] START $(date '+%m-%d %H:%M')"
  bash $S/e4_stage.sh && echo "[E4] DONE $(date '+%m-%d %H:%M')" \
    || echo "[E4] FAILED (続行して E2b へ) $(date '+%m-%d %H:%M')"
else
  echo "[E4] SKIP (e4_stage.sh なし) $(date '+%m-%d %H:%M')"
fi
mkdir -p checkpoints/league_E2b_grace
run E2b-grace-only uv run python -m kokoro_shogi.train.league --checkpoint checkpoints/ppo2.pt \
  --out-dir checkpoints/league_E2b_grace --cultures 6 --generations 50 --games-per-pair 16 --max-plies 200 \
  --mutation-sigma 0.15 --grace 2 --fix-hypers
echo "[chain] ALL DONE $(date '+%m-%d %H:%M')"
