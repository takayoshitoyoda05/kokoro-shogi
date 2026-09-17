#!/bin/bash
# 120秒ごと: GPU温度/メモリ, RAM, ディスク, チェーン生存, 新規エラー行。1hごとに HEALTH-OK 心拍。
cd /home/toyod/projects/kokoro-shogi
LOG=checkpoints/f_chain.log; hot=0; lasthb=0; seen=$(grep -cE "Traceback|OOM|CUDA|RuntimeError|FAILED" $LOG)
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  ram=$(free -g | awk 'NR==2{print $7}'); disk=$(df -BG / | awk 'NR==2{print $4}' | tr -d G)
  if [ "${t:-0}" -ge 87 ]; then echo "HEALTH-CRITICAL GPU ${t}℃"; fi
  if [ "${t:-0}" -ge 80 ]; then hot=$((hot+1)); [ $hot -ge 2 ] && echo "HEALTH-ALERT GPU ${t}℃ が2回連続"; else hot=0; fi
  [ "${m:-0}" -ge 8100 ] && echo "HEALTH-WARN GPUメモリ ${m}MiB"
  [ "${ram:-99}" -lt 2 ] && echo "HEALTH-ALERT RAM残 ${ram}G"
  [ "${disk:-999}" -lt 20 ] && echo "HEALTH-ALERT ディスク残 ${disk}G"
  if ! grep -qE "\[chain\] ALL DONE|FAILED rc=" $LOG && ! pgrep -f "ops/(f_chain|e2b_stage).sh" >/dev/null; then echo "HEALTH-ALERT f_chain が終了印なしに消滅"; exit 1; fi
  n=$(grep -cE "Traceback|OOM|CUDA|RuntimeError|FAILED" $LOG); if [ "$n" -gt "$seen" ]; then grep -E "Traceback|OOM|CUDA|RuntimeError|FAILED" $LOG | tail -n $((n-seen)) | sed 's/^/HEALTH-ERROR /'; seen=$n; fi
  now=$(date +%s); if [ $((now-lasthb)) -ge 3600 ]; then stage=$(grep -oE "^\[[A-Za-z0-9-]+\] START" $LOG | tail -1 | cut -d' ' -f1); echo "HEALTH-OK $(date '+%H:%M') GPU ${t}℃ ${m}MiB RAM残${ram}G | $stage"; lasthb=$now; fi
  if grep -qE "\[chain\] ALL DONE|FAILED rc=" $LOG; then echo "HEALTH-END チェーン終了"; exit 0; fi
  sleep 120
done
