#!/bin/bash
# チェーンの工程遷移・キャリア目標・リーグ節目 (10世代ごと)・異常だけを流す
cd /home/toyod/projects/kokoro-shogi
tail -n +1 -F checkpoints/f_chain.log | grep --line-buffered -E "^\[(BL|E[0-9][A-Za-z-]*|chain)\]|^gen ([0-9]*0|1|50):|^iter +([0-9]*00|1):|^culture pool|ゲート|Traceback|Error|FAILED|OOM" | while read -r l; do echo "${l:0:180}"; done
