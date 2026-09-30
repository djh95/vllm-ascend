#!/usr/bin/env bash
# tip15 stage-3 matrix driver on 192.168.13.160 (rg-test-160).
# Note: CARDS contains spaces -> must be exported (not folded into $C word-split env string).
export CARDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
C="PY=/usr/local/python3.12.13/bin/python V030=/home/d00824595/vllm030_pkgs PRODUCT=/home/d00824595/workspace/20260908/runtime-guard/vllm-ascend ROOT=/home/d00824595/rg_matrix_160 PORT=8250 HBM_TOTAL=65536"
W="MODEL=/mnt/weight/Qwen2.5-7B-Instruct NAME=q25_7b"
for S in t1 t3; do
  env $C $W TP=1 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C $W TP=2 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C $W TP=4 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C $W TP=2 PP=2 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C $W TP=1 PP=1 DP=2 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C MODEL=/mnt/weight/Qwen3-8B NAME=q3_8b TP=2 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C MODEL=/mnt/weight/Qwen3-8B NAME=q3_8b TP=4 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C MODEL=/mnt/weight/DeepSeek-V2-Lite NAME=dsv2_lite TP=2 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C MODEL=/mnt/weight/DeepSeek-V2-Lite NAME=dsv2_lite TP=4 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  env $C MODEL=/mnt/weight/Qwen3-30B-A3B NAME=q3c_30b TP=4 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
done
env $C MODEL=/mnt/weight/Qwen3-0.6B NAME=q3_06b TP=1 PP=1 DP=1 STATE=t1 bash /home/d00824595/run_tip15_matrix.sh
env $C MODEL=/mnt/weight/Qwen3-0.6B NAME=q3_06b TP=1 PP=1 DP=1 STATE=t3 bash /home/d00824595/run_tip15_matrix.sh
echo MATRIX160_DONE
