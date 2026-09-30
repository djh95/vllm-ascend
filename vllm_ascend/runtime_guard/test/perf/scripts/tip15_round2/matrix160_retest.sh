#!/usr/bin/env bash
# Alternating t1/t3 retest for stage-3 suspects on 13.160 (q25_7b PP2xTP2 and
# TP4), replacing the block-ordered matrix baseline that drifted. Canonical
# protocol fix recorded in RUN_NOTES stage-4 entry: always alternate states.
export CARDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
export PY=/usr/local/python3.12.13/bin/python
export V030=/home/d00824595/vllm030_pkgs
export PRODUCT=/home/d00824595/workspace/20260908/runtime-guard/vllm-ascend
export ROOT=/home/d00824595/rg_matrix_160_retest
export PORT=8252
export HBM_TOTAL=65536
W="MODEL=/mnt/weight/Qwen2.5-7B-Instruct"
for i in 1 2; do
  env $W NAME=q25_7b_pp2 TP=2 PP=2 DP=1 STATE=t1 bash /home/d00824595/run_tip15_matrix.sh
  env $W NAME=q25_7b_pp2 TP=2 PP=2 DP=1 STATE=t3 bash /home/d00824595/run_tip15_matrix.sh
  env $W NAME=q25_7b_tp4 TP=4 PP=1 DP=1 STATE=t1 bash /home/d00824595/run_tip15_matrix.sh
  env $W NAME=q25_7b_tp4 TP=4 PP=1 DP=1 STATE=t3 bash /home/d00824595/run_tip15_matrix.sh
done
echo RETEST_DONE
