#!/usr/bin/env bash
export CARDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
export PY=/usr/local/python3.12.13/bin/python
export V030=/home/d00824595/vllm030_pkgs
export PRODUCT=/home/d00824595/workspace/20260908/runtime-guard/vllm-ascend
export ROOT=/home/d00824595/rg_matrix_160_bisect
export PORT=8253
export HBM_TOTAL=65536
for i in 1 2; do
  for S in t1 t2 t3; do
    MODEL=/mnt/weight/Qwen2.5-7B-Instruct NAME=q25_7b_pp2 TP=2 PP=2 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  done
done
echo BISECT_DONE
