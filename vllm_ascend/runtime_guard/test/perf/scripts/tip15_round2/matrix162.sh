#!/usr/bin/env bash
# Stage-3 perf matrix driver on 192.168.13.162 (container rg-tip11-162)
# 8 runs: dsv2_lite TP2 / q25_7b TP2 / q25_7b TP4 / q3c_30b TP4, each in t1 and t3 states.
export PY=/usr/local/python3.11.10/bin/python
export V030=/home/d00824595/rg_tip11_162/vllm030_pkgs
export PRODUCT=/home/d00824595/rg-tip15-0645
export ROOT=/home/d00824595/rg_matrix_162
export PORT=8251
export HBM_TOTAL=65536
export CARDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
for S in t1 t3; do
  MODEL=/mnt/weight/DeepSeek-V2-Lite NAME=dsv2_lite TP=2 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  MODEL=/mnt/weight/Qwen2.5-7B-Instruct NAME=q25_7b TP=2 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  MODEL=/mnt/weight/Qwen2.5-7B-Instruct NAME=q25_7b TP=4 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
  MODEL=/home/d00824595/rg162_weights/Qwen3-Coder-30B-A3B-Instruct NAME=q3c_30b TP=4 PP=1 DP=1 STATE=$S bash /home/d00824595/run_tip15_matrix.sh
done
echo MATRIX162_DONE
