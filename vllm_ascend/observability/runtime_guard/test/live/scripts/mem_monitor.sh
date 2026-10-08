#!/usr/bin/env bash
# Memory monitor for runtime_guard soak: sample total vllm host RSS + per-card HBM.
# Detects memory leak / idle leak-back over long runs.
#
#   nohup bash /data0/test-mrv2-cann91/rg_test/mem_monitor.sh [interval_sec] > /tmp/rg_mem_monitor.log 2>&1 &
#
# Log line (appended every interval):
#   <ts> total_rss_mb=<N> procs=<N> hbm_mb="<8 values cards 0-7>"
set -uo pipefail

LOG=/tmp/rg_mem_monitor.log
INTERVAL=${1:-600}

while true; do
  TS=$(date '+%F %H:%M:%S')
  # Total host RSS (MB) of all vllm processes: api_server + EngineCore + TP/PP workers.
  read TOTAL_RSS PROCS <<< "$(ps -eo rss,stat,comm,cmd | grep -iE 'api_server|enginecore|vllmworker' | grep -v grep | grep -v ' Z' | awk '{s+=$1; n++} END {printf "%.0f %d", s/1024, n}')"
  # Per-card HBM usage (MB), cards 0-7 in order. npu-smi format: "29472/ 32768".
  HBM=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+/ *32768' | sed 's|/.*||' | tr '\n' ' ')
  echo "$TS total_rss_mb=$TOTAL_RSS procs=$PROCS hbm_mb=\"$HBM\"" >> "$LOG"
  sleep "$INTERVAL"
done
