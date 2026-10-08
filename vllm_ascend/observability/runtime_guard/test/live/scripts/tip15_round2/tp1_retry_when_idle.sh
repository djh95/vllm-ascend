#!/usr/bin/env bash
for i in $(seq 1 120); do
  for c in 0 1 2 3 4 5 6 7; do
    h=$(npu-smi info 2>/dev/null | grep -A1 "^| $c     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+")
    if [ -n "$h" ] && [ "$h" -lt 5000 ]; then
      ROOT=/data0/test-mrv2-cann91/rg_tip15_tp1_v4 setsid bash /data0/test-mrv2-cann91/run_tip15_tp1_smoke.sh
      exit $?
    fi
  done
  sleep 300
done
