#!/usr/bin/env bash
# Retry PP=2 smoke when cards 0-3 free up (HCCL 16666 conflict from neighbor job).
B=/data0/test-mrv2-cann91
for i in $(seq 1 96); do
  busy=0
  for c in 0 1 2 3; do
    h=$(npu-smi info 2>/dev/null | grep -A1 "^| $c     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+")
    [ -n "$h" ] && [ "$h" -ge 5000 ] && { busy=1; break; }
  done
  if [ "$busy" = 0 ]; then
    echo "$(date +%F_%H:%M:%S) cards0-3 idle, retrying pp2" >> $B/rg_tip15_pp2_smoke/retry.log
    bash $B/run_tip15_pp2_smoke.sh
    exit $?
  fi
  sleep 300
done
echo "timeout waiting idle" >> $B/rg_tip15_pp2_smoke/retry.log
