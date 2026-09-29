#!/usr/bin/env bash
# Serial driver: probe session first, then torch profiler session (waits for
# the same card pair to free up). Prevents the two sessions racing on cards.
set -u
D=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/test/perf/scripts/tip15_probe
echo "$(date '+%F %H:%M:%S') === chain start: probe then prof ==="
bash $D/run_tip15_probe.sh
echo "$(date '+%F %H:%M:%S') === probe phase finished, buffer 30s ==="
sleep 30
bash $D/run_tip15_prof.sh
echo "$(date '+%F %H:%M:%S') === chain done (probe + prof) ==="
