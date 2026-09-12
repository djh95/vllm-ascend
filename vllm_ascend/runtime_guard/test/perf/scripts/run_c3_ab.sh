#!/usr/bin/env bash
# C3 A/B: detectors off vs on under reload=3 (perf_ab_quick).
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
export RG_PERF_OUT_AB="${RG_PERF_OUT_AB:-${RG_PERF_ROOT}/logs/${RUNNER}/perf_ab_quick.jsonl}"
mkdir -p "$(dirname "$RG_PERF_OUT_AB")"
echo "[perf] C3 runner=$RUNNER out=$RG_PERF_OUT_AB"
python3 -m vllm_ascend.runtime_guard.test.perf.perf_ab_quick
