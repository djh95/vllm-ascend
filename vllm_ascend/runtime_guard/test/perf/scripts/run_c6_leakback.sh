#!/usr/bin/env bash
# C6: idle leak-back after load (uses perf_lib leakback helpers when driven from Python).
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
export RG_PERF_OUT_LEAKBACK_AB="${RG_PERF_OUT_LEAKBACK_AB:-${RG_PERF_ROOT}/logs/${RUNNER}/leakback_ab.jsonl}"
mkdir -p "$(dirname "$RG_PERF_OUT_LEAKBACK_AB")"
echo "[perf] C6 runner=$RUNNER — after AB rounds, leave server idle RG_PERF_LEAKBACK_SEC (default 300s)"
echo "RSS/HBM should return near start (±30MB). See perf_lib leakback_* helpers."
echo "Wire: run perf_ab_quick (records leakback) or call leakback phase explicitly on lab."
