#!/usr/bin/env bash
# C3 A/B quick: requires T2/T3 server already up; RUNNER labeled in logs via env.
set -euo pipefail
RUNNER="${RUNNER:-v2}"
export VLLM_USE_V2_MODEL_RUNNER=$([ "$RUNNER" = v2 ] && echo 1 || echo 0)
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="${RG_PRODUCT_ROOT:-$ROOT}:${PYTHONPATH:-}"
export RG_PERF_ROOT="${RG_PERF_ROOT:-./rg_perf}"
mkdir -p "${RG_PERF_ROOT}/logs"
echo "[perf] C3 runner=$RUNNER RG_PERF_ROOT=$RG_PERF_ROOT"
python3 -m vllm_ascend.runtime_guard.test.perf.perf_ab_quick
