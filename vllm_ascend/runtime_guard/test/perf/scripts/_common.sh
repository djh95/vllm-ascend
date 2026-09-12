#!/usr/bin/env bash
# Perf shared env (analysis). Lab must set RG_PRODUCT_ROOT / MODEL / RG_PERF_*.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PRODUCT_ROOT="${RG_PRODUCT_ROOT:-$ROOT}"
export PYTHONPATH="${PRODUCT_ROOT}:${PYTHONPATH:-}"

RUNNER="${RUNNER:-v2}"
case "$RUNNER" in
  v2) export VLLM_USE_V2_MODEL_RUNNER=1 ;;
  v1) export VLLM_USE_V2_MODEL_RUNNER=0 ;;
  *) echo "RUNNER=v1|v2 required" >&2; exit 2 ;;
esac

export RG_PERF_ROOT="${RG_PERF_ROOT:-./rg_perf}"
mkdir -p "${RG_PERF_ROOT}/logs" "${RG_PERF_ROOT}/config"
MODEL="${MODEL:-}"
PORT="${PORT:-8017}"
TP="${TP:-1}"
echo "[perf] RUNNER=$RUNNER MODEL=$MODEL PORT=$PORT TP=$TP RG_PERF_ROOT=$RG_PERF_ROOT"
