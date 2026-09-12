#!/usr/bin/env bash
# T1: product HEAD, no additional-config (reload=0 default).
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
MODEL="${MODEL:?set MODEL}"
echo "[perf] T1 product=$PRODUCT_ROOT runner=$RUNNER"
if [[ -n "${START_CMD:-}" ]]; then
  eval "$START_CMD"
else
  echo "example:"
  echo "  PYTHONPATH=$PRODUCT_ROOT vllm serve \"$MODEL\" --port $PORT --tensor-parallel-size $TP"
  echo "then: RG_PERF_URL=http://127.0.0.1:${PORT}/v1/completions \\"
  echo "  python -m vllm_ascend.runtime_guard.test.perf.perf_baseline"
fi
