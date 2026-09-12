#!/usr/bin/env bash
# Serve T1 skeleton (product HEAD, no additional-config). RUNNER=v1|v2.
set -euo pipefail
RUNNER="${RUNNER:-v2}"
export VLLM_USE_V2_MODEL_RUNNER=$([ "$RUNNER" = v2 ] && echo 1 || echo 0)
MODEL="${MODEL:?set MODEL}"
PORT="${PORT:-8017}"
echo "[perf] T1 runner=$RUNNER — start product vllm serve here (fill lab flags)"
echo "  example: vllm serve \"$MODEL\" --port $PORT"
echo "  then: RG_PERF_URL=http://127.0.0.1:${PORT}/v1/completions python -m vllm_ascend.runtime_guard.test.perf.perf_baseline"
