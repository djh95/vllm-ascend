#!/usr/bin/env bash
# T0: merge-base / no product bind. Set RG_T0_ROOT to a worktree without runtime_guard bind.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
T0_ROOT="${RG_T0_ROOT:?set RG_T0_ROOT to merge-base worktree (no RuntimeGuardProcessor.bind)}"
MODEL="${MODEL:?set MODEL}"
export PYTHONPATH="${T0_ROOT}:${PYTHONPATH}"
echo "[perf] T0 from $T0_ROOT runner=$RUNNER"
if [[ -n "${START_CMD:-}" ]]; then
  eval "$START_CMD"
else
  echo "Fill START_CMD or run manually, e.g.:"
  echo "  cd \"$T0_ROOT\" && vllm serve \"$MODEL\" --port $PORT --tensor-parallel-size $TP"
fi
