#!/usr/bin/env bash
# tip7 perf Phase A: full C1/C2/C3 cross-rotation on default model
# (Qwen2.5-0.5B, TP=1, card 0), RUNNER=v2 then v1.
set -uo pipefail

PRODUCT=/data0/test-mrv2-cann91/rg-tip7-verify
T0=/data0/test-mrv2-cann91/rg-tip7-t0
PERF_SCRIPTS=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/test/perf/scripts
ROOT=/data0/test-mrv2-cann91/rg_tip7_perf
LOG=$ROOT/master.log
mkdir -p "$ROOT"

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# vllm 0.29 main package tree must PREPEND, but keep container's Ascend
# site-packages (acl/te modules) — serve() prepends the worktree on top.
V029=/data0/test-mrv2-cann91/vllm029_pkgs
FULL_PY="${V029}:${PYTHONPATH:-}"

run_one(){
  local runner="$1"
  log "=== [$runner] C1/C2 cross-rotate start (6 cycles) ==="
  RUNNER="$runner" \
  RG_PRODUCT_ROOT="$PRODUCT" RG_T0_ROOT="$T0" RG_PERF_ROOT="$ROOT/rg_perf" \
  PYTHONPATH="$FULL_PY" \
  MODEL=/data0/weights/Qwen2.5-0.5B-Instruct SNAME=dsv2 PORT=8017 CARD=0 TP=1 \
    bash "$PERF_SCRIPTS/run_c1_c2_cross_rotate.sh" >> "$LOG" 2>&1
  log "=== [$runner] C1/C2 exit=$? ==="
  log "=== [$runner] C3 AB start ==="
  RUNNER="$runner" \
  RG_PRODUCT_ROOT="$PRODUCT" RG_T0_ROOT="$T0" RG_PERF_ROOT="$ROOT/rg_perf" \
  PYTHONPATH="$FULL_PY" \
  MODEL=/data0/weights/Qwen2.5-0.5B-Instruct SNAME=dsv2 PORT=8017 CARD=0 TP=1 \
    bash "$PERF_SCRIPTS/run_c3_ab.sh" >> "$LOG" 2>&1
  log "=== [$runner] C3 exit=$? ==="
}

log "=== tip7 perf start PRODUCT=$PRODUCT T0=$T0 cards=0 model=Qwen2.5-0.5B tp=1 ==="

run_one v2
run_one v1

log "=== PHASE A DONE ==="
