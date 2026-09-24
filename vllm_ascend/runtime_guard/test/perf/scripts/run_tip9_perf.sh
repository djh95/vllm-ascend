#!/usr/bin/env bash
# tip9 perf: clean rerun. C1/C2 cross-rotate + C3 AB on latest config tip
# (70d219d68), v2 runner only per user instruction.
set -uo pipefail

PRODUCT=/data0/test-mrv2-cann91/rg-tip9-verify
T0=/data0/test-mrv2-cann91/rg-tip9-t0
PERF_SCRIPTS=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/test/perf/scripts
ROOT=/data0/test-mrv2-cann91/rg_tip9_perf
LOG=$ROOT/master.log
mkdir -p "$ROOT"

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

V029=/data0/test-mrv2-cann91/vllm029_pkgs
FULL_PY="${V029}:${PYTHONPATH:-}"

log "=== tip9 perf start PRODUCT=$PRODUCT T0=$T0 cards=0 model=Qwen2.5-0.5B tp=1 tip=70d219d68 ==="

log "=== [v2] C1/C2 cross-rotate start (6 cycles) ==="
RUNNER=v2 \
RG_PRODUCT_ROOT="$PRODUCT" RG_T0_ROOT="$T0" RG_PERF_ROOT="$ROOT/rg_perf" \
PYTHONPATH="$FULL_PY" \
MODEL=/data0/weights/Qwen2.5-0.5B-Instruct SNAME=dsv2 PORT=8017 CARD=0 TP=1 \
  bash "$PERF_SCRIPTS/run_c1_c2_cross_rotate.sh" >> "$LOG" 2>&1
log "=== [v2] C1/C2 exit=$? ==="

log "=== [v2] C3 AB start ==="
RUNNER=v2 \
RG_PRODUCT_ROOT="$PRODUCT" RG_T0_ROOT="$T0" RG_PERF_ROOT="$ROOT/rg_perf" \
PYTHONPATH="$FULL_PY" \
MODEL=/data0/weights/Qwen2.5-0.5B-Instruct SNAME=dsv2 PORT=8017 CARD=0 TP=1 \
  bash "$PERF_SCRIPTS/run_c3_ab.sh" >> "$LOG" 2>&1
log "=== [v2] C3 exit=$? ==="

log "=== TIP9 PERF DONE ==="
