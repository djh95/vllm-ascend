#!/usr/bin/env bash
# C1 (T0 vs T1) + C2 (T1 vs T2) cross-rotation: T0->T1->T2->... N=6 cycles.
# Ratio from per-state geometric mean across all cycles. Run under BOTH runners
# (RUNNER=v2 then RUNNER=v1); results stay separated in logs/{v2,v1}/.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
CYCLES="${CYCLES:-6}"
OUT_DIR="$RG_PERF_ROOT/logs/$RUNNER"
mkdir -p "$OUT_DIR"
SUMMARY="$RG_PERF_ROOT/results/c1_c2_${RUNNER}.txt"
: > "$SUMMARY"

CFG_T2=$(write_cfg_t2)
log "C1/C2 cross-rotate runner=$RUNNER cycles=$CYCLES -> $OUT_DIR"

wait_idle "$CARD" || { log "ABORT: card $CARD busy at start"; exit 1; }

cycle_one(){
  local c="$1" pid
  log "---- cycle $c/$CYCLES ----"
  pid=$(serve_and_wait "$T0_ROOT" "t0_c$c")
  measure_rounds "$OUT_DIR/t0_r${c}.jsonl" "T0"
  stop_own "$pid"; wait_idle "$CARD" || true

  pid=$(serve_and_wait "$PRODUCT_ROOT" "t1_c$c")
  measure_rounds "$OUT_DIR/t1_r${c}.jsonl" "T1"
  stop_own "$pid"; wait_idle "$CARD" || true

  pid=$(serve_and_wait "$PRODUCT_ROOT" "t2_c$c" "$CFG_T2" 3)
  measure_rounds "$OUT_DIR/t2_r${c}.jsonl" "T2"
  stop_own "$pid"; wait_idle "$CARD" || true
}

for c in $(seq 1 "$CYCLES"); do
  cycle_one "$c" 2>&1 | tee -a "$SUMMARY"
done

cat_state t0 "$OUT_DIR" "$OUT_DIR/t0_all.jsonl"
cat_state t1 "$OUT_DIR" "$OUT_DIR/t1_all.jsonl"
cat_state t2 "$OUT_DIR" "$OUT_DIR/t2_all.jsonl"

{
  echo "=== C1: T1/T0 (want >= 0.999) ==="
  echo -n "T0_geom T1_geom ratio: "; geom_ratio "$OUT_DIR/t0_all.jsonl" "$OUT_DIR/t1_all.jsonl"
  echo "=== C2: T2/T1 (want >= 0.999) ==="
  echo -n "T1_geom T2_geom ratio: "; geom_ratio "$OUT_DIR/t1_all.jsonl" "$OUT_DIR/t2_all.jsonl"
} | tee -a "$SUMMARY"

log "C1/C2 done; summary=$SUMMARY"
