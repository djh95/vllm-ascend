#!/usr/bin/env bash
# P0 batch launcher (NEW config schema — runtime_config_hot_reload bool via
# --additional-config; no legacy runtime_config nesting / reload_interval).
# Runs each P0 case as a self-contained script (own card pick / boot / kill /
# residual / HBM / trap-dump-cleanup) for the requested runners.
#   CASES   space list (default: P0-1/2/5/6/7/8 batch)
#   RUNNERS space list (default "v2 v1")
#   RG_OUT_ROOT  summary dir (default /data0/test-mrv2-cann91/rg_p0_batch)
#   CARDS   idle-card pool handed down to the case scripts (parallel streams
#           should partition the pool, e.g. CARDS="0 1 2 3" / "4 5 6 7")
# Per-case exit: 0=PASS 1=FAIL 3=SKIP.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CASES=${CASES:-"p0_01_guard_off p0_02_token_repeat p0_05_inject_nan p0_06_async_after_sample p0_07_dump_schema p0_08_disk_reclaim"}
RUNNERS=${RUNNERS:-"v2 v1"}
OUT=${RG_OUT_ROOT:-/data0/test-mrv2-cann91/rg_p0_batch}
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"
overall=0

for r in $RUNNERS; do
  case "$r" in
    v2) P=/data0/test-mrv2-cann91/rg-tip15-8f5e3 ;;
    v1) P=/data0/test-mrv2-cann91/rg-v1-9343 ;;
    *) echo "bad runner: $r" >&2; exit 2 ;;
  esac
  for c in $CASES; do
    echo "===== $c / $r $(date '+%F %T') =====" | tee -a "$SUMMARY"
    RUNNER=$r PRODUCT=$P CARDS="${CARDS:-0 1 2 3 4 5 6 7}" bash "$HERE/$c.sh" \
      > "$OUT/${c}_${r}.out" 2>&1
    rc=$?
    v=FAIL
    [ "$rc" = 0 ] && v=PASS
    [ "$rc" = 3 ] && v=SKIP
    { [ "$rc" = 0 ] || [ "$rc" = 3 ]; } || overall=1
    echo "$c $r VERDICT $v (rc=$rc)" | tee -a "$SUMMARY"
    grep -E "VERDICT|REPORT_PATH|REPORT_CHECK|BODY_CMP|ILL_TYPE|DF_|K1[123]|K_OVERALL|HOOK_EVIDENCE|EVIDENCE|SKIP reason" \
      "$OUT/${c}_${r}.out" 2>/dev/null | tail -14 >> "$SUMMARY" || true
  done
done
echo "BATCH_DONE overall=$overall ($(date '+%F %T'))" | tee -a "$SUMMARY"
exit $overall
