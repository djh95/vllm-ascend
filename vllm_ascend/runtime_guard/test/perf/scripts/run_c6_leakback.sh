#!/usr/bin/env bash
# C6: idle leak-back — after stress (detectors on), RSS/HBM must return to
# within 30MB of the pre-stress baseline.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
CFG_T3=$(write_cfg_t3)
OUT="$RG_PERF_ROOT/logs/$RUNNER/leakback.jsonl"
SUMMARY="$RG_PERF_ROOT/results/c6_${RUNNER}.txt"
: > "$SUMMARY"

wait_idle "$CARD" || { log "ABORT: card busy"; exit 1; }
pid=$(serve_and_wait "$PRODUCT_ROOT" "t3_c6" "$CFG_T3" 3)

cd "$PERF_DIR"
RUNNER="$RUNNER" RG_PERF_CFG="$CFG_T3" RG_PERF_OUT_LEAKBACK_AB="$OUT" \
  RG_PERF_LEAKBACK_SEC="${RG_PERF_LEAKBACK_SEC:-300}" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY' | tee -a "$SUMMARY"
import os
from perf_lib import warmup, run_rounds, run_leakback
warmup(rounds=1)
run_rounds([("T3", True)] * 3, os.environ["RG_PERF_OUT_LEAKBACK_AB"] + ".stress", with_mem=False)
print("[stress done, entering idle leak-back]", flush=True)
run_leakback("T3", os.environ["RG_PERF_OUT_LEAKBACK_AB"])
print("c6 done", flush=True)
PY

stop_own "$pid"; wait_idle "$CARD" || true

"$PY" - "$OUT" <<'PY' >> "$SUMMARY"
import json, sys
start = end = None
for l in open(sys.argv[1]):
    if not l.strip(): continue
    r = json.loads(l)
    if r.get("phase") == "leakback_start": start = r
    if r.get("phase") == "leakback_end": end = r
if start and end:
    d = end.get("rss_delta_kb", 0)
    print(f"C6 leakback rss_delta_kb={d} (want <= 30720) -> {'PASS' if d <= 30720 else 'FAIL'}")
    print(f"  start rss_kb={start.get('rss_kb')} end rss_kb={end.get('rss_kb')} hbm={end.get('hbm_mb')}")
else:
    print("C6 no leakback data (check leakback.jsonl)")
PY
log "C6 done; summary=$SUMMARY"
