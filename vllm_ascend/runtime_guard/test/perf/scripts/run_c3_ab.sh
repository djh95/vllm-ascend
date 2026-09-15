#!/usr/bin/env bash
# C3: T2 (detectors off) vs T3 (detectors on) under reload=3, cross-rotated B/A x3.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="$RG_PERF_ROOT/logs/$RUNNER/c3_ab.jsonl"
SUMMARY="$RG_PERF_ROOT/results/c3_${RUNNER}.txt"
: > "$SUMMARY"
CFG_T2=$(write_cfg_t2)
export RG_PERF_CFG="$CFG_T2"

wait_idle "$CARD" || { log "ABORT: card busy"; exit 1; }
pid=$(serve_and_wait "$PRODUCT_ROOT" "t2_c3" "$CFG_T2" 3)

cd "$PERF_DIR"
RUNNER="$RUNNER" RG_PERF_OUT="$OUT" RG_PERF_CFG="$CFG_T2" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY' | tee -a "$SUMMARY"
import json, os, time
from perf_lib import set_detectors, trigger, warmup, post, REQS
out = os.environ["RG_PERF_OUT"]
runner = os.environ["RUNNER"]
set_detectors(True); trigger(); time.sleep(1)
warmup(rounds=1)
with open(out, "w") as f:
    for idx, (state, on) in enumerate([("B", True), ("A", False)] * 3, 1):
        set_detectors(on); trigger(); time.sleep(1)
        for tag, p, mt in REQS:
            wall, pt, ct, tps = post(p, mt)
            rec = {"round": idx, "state": state, "runner": runner, "tag": tag,
                   "wall_s": round(wall, 2), "prompt_tok": pt, "compl_tok": ct,
                   "out_tok_s": round(tps, 2)}
            f.write(json.dumps(rec) + "\n"); f.flush()
print("c3 measure done", flush=True)
PY

stop_own "$pid"; wait_idle "$CARD" || true

"$PY" - "$OUT" <<'PY' >> "$SUMMARY"
import json, math, sys
def gm(state):
    vals = [float(json.loads(l)["out_tok_s"]) for l in open(sys.argv[1])
            if l.strip() and json.loads(l)["state"] == state]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
a, b = gm("A"), gm("B")
print(f"C3 T2(A)_geom={a:.3f} T3(B)_geom={b:.3f} T3/T2={b/a:.5f} (want >= 0.990)")
PY
log "C3 done; summary=$SUMMARY"
