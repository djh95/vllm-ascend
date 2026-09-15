#!/usr/bin/env bash
# C3 variant: isolate the two "real" detectors — spec_acceptance + token_repeat.
# output_substring + logits_finite stay DISABLED the whole time.
# A = spec_acceptance+token_repeat off, B = on, cross-rotated B/A x3 (same as run_c3_ab.sh).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="$RG_PERF_ROOT/logs/$RUNNER/c3_spec_tokenrep.jsonl"
SUMMARY="$RG_PERF_ROOT/results/c3_spec_tokenrep_${RUNNER}.txt"
: > "$SUMMARY"

write_cfg_two(){
  local cfg="$RG_PERF_ROOT/config/t3_only_spec_tokenrep.json"
  cat > "$cfg" <<'EOF'
{
  "reload_interval_seconds": 3,
  "dump": { "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false },
  "actions": { "defaults": { "on_trigger": ["report"] } },
  "report": { "save_sensitive_info": false, "max_per_req": 1 },
  "detector": {
    "spec_acceptance": { "enabled": true, "window": 10, "low_threshold": 0.3, "len_low_threshold": 1.4, "high_threshold": 0.96, "len_high_threshold": 2.8 },
    "token_repeat": { "enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 32, "consecutive_hits": 1 },
    "output_substring": { "enabled": false, "patterns": [], "add_special_tokens": false, "match_prefix": false },
    "logits_finite": { "enabled": false }
  }
}
EOF
  echo "$cfg"
}

CFG=$(write_cfg_two)
export RG_PERF_CFG="$CFG"

wait_idle "$CARD" || { log "ABORT: card busy"; exit 1; }
pid=$(serve_and_wait "$PRODUCT_ROOT" "t3_two" "$CFG" 3)

cd "$PERF_DIR"
RUNNER="$RUNNER" RG_PERF_OUT="$OUT" RG_PERF_CFG="$CFG" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY' | tee -a "$SUMMARY"
import json, os, time
from perf_lib import trigger, warmup, post, REQS
out = os.environ["RG_PERF_OUT"]
runner = os.environ["RUNNER"]
cfg = os.environ["RG_PERF_CFG"]

def set_two(on):
    c = json.load(open(cfg))
    det = c.setdefault("detector", {})
    det.setdefault("spec_acceptance", {})["enabled"] = on
    det.setdefault("token_repeat", {})["enabled"] = on
    det.setdefault("output_substring", {})["enabled"] = False
    det.setdefault("logits_finite", {})["enabled"] = False
    json.dump(c, open(cfg, "w"), indent=2)

set_two(True); trigger(); time.sleep(1)
warmup(rounds=1)
with open(out, "w") as f:
    for idx, (state, on) in enumerate([("B", True), ("A", False)] * 3, 1):
        set_two(on); trigger(); time.sleep(1)
        for tag, p, mt in REQS:
            wall, pt, ct, tps = post(p, mt)
            rec = {"round": idx, "state": state, "runner": runner, "tag": tag,
                   "wall_s": round(wall, 2), "prompt_tok": pt, "compl_tok": ct,
                   "out_tok_s": round(tps, 2)}
            f.write(json.dumps(rec) + "\n"); f.flush()
print("c3 spec_tokenrep measure done", flush=True)
PY

stop_own "$pid"; wait_idle "$CARD" || true

"$PY" - "$OUT" <<'PY' >> "$SUMMARY"
import json, math, sys
def gm(state):
    vals = [float(json.loads(l)["out_tok_s"]) for l in open(sys.argv[1])
            if l.strip() and json.loads(l)["state"] == state]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
a, b = gm("A"), gm("B")
print(f"C3(spec_acceptance+token_repeat only) T2(A)_geom={a:.3f} T3(B)_geom={b:.3f} T3/T2={b/a:.5f} (want >= 0.990)")
PY
log "C3 spec_tokenrep done; summary=$SUMMARY"
