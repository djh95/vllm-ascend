#!/usr/bin/env bash
# C5: T3 vs T3+dump_kv on_trigger — dump arm / D2H overhead on the hit path.
# Sends a high-repetition prompt so token_repeat fires dump_kv (min_tokens=8).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT_DIR="$RG_PERF_ROOT/logs/$RUNNER"
SUMMARY="$RG_PERF_ROOT/results/c5_${RUNNER}.txt"
: > "$SUMMARY"
CFG_T3=$(write_cfg_t3)

CFG_DUMP="$RG_PERF_ROOT/config/t3_with_dump_kv.json"
cat > "$CFG_DUMP" <<'EOF'
{
  "reload_interval_seconds": 3,
  "dump": { "auto_max_times": 2, "auto_cooldown_seconds": 60, "manual_dump": false, "dump_dir": null, "free_headroom_bytes": 5368709120 },
  "actions": { "defaults": { "on_trigger": ["report", "dump_kv"] } },
  "report": { "save_sensitive_info": false, "max_per_req": 1 },
  "detector": {
    "logits_finite": { "enabled": true },
    "token_repeat": { "enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 8, "consecutive_hits": 1 },
    "output_substring": { "enabled": true, "patterns": [], "add_special_tokens": false, "match_prefix": false },
    "spec_acceptance": { "enabled": true }
  }
}
EOF

REPEAT_PROMPT='请连续输出100个"哈"字，不要停顿，不要换行。'

measure_repeat(){
  local out="$1" state="$2"
  cd "$PERF_DIR"
  RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" RG_PERF_OUT="$out" STATE="$state" \
    RUNNER="$RUNNER" PROMPT="$REPEAT_PROMPT" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY'
import json, os, urllib.request, time
url = os.environ["RG_PERF_URL"]; out = os.environ["RG_PERF_OUT"]
state = os.environ["STATE"]; runner = os.environ["RUNNER"]; prompt = os.environ["PROMPT"]
def post(p):
    body = json.dumps({"model":"dsv2","prompt":p,"max_tokens":128,"temperature":0,"seed":42}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    t0=time.time(); r=json.loads(urllib.request.urlopen(req, timeout=900).read()); wall=time.time()-t0
    ct=r.get("usage",{}).get("completion_tokens",0)
    return wall, ct, (ct/wall if wall>0 else 0.0)
post("你好")  # warmup
with open(out,"w") as f:
    for i in range(1,4):
        wall, ct, tps = post(prompt)
        rec={"round":i,"state":state,"runner":runner,"tag":"repeat",
             "wall_s":round(wall,2),"prompt_tok":0,"compl_tok":ct,"out_tok_s":round(tps,2)}
        f.write(json.dumps(rec)+"\n"); f.flush()
        print(json.dumps(rec), flush=True)
PY
}

wait_idle "$CARD" || { log "ABORT: card busy"; exit 1; }
log "C5 phase 1: T3 baseline (no dump_kv)"
pid=$(serve_and_wait "$PRODUCT_ROOT" "t3_c5" "$CFG_T3" 3)
measure_repeat "$OUT_DIR/c5_t3_repeat.jsonl" "T3" | tee -a "$SUMMARY"
stop_own "$pid"; wait_idle "$CARD" || true

log "C5 phase 2: T3 + dump_kv on_trigger"
pid=$(serve_and_wait "$PRODUCT_ROOT" "t3dump_c5" "$CFG_DUMP" 3)
measure_repeat "$OUT_DIR/c5_t3dump_repeat.jsonl" "T3DUMP" | tee -a "$SUMMARY"
sleep 3
stop_own "$pid"; wait_idle "$CARD" || true

{
  echo "=== C5: dump_kv arm/D2H overhead ==="
  echo -n "T3_geom T3DUMP_geom ratio(T3DUMP/T3): "
  geom_ratio "$OUT_DIR/c5_t3_repeat.jsonl" "$OUT_DIR/c5_t3dump_repeat.jsonl"
  echo "--- dump_kv activity in serve log (grep dump) ---"
  grep -iE "dump_kv|kv_cache|dump" "$RG_PERF_ROOT/serve/serve_${RUNNER}_t3dump_c5.log" | tail -5 || echo "(none)"
} | tee -a "$SUMMARY"
log "C5 done; summary=$SUMMARY"
