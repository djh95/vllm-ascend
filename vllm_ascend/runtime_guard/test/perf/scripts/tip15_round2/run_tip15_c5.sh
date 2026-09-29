#!/usr/bin/env bash
# C5 dump-hit overhead for product tip b1b58921f+2ca3cdab5: T3 (report-only)
# vs T3+dump_kv on the token_repeat hit path. High-repetition prompt makes
# token_repeat fire (min_tokens=8) so dump_kv actually arms -> the measured
# ratio reflects the real hit path (unlike the old 0.98x noise where no dump
# ever landed). Gate: official threshold TBD -> record ratio + assert dump hit.
# RUNNER=v2|v1 (default v2). Model Qwen2.5-7B TP=2, cards 4,5, port 8183.
# SAFETY: waits idle cards; pgid-scoped cleanup; residual ignores defunct.
set -uo pipefail
RUNNER=${RUNNER:-v2}
PY=/opt/slime/venv/bin/python
V030=/data0/test-mrv2-cann91/vllm030_pkgs
TREE=/data0/test-mrv2-cann91/rg-tip15-8f5e3
ROOT=/data0/test-mrv2-cann91/rg_c5_tip15_${RUNNER}
LOG=$ROOT/master.log
PORT=8183
MODEL=/data0/weights/Qwen2.5-7B-Instruct
CARDS=4,5
REPEAT_PROMPT='请连续输出100个"哈"字，不要停顿，不要换行。'
mkdir -p "$ROOT"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

hbm(){ npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+"; }
wait_idle_cards(){
  local i c ok
  for i in $(seq 1 180); do
    ok=1; IFS=',' read -ra _cs <<< "$CARDS"
    for c in "${_cs[@]}"; do [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null || { ok=0; break; }; done
    [ "$ok" = 1 ] && return 0; sleep 20
  done
  return 1
}
wait_health(){
  local i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done
  echo 0
}
stop_pgid(){ kill -- -"$1" 2>/dev/null || true; sleep 3; kill -9 -- -"$1" 2>/dev/null || true; sleep 2; }
write_cfg_t3(){
  cat > "$ROOT/cfg_t3.json" <<EOF
{"dump": {"dump_dir": "$ROOT/dump_t3", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 32, "consecutive_hits": 1}, "spec_acceptance": {"enabled": true}}}
EOF
}
write_cfg_dump(){
  cat > "$ROOT/cfg_dump.json" <<EOF
{"dump": {"dump_dir": "$ROOT/dump_arm", "auto_max_times": 2, "auto_cooldown_seconds": 60, "manual_dump": false, "free_headroom_bytes": 5368709120}, "actions": {"defaults": {"on_trigger": ["report", "dump_kv"]}}, "report": {"save_sensitive_info": false, "max_per_req": 1}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 8, "consecutive_hits": 1}, "spec_acceptance": {"enabled": true}}}
EOF
}
boot(){
  local state=$1 cfg=$2
  ( cd "$TREE"
    env PYTHONPATH="$V030:$TREE:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=$CARDS \
      VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=$([ "$RUNNER" = v2 ] && echo 1 || echo 0) \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name dsv2 --port "$PORT" \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --enforce-eager \
      --additional-config "{\"runtime_config_path\": \"$cfg\", \"runtime_config_hot_reload\": true, \"runtime_report_dir\": \"$ROOT/report_${state}\"}" \
      > "$ROOT/serve_${state}.log" 2>&1 & echo $! )
}
measure_repeat(){
  local state=$1 out="$ROOT/c5_${state}.jsonl"
  RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" RG_PERF_OUT="$out" STATE="$state" RUNNER="$RUNNER" PROMPT="$REPEAT_PROMPT" "$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, urllib.request, time
url = os.environ["RG_PERF_URL"]; out = os.environ["RG_PERF_OUT"]
state = os.environ["STATE"]; runner = os.environ["RUNNER"]; prompt = os.environ["PROMPT"]
# HARDENING (2026-09-29): round on 2ca3cdab5 the bare urlopen hung 90min on the first
# request with zero POST lines in serve log. Use an explicit no-proxy opener, short
# timeout, and per-request diagnostics so a rerun either succeeds or fails loudly.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(p):
    body = json.dumps({"model": "dsv2", "prompt": p, "max_tokens": 128, "temperature": 0, "seed": 42}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(_OPENER.open(req, timeout=120).read())
    except Exception as e:
        print(f"REQ_FAIL after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}", flush=True)
        raise
    wall = time.time() - t0
    ct = r.get("usage", {}).get("completion_tokens", 0)
    return wall, ct, (ct / wall if wall > 0 else 0.0)
print("WARMUP_START", flush=True)
post("你好")
print("WARMUP_DONE", flush=True)
with open(out, "w") as f:
    for i in range(1, 4):
        wall, ct, tps = post(prompt)
        rec = {"round": i, "state": state, "runner": runner, "tag": "repeat",
               "wall_s": round(wall, 2), "compl_tok": ct, "out_tok_s": round(tps, 2)}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps(rec), flush=True)
PY
}

log "=== C5 tip15 dump-hit overhead: RUNNER=$RUNNER cards=$CARDS port=$PORT ==="
write_cfg_t3; write_cfg_dump
mkdir -p "$ROOT/dump_arm" "$ROOT/report_t3" "$ROOT/report_t3dump"

log "phase 1: T3 report-only baseline"
if ! wait_idle_cards; then log "ABORT: cards never idle"; log "C5_TIP15 ABORT"; exit 1; fi
pgid=$(boot t3 "$ROOT/cfg_t3.json")
log "t3 pgid=$pgid"
if [ "$(wait_health)" != 1 ]; then log "t3 HEALTH_FAIL"; tail -20 "$ROOT/serve_t3.log" | tee -a "$LOG"; stop_pgid "$pgid"; exit 1; fi
log "t3 health=200"
measure_repeat t3
stop_pgid "$pgid"; sleep 5

log "phase 2: T3 + dump_kv on_trigger"
if ! wait_idle_cards; then log "ABORT: cards never idle"; log "C5_TIP15 ABORT"; exit 1; fi
pgid=$(boot t3dump "$ROOT/cfg_dump.json")
log "t3dump pgid=$pgid"
if [ "$(wait_health)" != 1 ]; then log "t3dump HEALTH_FAIL"; tail -20 "$ROOT/serve_t3dump.log" | tee -a "$LOG"; stop_pgid "$pgid"; exit 1; fi
log "t3dump health=200"
measure_repeat t3dump
sleep 3
inc=$(find "$ROOT/report_t3dump" -maxdepth 3 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
pt_count=$(find "$ROOT/dump_arm" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
log "t3dump incident_reports=$inc dump_pt=$pt_count"
stop_pgid "$pgid"; sleep 5

RUNNER="$RUNNER" "$PY" - "$ROOT" "$LOG" <<'PY' 2>&1 | tee -a "$LOG"
import glob, json, math, os, sys
root, logp = sys.argv[1], sys.argv[2]
def gm(path):
    vals = [float(json.loads(l)["out_tok_s"]) for l in open(path) if l.strip()]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
g_t3, g_dump = gm(f"{root}/c5_t3.jsonl"), gm(f"{root}/c5_t3dump.jsonl")
ratio = g_dump / g_t3 if g_t3 else 0.0
pt = len(glob.glob(f"{root}/dump_arm/**/*.pt", recursive=True))
hit = pt > 0
overall = "PASS" if hit else "FAIL"
msg = f"C5_TIP15 RUNNER={os.environ.get('RUNNER','v2')} T3_geom={g_t3:.3f} T3DUMP_geom={g_dump:.3f} ratio={ratio:.5f} dump_hit={hit} (pt={pt}) {overall}"
print(msg)
open(logp, "a").write(msg + "\n")
PY
log "=== C5 TIP15 DONE RUNNER=$RUNNER ==="
