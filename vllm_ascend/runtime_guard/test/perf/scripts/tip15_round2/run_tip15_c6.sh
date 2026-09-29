#!/usr/bin/env bash
# C6 idle leak-back for product tip b1b58921f+2ca3cdab5: after a stress phase
# with detectors on (T3), the engine must return RSS/HBM to the pre-stress
# baseline. Samples are scoped to OUR pgid + OUR cards only — the shared
# container runs other users' vllm processes which previously polluted this
# metric (v1 leakback 受污染).
# Gate: end rss_delta <= 30MB AND hbm_delta <= 30MB vs post-stress baseline.
# RUNNER=v2|v1 (default v2). Model Qwen2.5-7B TP=2, cards 4,5, port 8184.
# SAFETY: waits idle cards; pgid-scoped cleanup; residual ignores defunct.
set -uo pipefail
RUNNER=${RUNNER:-v2}
PY=/opt/slime/venv/bin/python
V030=/data0/test-mrv2-cann91/vllm030_pkgs
TREE=/data0/test-mrv2-cann91/rg-tip15-8f5e3
ROOT=/data0/test-mrv2-cann91/rg_c6_tip15_${RUNNER}
LOG=$ROOT/master.log
PORT=8184
MODEL=/data0/weights/Qwen2.5-7B-Instruct
CARDS=4,5
LEAKBACK_SEC=${LEAKBACK_SEC:-300}
LEAKBACK_INT=${LEAKBACK_INT:-15}
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
{"dump": {"dump_dir": "$ROOT/dump", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true}, "spec_acceptance": {"enabled": true}}}
EOF
}

log "=== C6 tip15 leakback: RUNNER=$RUNNER cards=$CARDS port=$PORT sec=$LEAKBACK_SEC ==="
write_cfg_t3
if ! wait_idle_cards; then log "ABORT: cards never idle"; log "C6_TIP15 ABORT"; exit 1; fi
pid=$( cd "$TREE"
  env PYTHONPATH="$V030:$TREE:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=$CARDS \
    VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=$([ "$RUNNER" = v2 ] && echo 1 || echo 0) \
  setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name dsv2 --port "$PORT" \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --enforce-eager \
    --additional-config "{\"runtime_config_path\": \"$ROOT/cfg_t3.json\", \"runtime_config_hot_reload\": true, \"runtime_report_dir\": \"$ROOT/report\"}" \
    > "$ROOT/serve.log" 2>&1 & echo $! )
log "boot pgid=$pid"
if [ "$(wait_health)" != 1 ]; then
  log "HEALTH_FAIL"; tail -20 "$ROOT/serve.log" | tee -a "$LOG"
  stop_pgid "$pid"; log "C6_TIP15 FAIL (health)"; exit 1
fi
log "health=200"

log "stress phase: REQS x2 + repeat prompts (detectors on)"
RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" RG_STRESS_OUT="$ROOT/stress.jsonl" "$PY" - <<'PY' 2>&1 | tail -3 | tee -a "$LOG"
import json, os, urllib.request
url = os.environ["RG_PERF_URL"]
reqs = [("short", "请介绍一下长城的历史和主要关口。", 64),
        ("medium", "请介绍一下李白的人生经历和代表作品。", 128),
        ("repeat", '请连续输出100个"哈"字，不要停顿，不要换行。', 128)]
with open(os.environ["RG_STRESS_OUT"], "w") as f:
    for cycle in (1, 2):
        for tag, p, mt in reqs:
            body = json.dumps({"model": "dsv2", "prompt": p, "max_tokens": mt, "temperature": 0, "seed": 42}).encode()
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}), timeout=900).read())
            rec = {"cycle": cycle, "tag": tag, "compl_tok": r.get("usage", {}).get("completion_tokens", 0)}
            f.write(json.dumps(rec) + "\n"); f.flush()
print("stress done")
PY

log "leak-back phase: ${LEAKBACK_SEC}s idle sampling (pgid=$pid scoped)"
rss_kb(){ ps -eo pgid,rss --no-headers | awk -v g="$pid" '$1==g {s+=$2} END {print s+0}'; }
hbm_sum(){ local c t=0; IFS=',' read -ra _cs <<< "$CARDS"; for c in "${_cs[@]}"; do t=$((t + $(hbm "$c" 2>/dev/null || echo 0))); done; echo "$t"; }
BASE_RSS=$(rss_kb)
BASE_HBM=$(hbm_sum)
log "baseline rss_kb=$BASE_RSS hbm_mb=$BASE_HBM"
: > "$ROOT/leakback.jsonl"
LAST_RSS=$BASE_RSS; LAST_HBM=$BASE_HBM
n=$(( LEAKBACK_SEC / LEAKBACK_INT ))
for i in $(seq 1 "$n"); do
  sleep "$LEAKBACK_INT"
  LAST_RSS=$(rss_kb); LAST_HBM=$(hbm_sum)
  echo "{\"phase\":\"leakback_sample\",\"elapsed_s\":$((i * LEAKBACK_INT)),\"rss_kb\":$LAST_RSS,\"rss_delta_kb\":$((LAST_RSS - BASE_RSS)),\"hbm_mb\":$LAST_HBM,\"hbm_delta_mb\":$((LAST_HBM - BASE_HBM))}" >> "$ROOT/leakback.jsonl"
done
RSS_DELTA_MB=$(( (LAST_RSS - BASE_RSS) / 1024 ))
HBM_DELTA_MB=$(( LAST_HBM - BASE_HBM ))
log "end rss_delta_mb=$RSS_DELTA_MB hbm_delta_mb=$HBM_DELTA_MB"
ok=1
[ "$RSS_DELTA_MB" -le 30 ] || { ok=0; log "FAIL: rss leak $RSS_DELTA_MB MB > 30MB"; }
[ "$HBM_DELTA_MB" -le 30 ] && [ "$HBM_DELTA_MB" -ge -30 ] || { ok=0; log "FAIL: hbm leak $HBM_DELTA_MB MB (|delta|>30MB)"; }
if [ "$ok" = 1 ]; then log "C6_TIP15 RUNNER=$RUNNER rss_delta_mb=$RSS_DELTA_MB hbm_delta_mb=$HBM_DELTA_MB (gate: rss<=30MB, |hbm|<=30MB) PASS"; else log "C6_TIP15 RUNNER=$RUNNER rss_delta_mb=$RSS_DELTA_MB hbm_delta_mb=$HBM_DELTA_MB (gate: rss<=30MB, |hbm|<=30MB) FAIL"; fi

stop_pgid "$pid"
sleep 3
out=$(ps -eo pid,ppid,pgid,etime,args | awk -v g="$pid" '$3==g && $1!=g' | grep -v defunct | head -10)
[ -n "$out" ] && { log "RESIDUAL_LIVE:"; echo "$out" | tee -a "$LOG"; kill -9 -- -"$pid" 2>/dev/null || true; } || log "residual: none"
ok=1; IFS=',' read -ra _cs <<< "$CARDS"
for c in "${_cs[@]}"; do [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null || { ok=0; log "HBM card $c still held"; }; done
[ "$ok" = 1 ] && log "HBM released on all cards"
log "=== C6 TIP15 DONE RUNNER=$RUNNER ==="
