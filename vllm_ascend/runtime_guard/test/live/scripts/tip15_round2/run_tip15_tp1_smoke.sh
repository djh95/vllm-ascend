#!/usr/bin/env bash
# P0-3 topology smoke v3: TP=1 degenerate lane (no collectives, local poll).
# v3 fixes vs v2 (found on 13.160/13.162 during multi-host rollout):
#   1. hbm() dual-format: legacy "N     910" rows AND npu-smi 26.0.rc1
#      "Ascend910" rows where card id sits on the chip line col $3.
#   2. boot no longer captures $(... & echo $!) — setsid+subshell races into
#      a do_wait deadlock; we launch detached and pgrep the pid instead.
#   3. write_cfg moved AFTER health=200: since 1aa5f5f65 the config leader
#      ensure_persisted() OVERWRITES the JSON with startup defaults, wiping a
#      pre-written manual_dump:true. Toggle on after serve is up; hot reload
#      (3s poll) picks it up. (v2 PASS on 9.103 was measured on 0645cf331,
#      pre-watermark semantics.)
# Env overrides: PY= V030= PRODUCT= MODEL= ROOT= PORT= HBM_TOTAL= CARDS=
set -uo pipefail
PY=${PY:-/opt/slime/venv/bin/python}
V030=${V030:-/data0/test-mrv2-cann91/vllm030_pkgs}
PRODUCT=${PRODUCT:-/data0/test-mrv2-cann91/rg-tip15-8f5e3}
MODEL=${MODEL:-/data0/weights/Qwen2.5-7B-Instruct}
ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_tip15_tp1_smoke}
PORT=${PORT:-8193}
HBM_TOTAL=${HBM_TOTAL:-32768}
CARDS=${CARDS:-"0 1 2 3 4 5 6 7"}
LOG=$ROOT/master.log
mkdir -p "$ROOT" "$ROOT/report"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }
PASS=1
fail(){ PASS=0; log "FAIL: $*"; }

hbm(){
  local v
  v=$(npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*$HBM_TOTAL" | tail -1 | grep -oE "^[0-9]+")
  if [ -z "$v" ]; then
    v=$(npu-smi info 2>/dev/null | awk -v n="$1" -v t="$HBM_TOTAL" \
      '$1=="|" && $2 ~ /^[0-9]+$/ && $3==n { if (match($0, "[0-9]+[ ]*/[ ]*" t)) { s=substr($0, RSTART, RLENGTH); sub(/[ ].*/, "", s); print s; exit }; if ((getline) > 0 && match($0, "[0-9]+[ ]*/[ ]*" t)) { s=substr($0, RSTART, RLENGTH); sub(/[ ].*/, "", s); print s; exit } }')
  fi
  echo "$v"
}
pick_idle_card(){
  local c; for c in $CARDS; do
    if [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null; then echo "$c"; return 0; fi
  done
  return 1
}
wait_idle_card(){
  local i c; for i in $(seq 1 90); do
    c=$(pick_idle_card) && { echo "$c"; return 0; }
    sleep 20
  done
  return 1
}
wait_health(){
  local i; for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done; echo 0
}
stop_pgid(){ kill -- -"$1" 2>/dev/null || true; sleep 5; kill -9 -- -"$1" 2>/dev/null || true; sleep 3; }
residual_check(){
  local label=$1 pgid=$2 out
  sleep 3
  out=$(ps -eo pid,ppid,pgid,etime,args | awk -v g="$pgid" '$3==g && $1!=g' | grep -v defunct | head -20)
  if [ -n "$out" ]; then
    log "RESIDUAL_AFTER $label: survivors (force kill):"; echo "$out" | tee -a "$LOG"
    kill -9 -- -"$pgid" 2>/dev/null || true
  else
    log "residual $label: none"
  fi
}
write_cfg(){
  cat > "$ROOT/runtime_config.json" <<CFG
{"dump": {"dump_dir": "$ROOT/dump", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": $1}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true}, "spec_acceptance": {"enabled": true}}}
CFG
}
boot(){  # launch detached; echo pid of the api_server process group leader
  ( cd "$PRODUCT" && env PYTHONPATH="$V030:$PRODUCT:${PYTHONPATH:-}" \
      ASCEND_RT_VISIBLE_DEVICES=$1 VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
      setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name dsv2 --port "$PORT" \
      --gpu-memory-utilization 0.85 --enforce-eager \
      --additional-config "{\"runtime_config_path\": \"$ROOT/runtime_config.json\", \"runtime_config_hot_reload\": true, \"runtime_report_dir\": \"$ROOT/report\"}" \
      > "$ROOT/serve.log" 2>&1 & )
}

log "=== tip15 P0-3 TP1 smoke v3 start PRODUCT=$PRODUCT port=$PORT host=$(hostname 2>/dev/null) ==="
CARD=$(wait_idle_card) || { log "FAIL: no idle card"; log "TIP15_TP1_SMOKE_VERDICT FAIL"; exit 1; }
log "picked card=$CARD"
write_cfg false   # startup defaults; leader will persist its own copy anyway

boot "$CARD"
sleep 5
pgid=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port $PORT" | head -1)
[ -n "$pgid" ] || pgid=$(grep -oE "APIServer pid=[0-9]+" "$ROOT/serve.log" | tail -1 | grep -oE "[0-9]+")
log "boot pid=$pgid card=$CARD"
if [ -z "$pgid" ] || [ "$(wait_health)" != 1 ]; then
  fail "no pid or health timeout"
  grep -iE "error|exception|Traceback" "$ROOT/serve.log" | head -8 | tee -a "$LOG"
  [ -n "$pgid" ] && { stop_pgid "$pgid"; residual_check "tp1" "$pgid"; }
  log "TIP15_TP1_SMOKE_VERDICT FAIL"; exit 1
fi
log "health=200"

sleep 5          # let ensure_persisted() finish overwriting the JSON
write_cfg true   # toggle manual_dump ON via hot reload (3s poll)
sleep 6          # hot reload pickup window
log "manual_dump toggled on after startup (watermark-safe continuous mode)"

short=$(curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"dsv2","prompt":"用一句话介绍长城","max_tokens":64,"temperature":0,"seed":42}' | head -c 400)
echo "$short" > "$ROOT/short.json"
[ ${#short} -gt 50 ] && log "short completion OK" || fail "short completion suspicious"

curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"dsv2","prompt":"请连续输出60个哈字：哈哈哈哈","max_tokens":96,"temperature":0,"seed":42}' > "$ROOT/repeat.json"
sleep 10

"$PY" - "$ROOT" <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, os, sys
root = sys.argv[1]
man_pt = sorted(set(glob.glob(f"{root}/dump/**/*.pt", recursive=True) + glob.glob(f"{root}/report/kv_cache/**/*.pt", recursive=True)))
ranks = sorted({os.path.basename(os.path.dirname(p)) for p in man_pt})
print(f"DUMP_CHECK manual_pt={len(man_pt)} ranks={ranks}")
PYEOF

long=$(curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"dsv2","prompt":"写一篇150字的短文介绍李白","max_tokens":384,"temperature":0,"seed":42}')
echo "$long" > "$ROOT/long.json"
log "long resp_len=${#long}"

n_started=$(grep -c "bus worker started" "$ROOT/serve.log" 2>/dev/null || true)
[ "${n_started:-0}" = 1 ] 2>/dev/null || fail "expected exactly 1 bus worker, got ${n_started:-0}"
log "bus_worker started=${n_started:-0} stopped=0 (stopped==0 is normal under pgid kill flow)"

inc=$(find "$ROOT/report" -mindepth 2 -maxdepth 2 -name 'report_*.json' 2>/dev/null | wc -l | tr -d ' ')
[ "${inc:-0}" -ge 1 ] 2>/dev/null || fail "no incident report json"
log "incident_report_json=$inc"

npt=$(find "$ROOT/dump" "$ROOT/report" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
[ "${npt:-0}" -ge 1 ] 2>/dev/null || fail "no manual dump .pt produced"
log "manual_pt_count=$npt"

mis=$(grep -c "wave misalignment" "$ROOT/serve.log" 2>/dev/null || true)
[ "${mis:-0}" = 0 ] 2>/dev/null || fail "wave misalignment asserted"
tb=$(grep -c "Traceback" "$ROOT/serve.log" 2>/dev/null || true)
[ "${tb:-0}" = 0 ] 2>/dev/null || fail "traceback in serve log"

stop_pgid "$pgid"
residual_check "tp1" "$pgid"
h=$(hbm "$CARD"); [ -n "$h" ] && [ "$h" -lt 5000 ] && log "HBM released on card $CARD ($h)" || fail "HBM not released on card $CARD ($h)"

if [ "$PASS" = 1 ]; then log "TIP15_TP1_SMOKE_VERDICT PASS"; else log "TIP15_TP1_SMOKE_VERDICT FAIL"; fi
log "=== tip15 P0-3 TP1 smoke v3 done ==="
exit $((1 - PASS))
