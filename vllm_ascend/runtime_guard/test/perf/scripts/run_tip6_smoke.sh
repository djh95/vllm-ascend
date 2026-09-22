#!/usr/bin/env bash
set -uo pipefail
PY=/opt/slime/venv/bin/python
V029=/data0/test-mrv2-cann91/vllm029_pkgs
PRODUCT=/data0/test-mrv2-cann91/rg-tip6-verify
MODEL=/data0/weights/DeepSeek-V2-Lite
ROOT=/data0/test-mrv2-cann91/rg_tip6_smoke
LOG=$ROOT/master.log
mkdir -p "$ROOT"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

hbm(){ npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+"; }
wait_idle(){
  local i h2 h3
  for i in $(seq 1 120); do
    h2=$(hbm 2); h3=$(hbm 3)
    if [ -n "$h2" ] && [ -n "$h3" ] && [ "$h2" -lt 5000 ] && [ "$h3" -lt 5000 ]; then return 0; fi
    sleep 30
  done
  return 1
}
wait_release(){
  local i h2
  for i in $(seq 1 20); do
    h2=$(hbm 2)
    if [ -n "$h2" ] && [ "$h2" -lt 5000 ]; then return 0; fi
    sleep 15
  done
  return 1
}
wait_health(){
  local port=$1 i
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done
  echo 0
}
write_cfg(){
  local path=$1 dumpdir=$2 md=$3
  cat > "$path" <<CFG
{"detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 32, "consecutive_hits": 1}, "output_substring": {"enabled": true, "patterns": ["ZZQQXXYY_NEVER_MATCH"]}, "spec_acceptance": {"enabled": true}}, "actions": {"defaults": {"on_trigger": ["report"]}}, "dump": {"dump_dir": "$dumpdir", "auto_max_times": 0, "manual_dump": $md}}
CFG
}

run_arm_once(){
  local runner=$1 port=$2
  local arm="$ROOT/$runner" RP=""
  [ "$runner" = v2 ] && RP="VLLM_USE_V2_MODEL_RUNNER=1"
  mkdir -p "$arm/report" "$arm/dump"
  ( cd "$PRODUCT"
    env PYTHONPATH="$V029:$PRODUCT:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=2,3 VLLM_BATCH_INVARIANT=1 $RP \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name t5 --port "$port" \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.80 --enforce-eager \
      --additional-config "{\"runtime_config\": {\"detector\": {\"logits_finite\": {\"enabled\": true}, \"token_repeat\": {\"enabled\": true, \"window\": 32, \"repeat_sum_threshold\": 64, \"min_tokens\": 32, \"consecutive_hits\": 1}, \"output_substring\": {\"enabled\": true, \"patterns\": [\"ZZQQXXYY_NEVER_MATCH\"]}, \"spec_acceptance\": {\"enabled\": true}}, \"actions\": {\"defaults\": {\"on_trigger\": [\"report\"]}}, \"dump\": {\"dump_dir\": \"$arm/dump\", \"auto_max_times\": 0, \"manual_dump\": 0}}, \"runtime_config_reload_interval\": 2, \"runtime_config_path\": \"$arm/runtime_config.json\", \"runtime_report_dir\": \"$arm/report\"}" \
      > "$arm/serve.log" 2>&1 & echo $! ) > "$ROOT/pid_${runner}.txt"
  local pid; pid=$(cat "$ROOT/pid_${runner}.txt")
  log "[$runner] boot pid=$pid"
  local ok; ok=$(wait_health "$port")
  log "[$runner] health=$ok"
  if [ "$ok" != 1 ]; then
    log "[$runner] HEALTH TIMEOUT; serve.log tail:"
    tail -30 "$arm/serve.log" | tee -a "$LOG"
    kill -- -"$pid" 2>/dev/null || true; sleep 3; kill -9 -- -"$pid" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCor[e]" 2>/dev/null || true
    return 1
  fi
  local out
  out=$(curl -s --max-time 120 http://127.0.0.1:$port/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"t5\",\"prompt\":\"Once upon a time in a distant land\",\"max_tokens\":24,\"temperature\":0}")
  echo "$out" > "$arm/completions.json"
  log "[$runner] completion resp_len=${#out}"
  curl -s --max-time 300 http://127.0.0.1:$port/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"t5\",\"prompt\":\"Write a long story about the sea, ships and sailors crossing the ocean\",\"max_tokens\":400,\"temperature\":0}" > "$arm/long.json" &
  local cpid=$!
  sleep 6
  write_cfg "$arm/runtime_config.json" "$arm/dump" 1
  sleep 15
  local nf; nf=$(find "$arm/dump" -type f 2>/dev/null | wc -l)
  log "[$runner] manual_dump files=$nf"
  wait "$cpid" || true
  log "[$runner] long resp_len=$(wc -c < "$arm/long.json" 2>/dev/null)"
  write_cfg "$arm/runtime_config.json" "$arm/dump" 0
  sleep 4
  log "[$runner] guard_log_lines=$(grep -c runtime_guard "$arm/serve.log")"
  grep -iE "reload.*(fail|error)|Traceback" "$arm/serve.log" | head -5 | tee -a "$LOG"
  kill -- -"$pid" 2>/dev/null || true; sleep 5; kill -9 -- -"$pid" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCor[e]" 2>/dev/null || true
  wait_release && log "[$runner] HBM released" || log "[$runner] HBM still held!"
}

run_arm_with_retry(){
  local runner=$1 port=$2 attempt
  for attempt in 1 2; do
    if wait_idle; then
      log "[$runner] attempt $attempt: cards idle"
      run_arm_once "$runner" "$port" && return 0
      log "[$runner] attempt $attempt failed"
    else
      log "[$runner] wait_idle timeout attempt $attempt"
    fi
  done
  return 1
}

log "=== tip6 smoke start PRODUCT=$PRODUCT vllm029 tip=9e314a780 ==="
run_arm_with_retry v2 8141
run_arm_with_retry v1 8140
log "=== DONE ==="
