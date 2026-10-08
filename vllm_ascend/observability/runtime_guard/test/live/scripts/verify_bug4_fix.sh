#!/usr/bin/env bash
# Verify BUG-4 fix (03bfd2a54) on real NPU: v2 dump_kv should now produce .pt.
# Pinned to cards 3,4 (away from W1/W2 workers on 0,1,2).
# Cases: manual_dump v2 TP=1, manual_dump v2 TP=2, auto nan v2 TP=1, manual v1 TP=1 (control).
set -uo pipefail

PRODUCT=/data0/test-mrv2-cann91/vllm-ascend
CFGDIR=/data0/test-mrv2-cann91/vllm-ascend/vllm_ascend/runtime_guard/test/live/configs
PY=/opt/slime/venv/bin/python
MODEL=/data0/weights/Qwen2.5-0.5B-Instruct
SNAME=qwen05
PORT=8093
ROOT=/tmp/rg_verify_bug4fix
mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.txt"
: > "$SUMMARY"

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$SUMMARY"; }

stop_own() {
  local pgid=$1
  kill -- -"$pgid" 2>/dev/null || true
  sleep 3
  kill -9 -- -"$pgid" 2>/dev/null || true
  sleep 2
}

mkcfg_manual() {
  "$PY" - "$CFGDIR/p0_03_manual_dump.json" "$1" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
c.setdefault("dump",{})["dump_dir"]=sys.argv[2]
print(json.dumps(c,ensure_ascii=False))
PY
}

mkcfg_auto() {
  "$PY" - "$1" <<'PY'
import json,sys
dumpdir=sys.argv[1]
cfg={
  "reload_interval_seconds": 3,
  "dump": {"auto_max_times": 1, "manual_dump": False, "dump_dir": dumpdir},
  "actions": {"defaults": {"on_trigger": ["report", "dump_kv"]}},
  "detector": {
    "logits_finite": {"enabled": True},
    "token_repeat": {"enabled": False},
    "output_substring": {"enabled": False, "patterns": ["李白"], "match_prefix": False},
    "spec_acceptance": {"enabled": False},
  },
}
print(json.dumps(cfg,ensure_ascii=False))
PY
}

run_case() {
  local tag=$1 tp=$2 runner=$3 cards=$4 mode=$5
  local D="$ROOT/${tag}"
  mkdir -p "$D/report" "$D/dump"
  local log="$D/serve.log"
  local overlay
  if [ "$mode" = auto ]; then
    overlay=$(mkcfg_auto "$D/dump")
    export RG_INJECT="nan_logits"
  else
    overlay=$(mkcfg_manual "$D/dump")
    unset RG_INJECT
  fi
  cd "$PRODUCT"
  export PYTHONPATH="$PRODUCT:${PYTHONPATH:-}"
  export ASCEND_RT_VISIBLE_DEVICES="$cards"
  export VLLM_BATCH_INVARIANT=1
  if [ "$runner" = v2 ]; then export VLLM_USE_V2_MODEL_RUNNER=1; else export VLLM_USE_V2_MODEL_RUNNER=0; fi
  setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SNAME" --port "$PORT" \
    --tensor-parallel-size "$tp" --gpu-memory-utilization 0.85 --enforce-eager \
    --additional-config "{\"runtime_config\": $overlay, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$D/runtime_config.json\", \"runtime_report_dir\": \"$D/report\"}" \
    > "$log" 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" != 1 ]; then
    log "[$tag] tp=$tp $runner HEALTH_TIMEOUT"
    tail -25 "$log" >> "$SUMMARY" 2>/dev/null
    stop_own "$pid"
    return
  fi
  curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SNAME\",\"prompt\":\"hello dump\",\"max_tokens\":16,\"temperature\":0,\"seed\":42}" \
    > "$D/response.json" 2>/dev/null || true
  sleep 8
  local nrep ndump nskip
  nrep=$(find "$D/report" -name 'report_*.json' 2>/dev/null | wc -l)
  ndump=$(find "$D/dump" -name '*.pt' 2>/dev/null | wc -l)
  nskip=$(find "$D/dump" -name 'dump_skipped.json' 2>/dev/null | wc -l)
  log "[$tag] tp=$tp $runner reports=$nrep dumps=$ndump skipped=$nskip"
  grep -iE "dump|skip|empty_block|d2h|block_ids|req_states|error" "$log" 2>/dev/null | tail -10 >> "$SUMMARY"
  stop_own "$pid"
}

log "cards: 3,4 (pinned)"
run_case manual_tp1_v2 1 v2 3 manual
run_case manual_tp1_v1 1 v1 3 manual
run_case manual_tp2_v2 2 v2 "3,4" manual
run_case auto_nan_v2 1 v2 3 auto
log "ALL DONE"
