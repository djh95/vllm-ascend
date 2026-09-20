#!/usr/bin/env bash
set -uo pipefail
# C5 v2-only pass (2026-09-20): dump D2H+save timing on the v2 model runner,
# arm A only, on the 1ae55b060 verify worktree (newest tip whose v2 boots on
# this container's vllm 0.28.0; runtime_guard content is byte-identical to
# remote tip bc0629421). Cards 2,3 shared -> wait up to 60min for idle.
PY=/opt/slime/venv/bin/python
ENVDIR=/data0/test-mrv2-cann91/rg_c56
PRODUCT=/data0/test-mrv2-cann91/rg-rebase-verify
SCRIPTS=$ENVDIR/scripts
CORPUS=$ENVDIR/data/longbench_corpus.jsonl
MODEL=/data0/weights/DeepSeek-V2-Lite
ROOT=$ENVDIR/run_v2
RES=$ENVDIR/results/c5_v2_1ae55b060_$(date +%m%d_%H%M)
TIERS=1024,4096,16384,65536,131072
mkdir -p "$RES" "$ROOT/A/report" "$ROOT/A/dump"
LOG=$ROOT/master.log
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

wait_idle(){
  local i out h2 h3
  for i in $(seq 1 120); do
    out=$(npu-smi info 2>/dev/null || true)
    # process table clear != HBM released (driver lags / neighbor jobs cycle
    # on these shared cards); require HBM < 5GB on both cards.
    h2=$(npu-smi info 2>/dev/null | grep -A1 "^| 2     910" | grep -oE "[0-9]+ / 32768" | tail -1 | grep -oE "^[0-9]+")
    h3=$(npu-smi info 2>/dev/null | grep -A1 "^| 3     910" | grep -oE "[0-9]+ / 32768" | tail -1 | grep -oE "^[0-9]+")
    if [ -n "$h2" ] && [ -n "$h3" ] && [ "$h2" -lt 5000 ] && [ "$h3" -lt 5000 ]; then
      return 0
    fi
    sleep 30
  done
  return 1
}
wait_health(){
  local i
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:8130/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done
  echo 0
}

log "=== C5 v2-only start PRODUCT=$PRODUCT model=$MODEL ==="
wait_idle || { log "cards 2,3 busy after 60min, abort"; exit 1; }
log "cards idle, booting guard arm (v2)"
( cd "$PRODUCT";
  PYTHONPATH="$SCRIPTS/c5_shim:$PRODUCT:${PYTHONPATH:-}" \
  RG_C5_TIMING=1 RG_C5_TIMING_OUT=$ROOT/c5_timing.jsonl \
  ASCEND_RT_VISIBLE_DEVICES=2,3 \
  VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name c56 --port 8130 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.80 --enforce-eager \
    --additional-config "{\"runtime_config\": {\"detector\": {\"logits_finite\": {\"enabled\": true}, \"token_repeat\": {\"enabled\": true, \"window\": 32, \"repeat_sum_threshold\": 64, \"min_tokens\": 32, \"consecutive_hits\": 1}, \"output_substring\": {\"enabled\": true, \"patterns\": [\"ZZQQXXYY_NEVER_MATCH\"]}, \"spec_acceptance\": {\"enabled\": true}}, \"actions\": {\"defaults\": {\"on_trigger\": [\"report\"]}}, \"dump\": {\"dump_dir\": \"$ROOT/A/dump\", \"auto_max_times\": 0, \"manual_dump\": 0}}, \"runtime_config_reload_interval\": 2, \"runtime_config_path\": \"$ROOT/A/runtime_config.json\", \"runtime_report_dir\": \"$ROOT/A/report\"}" \
    > "$ROOT/A/serve.log" 2>&1 & echo $! ) > "$ROOT/pid.txt"
PID_A=$(cat "$ROOT/pid.txt")
log "boot pid=$PID_A"
OK_A=$(wait_health)
log "health=$OK_A"
if [ "$OK_A" != 1 ]; then
  log "HEALTH TIMEOUT, serve.log tail:"
  tail -30 "$ROOT/A/serve.log" >> "$LOG" 2>/dev/null
  kill -- -"$PID_A" 2>/dev/null || true; sleep 3; kill -9 -- -"$PID_A" 2>/dev/null || true
  exit 1
fi

log "--- C5 dump timing (v2, tiers $TIERS) ---"
RG_C56_CORPUS="$CORPUS" RG_C56_MODEL="$MODEL" "$PY" "$SCRIPTS/c56_driver.py" c5 \
  --port 8130 --config "$ROOT/A/runtime_config.json" \
  --timing "$ROOT/c5_timing.jsonl" --tiers "$TIERS" | tee "$RES/c5_groups.jsonl"
log "C5 v2 done"

kill -- -"$PID_A" 2>/dev/null || true; sleep 3; kill -9 -- -"$PID_A" 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true
cp "$ROOT/c5_timing.jsonl" "$RES/c5_timing_raw.jsonl" 2>/dev/null || true
du -sh "$ROOT/A/dump" >> "$RES/verdict.txt" 2>/dev/null || true

"$PY" - "$RES" <<'PYVERDICT' | tee "$RES/verdict.txt"
import json, sys, os
res = sys.argv[1]
print("== C5 dump timing (v2 runner, 1ae55b060) ==")
try:
    rows = [json.loads(l) for l in open(os.path.join(res, "c5_groups.jsonl")) if l.strip()]
except Exception:
    rows = []
for r in rows:
    if "error" in r:
        print(f"tier={r['tier']} rep={r['rep']} ERROR {r['error']}"); continue
    for tag, v in sorted((r.get("ranks") or {}).items()):
        print(f"tier={r['tier']:>7} rep={r['rep']} rank={tag} prompt_tok={r['prompt_tokens']} "
              f"D2H={v.get('d2h_ms')}ms save={v.get('save_ms', 0)}ms "
              f"layers={v.get('layers')} bytes={v.get('d2h_bytes', 0)/2**20:.0f}MiB "
              f"files={v.get('save_files', 0)}")
PYVERDICT
log "=== DONE results in $RES ==="
