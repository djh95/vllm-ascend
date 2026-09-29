#!/usr/bin/env bash
# Per-wave timestamp probe session (T2 only, ~100s traffic) for tip15 root-cause:
# collect [RG-BUS-STATS] + [RG-BUS-WAVE] per-rank wall-clock timelines.
set -uo pipefail
PY=/opt/slime/venv/bin/python
V030=/data0/test-mrv2-cann91/vllm030_pkgs
TREE=/data0/test-mrv2-cann91/rg-tip15-8f5e3
PERF_DIR=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/test/perf
ROOT=/data0/test-mrv2-cann91/rg_probe_tip15
LOG=$ROOT/master.log
PORT=8173
MODEL=/data0/weights/Qwen2.5-7B-Instruct
CARDS=4,5
mkdir -p "$ROOT"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }
hbm(){ npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+"; }
wait_idle_cards(){
  for i in $(seq 1 90); do
    ok=1
    for c in 4 5; do [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null || { ok=0; break; }; done
    [ "$ok" = 1 ] && return 0
    sleep 20
  done
  return 1
}

cat > "$ROOT/cfg_t2.json" <<EOF
{"reload_interval_seconds": 3, "dump": {"dump_dir": "$ROOT/dump_t2", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "detector": {"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "output_substring": {"enabled": false}, "spec_acceptance": {"enabled": false}}}
EOF

log "=== probe session: waiting idle cards $CARDS ==="
wait_idle_cards || { log "ABORT: cards never idle"; exit 1; }
( cd "$TREE"
  env PYTHONPATH="$V030:$TREE:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=$CARDS \
    VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name dsv2 --port "$PORT" --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --enforce-eager \
    --additional-config "{\"runtime_config_path\": \"$ROOT/cfg_t2.json\", \"runtime_config_reload_interval\": 3, \"runtime_report_dir\": \"$ROOT/report_t2\"}" \
    > "$ROOT/serve_t2.log" 2>&1 & echo $! )
log "serve port=$PORT (stopped via --port pattern)"
ok=0
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  sleep 10
done
[ "$ok" = 1 ] || { log "HEALTH_FAIL (see serve_t2.log)"; pkill -f -- "--port $PORT"; sleep 3; pkill -9 -f -- "--port $PORT"; exit 1; }
log "health=200, warmup"
( cd "$PERF_DIR" && RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY'
from perf_lib import warmup
warmup(rounds=1)
PY
)
log "warmup done, driving 100s steady traffic"
( cd "$PERF_DIR" && RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" RG_DURATION=100 PYTHONPATH="$PERF_DIR" "$PY" - <<'PY'
import os, time
from perf_lib import post, REQS
end = time.time() + float(os.environ["RG_DURATION"])
i = 0
while time.time() < end:
    tag, p, mt = REQS[i % len(REQS)]
    post(p, mt)
    i += 1
print(f"probe traffic: sent {i} requests")
PY
)
log "traffic done, stopping serve"
pkill -f -- "--port $PORT"; sleep 3; pkill -9 -f -- "--port $PORT"; sleep 5
log "wave_lines=$(grep -c 'RG-BUS-WAVE' "$ROOT/serve_t2.log" 2>/dev/null) stats_lines=$(grep -c 'RG-BUS-STATS' "$ROOT/serve_t2.log" 2>/dev/null)"
log "=== probe session done ==="
