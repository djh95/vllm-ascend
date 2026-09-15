#!/usr/bin/env bash
set -uo pipefail
# P0 smoke FULL launcher: launches server + curl + verify, 6 cases x (v2,v1).
# Machine-agnostic via env; defaults tuned for test-mrv2-cann91 (CANN 9.1 py3.12).
#   RG_PRODUCT_ROOT  product (config-branch) checkout
#   PY               python (must have vllm + vllm_ascend)
#   MODEL            model dir (small: Qwen2.5-0.5B-Instruct)
#   SERVED_MODEL_NAME served name
#   NPU_DEV          single NPU index for these cases
#   PORT             api port
#   RG_OUT_ROOT      output root (summary.txt + per-case report/dump/log)
# Results -> $RG_OUT_ROOT/summary.txt
PRODUCT="${RG_PRODUCT_ROOT:-/data0/test-mrv2-cann91/vllm-ascend}"
CFGDIR="${RG_CFGDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../configs" && pwd)}"
PY="${PY:-/opt/slime/venv/bin/python}"
MODEL="${MODEL:-/data0/weights/Qwen2.5-0.5B-Instruct}"
SNAME="${SERVED_MODEL_NAME:-qwen05}"
NPU="${NPU_DEV:-0}"
PORT="${PORT:-8031}"
ROOT="${RG_OUT_ROOT:-/tmp/rg_p0_smoke}"
mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.txt"
: > "$SUMMARY"

stop_serve() {
  kill "$1" 2>/dev/null || true
  pkill -f '[a]pi_server' 2>/dev/null || true
  sleep 2
  pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true
  pkill -9 -f 'VLLM::DPCoordinato[r]' 2>/dev/null || true
  pkill -9 -f '[V]LLMWorker' 2>/dev/null || true
  sleep 3
}

run_case() {
  local case=$1 cfg=$2 inject=$3 expect=$4 prompt=$5 mt=$6
  for r in v2 v1; do
    local D="$ROOT/${case}_${r}"
    mkdir -p "$D/report" "$D/dump"
    local log="$D/serve.log"
    echo "########## $case / $r ##########" | tee -a "$SUMMARY"
    local overlay
    overlay=$("$PY" - "$cfg" "$D/dump" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
c.setdefault("dump",{})["dump_dir"]=sys.argv[2]
print(json.dumps(c,ensure_ascii=False))
PY
    )
    cd "$PRODUCT"
    export PYTHONPATH="$PRODUCT:${PYTHONPATH:-}"
    export ASCEND_RT_VISIBLE_DEVICES="$NPU"
    export VLLM_BATCH_INVARIANT=1
    if [ "$r" = v2 ]; then export VLLM_USE_V2_MODEL_RUNNER=1; else export VLLM_USE_V2_MODEL_RUNNER=0; fi
    if [ -n "$inject" ]; then export RG_INJECT="$inject"; else unset RG_INJECT; fi
    nohup "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name "$SNAME" --port "$PORT" \
      --gpu-memory-utilization 0.85 \
      --enforce-eager \
      --additional-config "{\"runtime_config\": $overlay, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$D/runtime_config.json\", \"runtime_report_dir\": \"$D/report\"}" \
      > "$log" 2>&1 &
    local pid=$!
    local ok=0
    for i in $(seq 1 120); do
      if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; echo "[$case/$r] health OK ~${i}0s" | tee -a "$SUMMARY"; break; fi
      sleep 10
    done
    if [ "$ok" != 1 ]; then
      echo "[$case/$r] HEALTH TIMEOUT" | tee -a "$SUMMARY"
      tail -25 "$log" >> "$SUMMARY" 2>/dev/null
      stop_serve "$pid"
      continue
    fi
    curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SNAME\",\"prompt\":\"$prompt\",\"max_tokens\":$mt,\"temperature\":0,\"seed\":42}" \
      > "$D/response.json" 2>/dev/null || true
    sleep 6
    local nrep ndump
    nrep=$(find "$D/report" -name 'report_*.json' 2>/dev/null | wc -l)
    ndump=$(find "$D/dump" -name '*.pt' 2>/dev/null | wc -l)
    echo "[$case/$r] expect=$expect reports=$nrep dumps=$ndump" | tee -a "$SUMMARY"
    echo "--- key log ---" >> "$SUMMARY"
    grep -iE "runtime_guard|detector|report|dump|skip|INJECT|anomaly|incident|manual" "$log" 2>/dev/null | tail -20 >> "$SUMMARY"
    echo "--- response head ---" >> "$SUMMARY"
    head -c 150 "$D/response.json" 2>/dev/null >> "$SUMMARY"; echo >> "$SUMMARY"
    stop_serve "$pid"
  done
}

run_case p0_01_guard_off    "$CFGDIR/p0_01_guard_off.json"            ""           none   "ping"        8
run_case p0_05_inject_nan   "$CFGDIR/p0_05_inject_nan.json"           nan_logits   report "hello"      32
run_case g02_inf_logits     "$CFGDIR/g02_inf_logits.json"             inf_logits   report "hello"      32
run_case g03_forbidden      "$CFGDIR/g03_forbidden_substring.json"    "forbidden_substring:5" report "hello" 32
run_case g04_token_loop     "$CFGDIR/p0_02_token_repeat.json"         token_loop   report "hello"      64
run_case p0_03_manual_dump  "$CFGDIR/p0_03_manual_dump.json"          ""           dump   "hello dump" 16

echo "ALL P0 SMOKE DONE" >> "$SUMMARY"
