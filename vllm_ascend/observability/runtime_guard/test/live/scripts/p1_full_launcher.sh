#!/usr/bin/env bash
set -uo pipefail
# P1 full functional FULL launcher (feasible cases):
#   p0_04 tp_rank_gate (TP=2, v2+v1) / p0_07 dump_schema (v1) / g06 bad_reload / p0_08 disk_reclaim
# Machine-agnostic via env; defaults tuned for test-mrv2-cann91 (CANN 9.1 py3.12).
#   RG_PRODUCT_ROOT  product (config-branch) checkout
#   RG_ANALYSIS_ROOT analysis-branch checkout (for verify_request_kv / inspect_kv_dump)
#   PY / MODEL / SERVED_MODEL_NAME
#   NPU_DEV          two NPU ids for TP=2 (e.g. "0,1")
# Results -> $RG_OUT_ROOT/summary.txt
PRODUCT="${RG_PRODUCT_ROOT:-/data0/test-mrv2-cann91/vllm-ascend}"
ANALYSIS="${RG_ANALYSIS_ROOT:-/data0/test-mrv2-cann91/vllm-ascend}"
CFGDIR="${RG_CFGDIR:-$ANALYSIS/vllm_ascend/runtime_guard/test/live/configs}"
PY="${PY:-/opt/slime/venv/bin/python}"
MODEL="${MODEL:-/data0/weights/Qwen2.5-0.5B-Instruct}"
SNAME="${SERVED_MODEL_NAME:-qwen05}"
ROOT="${RG_OUT_ROOT:-/tmp/rg_p1_full}"
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

wait_health() {
  local port=$1
  for i in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then echo 1; return; fi
    sleep 10
  done
  echo 0
}

# launch <case> <cfg> <inject> <prompt> <mt> <npu_list> <tp_size> <port> <runner(v1|v2)>
launch() {
  local case=$1 cfg=$2 inject=$3 prompt=$4 mt=$5 npu=$6 tp=$7 port=$8 runner=$9
  local D="$ROOT/$case"
  mkdir -p "$D/report" "$D/dump"
  local log="$D/serve.log"
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
  export ASCEND_RT_VISIBLE_DEVICES="$npu"
  export VLLM_BATCH_INVARIANT=1
  if [ "$runner" = v1 ]; then export VLLM_USE_V2_MODEL_RUNNER=0; else export VLLM_USE_V2_MODEL_RUNNER=1; fi
  if [ -n "$inject" ]; then export RG_INJECT="$inject"; else unset RG_INJECT; fi
  local tpflag=""
  if [ "$tp" -gt 1 ]; then tpflag="--tensor-parallel-size $tp"; fi
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SNAME" --port "$port" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager $tpflag \
    --additional-config "{\"runtime_config\": $overlay, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$D/runtime_config.json\", \"runtime_report_dir\": \"$D/report\"}" \
    > "$log" 2>&1 &
  local pid=$!
  local ok=$(wait_health "$port")
  echo "$pid $ok"
}

# ============ p0_04_tp_rank_gate (TP=2, v2 then v1) ============
for r in v2 v1; do
  A="p0_04_tp_rank_gate_${r}"
  echo "########## p0_04_tp_rank_gate / TP=2 / $r ##########" | tee -a "$SUMMARY"
  read -r APID AOK < <(launch "$A" "$CFGDIR/p0_05_inject_nan.json" "nan_logits" "hello" 32 "0,1" 2 8032 "$r")
  echo "[$A] pid=$APID health=$AOK" | tee -a "$SUMMARY"
  if [ "$AOK" = "1" ]; then
    curl -s "http://127.0.0.1:8032/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SNAME\",\"prompt\":\"hello\",\"max_tokens\":32,\"temperature\":0,\"seed\":42}" \
      > "$ROOT/$A/response.json" 2>/dev/null || true
    sleep 6
    nrep=$(find "$ROOT/$A/report" -name 'report_*.json' 2>/dev/null | wc -l)
    echo "[$A] reports=$nrep (expect=1 from TP0 only)" | tee -a "$SUMMARY"
    for f in "$ROOT/$A/report"/report_*.json; do
      [ -f "$f" ] || continue
      echo "  rank=$( "$PY" -c "import json;print(json.load(open('$f')).get('rank'))" 2>/dev/null ) incident=$( "$PY" -c "import json;print(json.load(open('$f')).get('incident_type'))" 2>/dev/null )" | tee -a "$SUMMARY"
    done
    echo "--- key log ---" >> "$SUMMARY"
    grep -iE "not TP0|rank|skip|detector|report|anomaly|INJECT" "$ROOT/$A/serve.log" 2>/dev/null | tail -15 >> "$SUMMARY"
    stop_serve "$APID"
  else
    echo "[$A] HEALTH TIMEOUT" | tee -a "$SUMMARY"; tail -25 "$ROOT/$A/serve.log" >> "$SUMMARY"; stop_serve "$APID"
  fi
done

# ============ p0_07_dump_schema (v1 manual_dump + verify) ============
B="p0_07_dump_schema"
echo "########## p0_07_dump_schema / v1 ##########" | tee -a "$SUMMARY"
read -r BPID BOK < <(launch "$B" "$CFGDIR/p0_03_manual_dump.json" "" "hello dump" 16 "0" 1 8033 v1)
echo "[$B] pid=$BPID health=$BOK" | tee -a "$SUMMARY"
if [ "$BOK" = "1" ]; then
  curl -s "http://127.0.0.1:8033/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SNAME\",\"prompt\":\"hello dump\",\"max_tokens\":16,\"temperature\":0,\"seed\":42}" \
    > "$ROOT/$B/response.json" 2>/dev/null || true
  sleep 8
  ndump=$(find "$ROOT/$B/dump" -name '*.pt' 2>/dev/null | wc -l)
  nrep=$(find "$ROOT/$B/report" -name 'report_*.json' 2>/dev/null | wc -l)
  echo "[$B] reports=$nrep dumps=$ndump" | tee -a "$SUMMARY"
  rep=$(find "$ROOT/$B/report" -name 'report_*.json' 2>/dev/null | head -1)
  if [ -n "$rep" ]; then
    echo "--- verify_request_kv ---" | tee -a "$SUMMARY"
    # NOTE: analysis scripts must run with PYTHONPATH=$ANALYSIS (analysis-first). Putting
    # $PRODUCT first shadows vllm_ascend.runtime_guard with the config-branch package, which
    # has no `analysis` subpackage -> ModuleNotFoundError.
    (cd "$ANALYSIS" && PYTHONPATH="$ANALYSIS" "$PY" -m vllm_ascend.runtime_guard.analysis.scripts.verify_request_kv \
      --report "$rep" --report-dir "$ROOT/$B/report" 2>&1 | tail -25) >> "$SUMMARY"
    pt=$(find "$ROOT/$B/dump" -name '*.pt' 2>/dev/null | head -1)
    if [ -n "$pt" ]; then
      echo "--- inspect_kv_dump ---" | tee -a "$SUMMARY"
      (cd "$ANALYSIS" && PYTHONPATH="$ANALYSIS" "$PY" -m vllm_ascend.runtime_guard.analysis.scripts.inspect_kv_dump --path "$pt" 2>&1 | tail -15) >> "$SUMMARY"
    fi
  fi
  stop_serve "$BPID"
else
  echo "[$B] HEALTH TIMEOUT" | tee -a "$SUMMARY"; tail -25 "$ROOT/$B/serve.log" >> "$SUMMARY"; stop_serve "$BPID"
fi

# ============ g06_bad_reload ============
C="g06_bad_reload"
echo "########## g06_bad_reload ##########" | tee -a "$SUMMARY"
read -r CPID COK < <(launch "$C" "$CFGDIR/p0_01_guard_off.json" "" "g06" 4 "0" 1 8034 v2)
echo "[$C] pid=$CPID health=$COK" | tee -a "$SUMMARY"
if [ "$COK" = "1" ]; then
  CFG_LIVE="$ROOT/$C/runtime_config.json"
  cp "$CFG_LIVE" "$CFG_LIVE.bak.$$"
  printf '{ this is not valid json\n' > "$CFG_LIVE"
  sleep 6
  hc=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8034/health" 2>/dev/null || echo "000")
  echo "[$C] after bad JSON, health=$hc (expect 200, service alive)" | tee -a "$SUMMARY"
  curl -s "http://127.0.0.1:8034/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SNAME\",\"prompt\":\"g06\",\"max_tokens\":4,\"temperature\":0}" \
    > "$ROOT/$C/response.json" 2>/dev/null || true
  echo "--- key log (expect reject/unknown, keep old config) ---" >> "$SUMMARY"
  grep -iE "reload|reject|unknown|invalid|json|config|runtime_guard" "$ROOT/$C/serve.log" 2>/dev/null | tail -20 >> "$SUMMARY"
  mv "$CFG_LIVE.bak.$$" "$CFG_LIVE"
  echo "[$C] restored config" | tee -a "$SUMMARY"
  stop_serve "$CPID"
else
  echo "[$C] HEALTH TIMEOUT" | tee -a "$SUMMARY"; tail -25 "$ROOT/$C/serve.log" >> "$SUMMARY"; stop_serve "$CPID"
fi

# ============ p0_08_disk_reclaim ============
echo "########## p0_08_disk_reclaim ##########" | tee -a "$SUMMARY"
D="$ROOT/p0_08_disk_reclaim"
mkdir -p "$D/_p0_08_probe"
echo probe > "$D/_p0_08_probe/x.txt"
df -h "$D" | tail -1 | tee -a "$SUMMARY"
rm -rf "$D/_p0_08_probe"
if [ ! -e "$D/_p0_08_probe" ]; then echo "[p0_08] PASS — probe dir removed" | tee -a "$SUMMARY"; else echo "[p0_08] FAIL" | tee -a "$SUMMARY"; fi

echo "ALL P1 FULL DONE" >> "$SUMMARY"
