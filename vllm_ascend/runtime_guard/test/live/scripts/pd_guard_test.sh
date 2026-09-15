#!/usr/bin/env bash
# §6 PD-01..PD-03: runtime_guard hooks on BOTH P and D sides in 1P1D.
# P(card0, kv_producer) + D(card2, kv_consumer) + proxy. nan inject on D (decode samples).
# Expect: both boot (hooks no crash), request OK, D report>=1 + dump>=1, P report=0 + dump=0.
set -uo pipefail
PRODUCT=/data0/test-mrv2-cann91/rg-config-review
PROXY_DIR=/data0/test-mrv2-cann91/vllm-ascend/examples/disaggregated_prefill_v1
CFGDIR=/data0/test-mrv2-cann91/vllm-ascend/vllm_ascend/runtime_guard/test/live/configs
PY=/opt/slime/venv/bin/python
MODEL=/data0/weights/Qwen2.5-0.5B-Instruct
SNAME=qwen05
ROOT=/tmp/rg_pd_guard
rm -rf "$ROOT"; mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.txt"; : > "$SUMMARY"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$SUMMARY"; }

stop_own(){ local pid=$1; kill -- -"$pid" 2>/dev/null; sleep 3; kill -9 -- -"$pid" 2>/dev/null; sleep 2; }
wait_health(){ local port=$1; for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return; }; sleep 5; done; echo 0; }
mkcfg(){ "$PY" - "$1" "$2" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
c.setdefault("dump",{})["dump_dir"]=sys.argv[2]
print(json.dumps(c,ensure_ascii=False))
PY
}

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=localhost
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

KVC='{"kv_connector": "MooncakeConnectorV1", "kv_buffer_device": "npu", "kv_connector_extra_config": {"prefill": {"dp_size": 1, "tp_size": 1}, "decode": {"dp_size": 1, "tp_size": 1}}}'

serve_pd(){
  local label=$1 card=$2 port=$3 role=$4 kvport=$5 cfg=$6 inject=${7:-}
  local D="$ROOT/$label"; mkdir -p "$D/report" "$D/dump"
  cd "$PRODUCT"
  export PYTHONPATH="$PRODUCT:${PYTHONPATH:-}"
  export ASCEND_RT_VISIBLE_DEVICES="$card"
  export VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1
  if [ -n "$inject" ]; then export RG_INJECT="$inject"; else unset RG_INJECT; fi
  local kvcfg; kvcfg=$(printf '%s' "$KVC" | "$PY" -c "import json,sys; c=json.load(sys.stdin); c['kv_role']='$role'; c['kv_port']='$kvport'; print(json.dumps(c))")
  setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SNAME" \
    --host 127.0.0.1 --port "$port" --no-enable-prefix-caching \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --enforce-eager \
    --kv-transfer-config "$kvcfg" \
    --additional-config "{\"runtime_config\": $cfg, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$D/runtime_config.json\", \"runtime_report_dir\": \"$D/report\"}" \
    > "$D/serve.log" 2>&1 &
  echo $!
}

ndump(){ find "$1" -name '*.pt' 2>/dev/null | wc -l; }
nrep(){ find "$1" -name 'report_*.json' 2>/dev/null | wc -l; }

CFG_P=$(mkcfg "$CFGDIR/k02_on_trigger_dump.json" "$ROOT/prefiller/dump")
CFG_D=$(mkcfg "$CFGDIR/k02_on_trigger_dump.json" "$ROOT/decoder/dump")

log "===== §6 PD-01..03 (runtime_guard on P card0 + D card2, nan inject on D) ====="
PID_P=$(serve_pd prefiller 0 13700 kv_producer 30000 "$CFG_P")
PID_D=$(serve_pd decoder 2 13701 kv_consumer 30100 "$CFG_D" nan_logits)
log "pids P=$PID_P D=$PID_D"
OKP=$(wait_health 13700); OKD=$(wait_health 13701)
log "health P=$OKP D=$OKD"
if [ "$OKP" = 1 ] && [ "$OKD" = 1 ]; then
  cd "$PROXY_DIR"
  PYTHONPATH="$PRODUCT" "$PY" load_balance_proxy_server_example.py \
    --host 127.0.0.1 --port 8080 \
    --prefiller-hosts 127.0.0.1 --prefiller-ports 13700 \
    --decoder-hosts 127.0.0.1 --decoder-ports 13701 \
    > "$ROOT/proxy.log" 2>&1 &
  PID_X=$!
  log "proxy pid=$PID_X"
  sleep 8
  curl -s http://127.0.0.1:8080/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SNAME\",\"prompt\":\"hello pd guard\",\"max_tokens\":16,\"temperature\":0,\"seed\":42}" \
    > "$ROOT/resp.json" 2>/dev/null || true
  sleep 8
  RP=$(nrep "$ROOT/prefiller/report"); DP=$(ndump "$ROOT/prefiller/dump")
  RD=$(nrep "$ROOT/decoder/report"); DD=$(ndump "$ROOT/decoder/dump")
  reqok=0; grep -q '"choices"' "$ROOT/resp.json" 2>/dev/null && reqok=1
  log "[PD-01/02/03] request_ok=$reqok P reports=$RP dumps=$DP (expect 0/0)  D reports=$RD dumps=$DD (expect >=1/>=1)"
  log "--- D report fields ---"
  f=$(find "$ROOT/decoder/report" -name 'report_*.json' 2>/dev/null | head -1)
  [ -n "$f" ] && "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); print(sorted(r.keys()))" "$f" >> "$SUMMARY"
  log "--- P tail ---"; tail -8 "$ROOT/prefiller/serve.log" >> "$SUMMARY"
  log "--- D tail ---"; tail -8 "$ROOT/decoder/serve.log" >> "$SUMMARY"
  stop_own "$PID_X"; stop_own "$PID_P"; stop_own "$PID_D"
else
  log "PD_HEALTH_TIMEOUT"; tail -25 "$ROOT/prefiller/serve.log" >> "$SUMMARY"; tail -25 "$ROOT/decoder/serve.log" >> "$SUMMARY"
  stop_own "$PID_P"; stop_own "$PID_D"
fi
log "PD GUARD DONE"
