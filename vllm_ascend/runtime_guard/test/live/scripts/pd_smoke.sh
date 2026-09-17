#!/usr/bin/env bash
# PD 1P1D single-node smoke: P(card0)+D(card2)+proxy, Qwen2.5-0.5B.
# NOTE: D on card2, not card1 — card1's RDMA/HCCS network port is down
# (mooncake AdxlEngine "Device 1 transport init error: network port is down").
# Validates mooncake NPU wheel + proxy + P2P KV transfer end-to-end.
set -uo pipefail
PRODUCT=/data0/test-mrv2-cann91/vllm-ascend
PROXY_DIR=/data0/test-mrv2-cann91/vllm-ascend/examples/disaggregated_prefill_v1
PY=/opt/slime/venv/bin/python
MODEL=/data0/weights/Qwen2.5-0.5B-Instruct
SNAME=qwen05
ROOT=/tmp/rg_pd_smoke
mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.txt"; : > "$SUMMARY"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$SUMMARY"; }

stop_own(){ local pid=$1; kill -- -"$pid" 2>/dev/null; sleep 3; kill -9 -- -"$pid" 2>/dev/null; sleep 2; }
wait_health(){ local port=$1; for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return; }; sleep 5; done; echo 0; }

# export network env for mooncake/HCCL
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
  local label=$1 card=$2 port=$3 role=$4 kvport=$5
  local D="$ROOT/$label"; mkdir -p "$D"
  cd "$PRODUCT"
  export PYTHONPATH="$PRODUCT:${PYTHONPATH:-}"
  export ASCEND_RT_VISIBLE_DEVICES="$card"
  export VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1
  local kvcfg; kvcfg=$(printf '%s' "$KVC" | "$PY" -c "import json,sys; c=json.load(sys.stdin); c['kv_role']='$role'; c['kv_port']='$kvport'; print(json.dumps(c))")
  setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SNAME" \
    --host 127.0.0.1 --port "$port" --no-enable-prefix-caching \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --enforce-eager \
    --kv-transfer-config "$kvcfg" > "$D/serve.log" 2>&1 &
  echo $!
}

log "===== PD 1P1D smoke (P=card0:13700, D=card2:13701, proxy:8080) ====="
PID_P=$(serve_pd prefiller 0 13700 kv_producer 30000)
PID_D=$(serve_pd decoder 2 13701 kv_consumer 30100)
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
  log "proxy health: $(curl -s http://127.0.0.1:8080/health 2>/dev/null || echo 'no /health')"
  curl -s http://127.0.0.1:8080/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SNAME\",\"prompt\":\"hello pd 1p1d\",\"max_tokens\":16,\"temperature\":0,\"seed\":42}" \
    > "$ROOT/resp.json" 2>/dev/null || true
  sleep 6
  if grep -q '"choices"' "$ROOT/resp.json" 2>/dev/null; then
    log "PD REQUEST OK: $(head -c 300 "$ROOT/resp.json")"
  else
    log "PD REQUEST FAILED/NOT READY. resp: $(head -c 400 "$ROOT/resp.json")"
  fi
  log "--- prefiller tail ---"; tail -15 "$ROOT/prefiller/serve.log" >> "$SUMMARY"
  log "--- decoder tail ---"; tail -15 "$ROOT/decoder/serve.log" >> "$SUMMARY"
  log "--- proxy tail ---"; tail -15 "$ROOT/proxy.log" >> "$SUMMARY"
  stop_own "$PID_X"; stop_own "$PID_P"; stop_own "$PID_D"
else
  log "PD HEALTH_TIMEOUT"; tail -25 "$ROOT/prefiller/serve.log" >> "$SUMMARY"; tail -25 "$ROOT/decoder/serve.log" >> "$SUMMARY"
  stop_own "$PID_P"; stop_own "$PID_D"
fi
log "PD SMOKE DONE"
