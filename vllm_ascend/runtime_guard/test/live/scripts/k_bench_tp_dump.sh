#!/usr/bin/env bash
# K-bench: KV dump correctness benchmark — single-shot manual_dump (:2), TP-parametric.
# Captures one deterministic dump per run; run TP=1 as REF and TP>=2 as TARGET,
# then p0_09_tp_stitch.sh (completeness) + p0_10_kv_compare.sh (vs REF).
# Env (defaults tuned for test-mrv2-cann91):
#   RG_PRODUCT_ROOT  config-branch checkout      PY    python with vllm
#   MODEL            model dir                   NPU   ASCEND_RT_VISIBLE_DEVICES
#   PORT             api port                    RG_PROMPT / RG_MAX_TOKENS / RG_SEED
# usage: k_bench_tp_dump.sh <tp> <cards> <out_dir> [port]
set -uo pipefail
TP=${1:?tp}; CARDS=${2:?cards}; OUT=${3:?out}; PORT=${4:-8041}
PRODUCT="${RG_PRODUCT_ROOT:-/data0/test-mrv2-cann91/vllm-ascend}"
PY="${PY:-/opt/slime/venv/bin/python}"
MODEL="${MODEL:-/data0/weights/Qwen2.5-0.5B-Instruct}"
SNAME="${SERVED_MODEL_NAME:-qwen05}"
PROMPT="${RG_PROMPT:-hello dump}"
MAXT="${RG_MAX_TOKENS:-16}"
SEED="${RG_SEED:-42}"
mkdir -p "$OUT/report" "$OUT/dump"
cd "$PRODUCT"
export PYTHONPATH="$PRODUCT:${PYTHONPATH:-}"
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V2_MODEL_RUNNER="${RUNNER:-1}"
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
OVERLAY=$("$PY" - "$OUT/dump" <<'PY'
import json, sys
c = {
    "reload_interval_seconds": 3,
    "dump": {"auto_max_times": 0, "manual_dump": 2, "dump_dir": sys.argv[1]},
    "actions": {"defaults": {"on_trigger": ["report", "dump_kv"]}},
    "report": {"save_sensitive_info": True},
    "detector": {
        "logits_finite": {"enabled": False},
        "token_repeat": {"enabled": False},
        "output_substring": {"enabled": False},
        "spec_acceptance": {"enabled": False},
    },
}
print(json.dumps(c))
PY
)
echo "[k_bench] tp=$TP cards=$CARDS out=$OUT port=$PORT model=$MODEL"
nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SNAME" --port "$PORT" \
  --gpu-memory-utilization 0.85 --enforce-eager --tensor-parallel-size "$TP" \
  --additional-config "{\"runtime_config\": $OVERLAY, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$OUT/runtime_config.json\", \"runtime_report_dir\": \"$OUT/report\"}" \
  > "$OUT/serve.log" 2>&1 &
PID=$!
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo SERVE_DIED; tail -30 "$OUT/serve.log"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo HEALTH_TIMEOUT; tail -30 "$OUT/serve.log"; kill $PID; exit 1; }
echo "health OK"
curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SNAME\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAXT,\"temperature\":0,\"seed\":$SEED}" > "$OUT/response.json"
head -c 300 "$OUT/response.json"; echo
sleep 12
kill $PID 2>/dev/null
for i in $(seq 1 12); do kill -0 $PID 2>/dev/null || break; sleep 5; done
kill -9 $PID 2>/dev/null; pkill -9 -f "port.$PORT" 2>/dev/null
sleep 3
echo "pt files: $(find "$OUT/dump" -name '*.pt' | wc -l)"
echo "reports:  $(find "$OUT/report" -name 'report_*.json' | wc -l)"
find "$OUT/dump" -type d -name 'wave_*' | sort
echo "next: WAVE=<tp_dir wave_N> p0_09_tp_stitch.sh ; TARGET=<tpN wave> REF=<tp1 wave> COS_THRESH=0.999 p0_10_kv_compare.sh"
