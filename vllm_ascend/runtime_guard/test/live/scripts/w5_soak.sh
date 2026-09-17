#!/usr/bin/env bash
# W5 24h soak: high-concurrency + multi-model + periodic hot-reload.
# Start detached via nohup; record startup status only (do not wait 24h).
#
#   nohup bash /data0/test-mrv2-cann91/rg_test/w5_soak.sh > /tmp/rg_soak/master.log 2>&1 &
#
# Models:
#   A: Qwen2.5-0.5B-Instruct  card 4  port 8090  token_repeat detector
#   B: Qwen2.5-7B-Instruct    card 5  port 8091  logits_finite detector
# Each driver: 8 concurrent completions loop + hot-reload every 300s.
set -uo pipefail

PY=/opt/slime/venv/bin/python
PRODUCT=/data0/test-mrv2-cann91/vllm-ascend
ROOT=/tmp/rg_soak
mkdir -p "$ROOT"
MASTER_LOG="$ROOT/master.log"

log(){ echo "$(date '+%F %H:%M:%S') $*"; }
log "soak master start pid=$$"

# ---- server A ----
boot_server(){
  local name=$1 model=$2 card=$3 port=$4 overlay=$5
  local D="$ROOT/$name"
  mkdir -p "$D/report" "$D/dump"
  cd "$PRODUCT"
  PYTHONPATH="$PRODUCT:${PYTHONPATH:-}" \
  ASCEND_RT_VISIBLE_DEVICES="$card" \
  VLLM_BATCH_INVARIANT=1 \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "soak_$name" --port "$port" \
    --gpu-memory-utilization 0.85 --enforce-eager \
    --additional-config "{\"runtime_config\": $overlay, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$D/runtime_config.json\", \"runtime_report_dir\": \"$D/report\"}" \
    > "$D/serve.log" 2>&1 &
  echo $!
}

OV_A='{"detector":{"token_repeat":{"enabled":true,"window":32,"repeat_sum_threshold":64,"min_tokens":32,"consecutive_hits":1},"logits_finite":{"enabled":false},"output_substring":{"enabled":false},"spec_acceptance":{"enabled":false}},"actions":{"defaults":{"on_trigger":["report"]}},"dump":{"auto_max_times":0,"manual_dump":false}}'
OV_B='{"detector":{"logits_finite":{"enabled":true},"token_repeat":{"enabled":false},"output_substring":{"enabled":false},"spec_acceptance":{"enabled":false}},"actions":{"defaults":{"on_trigger":["report"]}},"dump":{"auto_max_times":0,"manual_dump":false}}'

log "boot server A (Qwen2.5-0.5B, card 4, port 8090)"
PID_A=$(boot_server A /data0/weights/Qwen2.5-0.5B-Instruct 4 8090 "$OV_A")
log "boot server B (Qwen2.5-7B, card 5, port 8091)"
PID_B=$(boot_server B /data0/weights/Qwen2.5-7B-Instruct 5 8091 "$OV_B")

wait_health(){ for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo 1; return; }; sleep 10; done; echo 0; }

OK_A=$(wait_health 8090); log "server A health=$OK_A pid=$PID_A"
OK_B=$(wait_health 8091); log "server B health=$OK_B pid=$PID_B"

# ---- concurrent request + periodic hot-reload driver (runs 24h) ----
"$PY" - "$ROOT" <<'PY'
import json, os, subprocess, sys, threading, time, random
ROOT = sys.argv[1]
PROMPTS = [
    "Write a short poem about the sea",
    "Explain quantum computing in simple terms",
    "Tell me a story about a brave cat",
    "What is the capital of France and why?",
    "Describe a sunny day in the mountains",
    "Write a haiku about autumn leaves",
    "Summarize the theory of relativity",
    "Give me a recipe for apple pie",
]
def one_req(port, name, i):
    p = PROMPTS[i % len(PROMPTS)]
    body = json.dumps({"model": f"soak_{name}", "prompt": p,
                       "max_tokens": 128, "temperature": 0.7})
    try:
        r = subprocess.run(["curl", "-s", f"http://127.0.0.1:{port}/v1/completions",
                            "-H", "Content-Type: application/json", "-d", body],
                           capture_output=True, timeout=120)
        return r.returncode
    except Exception:
        return -1
def hot_reload(name, round_no):
    # periodically flip a detector field to exercise hot-reload path
    cfg = os.path.join(ROOT, name, "runtime_config.json")
    try:
        with open(cfg) as f:
            d = json.load(f)
    except Exception:
        return
    if name == "A":
        d.setdefault("detector", {}).setdefault("token_repeat", {})["window"] = 32 + (round_no % 8) * 8
    else:
        d.setdefault("ascend_log", {})["level"] = "INFO" if round_no % 2 == 0 else "WARNING"
    try:
        with open(cfg, "w") as f:
            json.dump(d, f)
    except Exception:
        pass
def heartbeat():
    with open(os.path.join(ROOT, "heartbeat.log"), "a") as f:
        f.write(f"{time.strftime('%F %H:%M:%S')} alive\n")

START = time.time()
DURATION = 24 * 3600
round_no = 0
while time.time() - START < DURATION:
    round_no += 1
    # 8 concurrent requests per model
    threads = []
    for i in range(8):
        threads.append(threading.Thread(target=one_req, args=(8090, "A", i)))
        threads.append(threading.Thread(target=one_req, args=(8091, "B", i)))
    for t in threads: t.start()
    for t in threads: t.join()
    # hot-reload every 5 min
    if round_no % 5 == 0:
        hot_reload("A", round_no)
        hot_reload("B", round_no)
        with open(os.path.join(ROOT, "reload.log"), "a") as f:
            f.write(f"{time.strftime('%F %H:%M:%S')} hot_reload round={round_no}\n")
    heartbeat()
    time.sleep(10)
print("soak driver done (24h elapsed or interrupted)")
PY

log "soak driver exited; killing servers A/B"
kill -- -"$PID_A" 2>/dev/null; kill -- -"$PID_B" 2>/dev/null
log "soak master end"
