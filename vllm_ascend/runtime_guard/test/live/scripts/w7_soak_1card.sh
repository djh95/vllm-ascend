#!/usr/bin/env bash
# W7 24h soak: 1-card v1 (SOAK_PLAN S6-style, runner=v1), different model Qwen3-4B.
# Runs in parallel with w5 (2-card v2) and w6 (4-card TP+PP v2).
#
#   nohup bash /data0/test-mrv2-cann91/rg_test/w7_soak_1card.sh > /tmp/rg_soak_1card/master.log 2>&1 &
#
# Instance: Qwen3-4B, card 6, port 8093, v1, token_repeat + report.
# Driver: 8 concurrent + hot-reload every 5min + /health watchdog.
set -uo pipefail

PY=/opt/slime/venv/bin/python
PRODUCT=/data0/test-mrv2-cann91/rg-config-review
ROOT=/tmp/rg_soak_1card
NAME=s6_1card
CARD=6
PORT=8093
DURATION=$((24*3600))
mkdir -p "$ROOT/$NAME/report" "$ROOT/$NAME/dump"
MASTER_LOG="$ROOT/master.log"

log(){ echo "$(date '+%F %H:%M:%S') $*"; }
log "W7 soak master start pid=$$ name=$NAME card=$CARD port=$PORT runner=v1"

OVERLAY='{"detector":{"token_repeat":{"enabled":true,"window":32,"repeat_sum_threshold":64,"min_tokens":32,"consecutive_hits":1},"logits_finite":{"enabled":false},"output_substring":{"enabled":false},"spec_acceptance":{"enabled":false}},"actions":{"defaults":{"on_trigger":["report"]}},"dump":{"auto_max_times":0,"manual_dump":false}}'

cd "$PRODUCT"
D="$ROOT/$NAME"
PYTHONPATH="$PRODUCT:${PYTHONPATH:-}" \
ASCEND_RT_VISIBLE_DEVICES="$CARD" \
VLLM_BATCH_INVARIANT=1 \
VLLM_USE_V2_MODEL_RUNNER=0 \
setsid "$PY" -m vllm.entrypoints.openai.api_server \
  --model /data0/weights/Qwen3-4B --served-model-name "$NAME" --port "$PORT" \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --additional-config "{\"runtime_config\": $OVERLAY, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$D/runtime_config.json\", \"runtime_report_dir\": \"$D/report\"}" \
  > "$D/serve.log" 2>&1 &
PID=$!
log "boot server $NAME pid=$PID"

wait_health(){ for i in $(seq 1 360); do curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo 1; return; }; sleep 10; done; echo 0; }
OK=$(wait_health "$PORT"); log "server $NAME health=$OK pid=$PID"
if [ "$OK" != "1" ]; then
  echo "RESULT|$NAME|TP=1|v1|token_repeat+report|health=0|hang=0|crash=1|elapsed=0" >> "$ROOT/result.txt"
  log "health timeout; aborting"
  kill -- -"$PID" 2>/dev/null
  exit 1
fi

START_EPOCH=$(date +%s)

"$PY" - "$ROOT" "$NAME" "$PORT" "$PID" "$START_EPOCH" "$DURATION" <<'PY'
import json, os, subprocess, sys, threading, time
ROOT, NAME, PORT, PID, START_EPOCH, DURATION = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
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
def one_req(i):
    p = PROMPTS[i % len(PROMPTS)]
    body = json.dumps({"model": NAME, "prompt": p, "max_tokens": 128, "temperature": 0.7})
    try:
        r = subprocess.run(["curl", "-s", f"http://127.0.0.1:{PORT}/v1/completions",
                            "-H", "Content-Type: application/json", "-d", body],
                           capture_output=True, timeout=120)
        return r.returncode
    except Exception:
        return -1
def health_ok():
    try:
        return subprocess.run(["curl", "-sf", f"http://127.0.0.1:{PORT}/health"],
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
def proc_alive():
    try:
        os.kill(PID, 0); return True
    except OSError:
        return False
def hot_reload(round_no):
    cfg = os.path.join(ROOT, NAME, "runtime_config.json")
    try:
        with open(cfg) as f: d = json.load(f)
    except Exception:
        return
    d.setdefault("detector", {}).setdefault("token_repeat", {})["window"] = 32 + (round_no % 8) * 8
    try:
        with open(cfg, "w") as f: json.dump(d, f)
    except Exception:
        pass
def heartbeat(line):
    with open(os.path.join(ROOT, "heartbeat.log"), "a") as f:
        f.write(f"{time.strftime('%F %H:%M:%S')} {line}\n")

round_no = 0
fail_streak = 0
hung = 0
while time.time() - START_EPOCH < DURATION:
    round_no += 1
    threads = [threading.Thread(target=one_req, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    if round_no % 5 == 0:
        hot_reload(round_no)
        heartbeat(f"hot_reload round={round_no}")
    if not proc_alive():
        heartbeat("crash: server process exited")
        break
    if health_ok():
        fail_streak = 0
    else:
        fail_streak += 1
        if fail_streak >= 9:
            hung = 1
            heartbeat("hang: health down >15min")
            break
    time.sleep(100)
elapsed = int(time.time() - START_EPOCH)
health = 1 if health_ok() else 0
crash = 0 if proc_alive() else 1
with open(os.path.join(ROOT, "result.txt"), "a") as f:
    f.write(f"RESULT|{NAME}|TP=1|v1|token_repeat+report|health={health}|hang={hung}|crash={crash}|elapsed={elapsed}\n")
print(f"soak driver done: health={health} hang={hung} crash={crash} elapsed={elapsed}")
PY

log "soak driver exited; killing server $NAME"
kill -- -"$PID" 2>/dev/null
log "W7 soak master end"
