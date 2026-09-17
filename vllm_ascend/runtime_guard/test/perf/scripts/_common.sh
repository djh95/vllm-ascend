#!/usr/bin/env bash
# runtime_guard perf shared env + helpers (analysis branch, feat/runtime-guard-analysis).
# Lab-specific paths live here (no secrets). Override via env when porting to another lab.
set -euo pipefail

PY="${PY:-/opt/slime/venv/bin/python}"
MODEL="${MODEL:-/data0/weights/Qwen2.5-0.5B-Instruct}"
# served-model-name MUST match perf_lib.py hardcoded "model":"dsv2" (vllm 0.27 validates it).
SNAME="${SNAME:-dsv2}"
PORT="${PORT:-8017}"
CARD="${CARD:-0}"   # card 1 RDMA/HCCS port is down; default single-instance card 0
TP="${TP:-1}"

PRODUCT_ROOT="${RG_PRODUCT_ROOT:-/data0/test-mrv2-cann91/vllm-ascend}"
T0_ROOT="${RG_T0_ROOT:-/data0/test-mrv2-cann91/rg-perf-t0}"   # merge-base 37e382498, no runtime_guard
ANALYSIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
PERF_DIR="$ANALYSIS_ROOT/vllm_ascend/runtime_guard/test/perf"

RUNNER="${RUNNER:-v2}"
case "$RUNNER" in
  v2) export VLLM_USE_V2_MODEL_RUNNER=1 ;;
  v1) export VLLM_USE_V2_MODEL_RUNNER=0 ;;
  *) echo "RUNNER=v1|v2 required" >&2; exit 2 ;;
esac

RG_PERF_ROOT="${RG_PERF_ROOT:-$PERF_DIR/rg_perf}"
export RG_PERF_ROOT
mkdir -p "$RG_PERF_ROOT/logs/$RUNNER" "$RG_PERF_ROOT/config" \
         "$RG_PERF_ROOT/serve" "$RG_PERF_ROOT/report/$RUNNER" "$RG_PERF_ROOT/results"

export RG_PERF_URL="http://127.0.0.1:${PORT}/v1/completions"
export RG_PERF_NPU="$CARD"

log(){ echo "$(date '+%H:%M:%S') [perf:$RUNNER] $*"; }

# ---- T2 / T3 runtime_config JSON writers ----
write_cfg_t2(){
  local cfg="$RG_PERF_ROOT/config/t2_reload_detectors_off.json"
  cat > "$cfg" <<'EOF'
{
  "reload_interval_seconds": 3,
  "dump": { "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false },
  "detector": {
    "logits_finite": { "enabled": false },
    "token_repeat": { "enabled": false },
    "output_substring": { "enabled": false },
    "spec_acceptance": { "enabled": false }
  }
}
EOF
  echo "$cfg"
}

write_cfg_t3(){
  local cfg="$RG_PERF_ROOT/config/t3_reload_detectors_on.json"
  cat > "$cfg" <<'EOF'
{
  "reload_interval_seconds": 3,
  "dump": { "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false },
  "actions": { "defaults": { "on_trigger": ["report"] } },
  "report": { "save_sensitive_info": false, "max_per_req": 1 },
  "detector": {
    "logits_finite": { "enabled": true },
    "token_repeat": { "enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 32, "consecutive_hits": 1 },
    "output_substring": { "enabled": true, "patterns": [], "add_special_tokens": false, "match_prefix": false },
    "spec_acceptance": { "enabled": true, "window": 10, "low_threshold": 0.3, "len_low_threshold": 1.4, "high_threshold": 0.96, "len_high_threshold": 2.8 }
  }
}
EOF
  echo "$cfg"
}

wait_health(){
  local port="${1:-$PORT}" i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then echo 1; return; fi
    sleep 5
  done
  echo 0
}

# Kill vllm serve + TP workers. Primary = process-group kill of the recorded
# setsid leader PID (kills EngineCore + TP workers). We deliberately avoid
# "pkill -f vllm serve" (matches our own shell). npu-smi sweep is the safety net.
stop_own(){
  local pid="$1"
  kill -- -"$pid" 2>/dev/null || true
  sleep 3
  kill -9 -- -"$pid" 2>/dev/null || true
  sleep 2
}

wait_idle(){
  # Accepts a single card id or a comma list ("2,3"); npu-smi prints one
  # "No running processes found in NPU <id>" row per card, so every card in
  # the list must be checked separately (grepping "NPU 2,3" never matches).
  local cards="${1:-$CARD}" i out c ok
  for i in $(seq 1 60); do
    # Capture first, then grep the string — a `npu-smi | grep -q` pipeline under
    # `set -o pipefail` returns SIGPIPE (141) when grep exits early, so the `if`
    # would always be false. Here-string avoids that pipe entirely.
    out=$(npu-smi info 2>/dev/null || true)
    ok=1
    IFS=',' read -ra _cards <<< "$cards"
    for c in "${_cards[@]}"; do
      c="${c//[[:space:]]/}"
      [ -n "$c" ] || continue
      if ! grep -q "No running processes found in NPU $c" <<< "$out"; then
        ok=0
        break
      fi
    done
    if [ "$ok" = 1 ]; then
      return 0
    fi
    sleep 2
  done
  log "WARN: card $cards still busy after wait_idle"
  return 1
}

# Build the --additional-config JSON for runtime_guard (T2/T3 only; empty for T0/T1).
_additional_config_json(){
  local cfgpath="$1" reload="$2"
  "$PY" - "$cfgpath" "$reload" "$RG_PERF_ROOT/report/$RUNNER" <<'PY'
import json, sys
cfg, reload, report_dir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
print(json.dumps({
    "runtime_config_path": cfg,
    "runtime_config_reload_interval": reload,
    "runtime_report_dir": report_dir,
}))
PY
}

# serve <tree> <label> [cfgpath] [reload_interval]  -> echoes server pid
serve(){
  local tree="$1" label="$2" cfgpath="${3:-}" reload="${4:-0}"
  local logf="$RG_PERF_ROOT/serve/serve_${RUNNER}_${label}.log"
  : > "$logf"
  cd "$tree"
  export PYTHONPATH="$tree:${PYTHONPATH:-}"
  export ASCEND_RT_VISIBLE_DEVICES="$CARD"
  export VLLM_BATCH_INVARIANT=1
  local add=()
  if [ -n "$cfgpath" ]; then
    add=(--additional-config "$(_additional_config_json "$cfgpath" "$reload")")
  fi
  setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SNAME" \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP" --gpu-memory-utilization 0.85 --enforce-eager \
    "${add[@]}" > "$logf" 2>&1 &
  echo $!
}

# serve_and_wait <tree> <label> [cfgpath] [reload]  -> echoes pid (exit 1 on health timeout)
# NOTE: logs go to stderr so `pid=$(serve_and_wait ...)` captures ONLY the pid.
serve_and_wait(){
  local tree="$1" label="$2" cfgpath="${3:-}" reload="${4:-0}"
  local pid ok
  pid=$(serve "$tree" "$label" "$cfgpath" "$reload")
  log "[$label] boot pid=$pid tree=$tree" >&2
  ok=$(wait_health "$PORT")
  if [ "$ok" != 1 ]; then
    log "[$label] HEALTH_TIMEOUT" >&2
    tail -30 "$RG_PERF_ROOT/serve/serve_${RUNNER}_${label}.log" >&2 || true
    stop_own "$pid"
    return 1
  fi
  log "[$label] health=200 pid=$pid" >&2
  echo "$pid"
}

# measure N inner rounds (1 warmup dropped) -> outfile. Each jsonl line carries
# runner= and state= (T-label). Uses perf_lib from the perf dir (no vllm import).
# RG_INNER_ROUNDS defaults to 3 (warmup 1 + 3 measured) — cross-rotation already
# yields 6 serve cycles x 3 rounds x 3 REQS = 54 samples per state.
measure_rounds(){
  local out="$1" state="$2"
  cd "$PERF_DIR"
  RG_PERF_OUT="$out" RG_STATE="$state" RUNNER="$RUNNER" \
    RG_INNER_ROUNDS="${RG_INNER_ROUNDS:-3}" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY'
import json, os
from perf_lib import warmup, post, REQS
out = os.environ["RG_PERF_OUT"]
state = os.environ["RG_STATE"]
runner = os.environ["RUNNER"]
inner = int(os.environ.get("RG_INNER_ROUNDS", "3"))
warmup(rounds=1)
with open(out, "w") as f:
    for rnd in range(1, inner + 1):
        for tag, p, mt in REQS:
            wall, pt, ct, tps = post(p, mt)
            rec = {"round": rnd, "state": state, "runner": runner, "tag": tag,
                   "wall_s": round(wall, 2), "prompt_tok": pt, "compl_tok": ct,
                   "out_tok_s": round(tps, 2)}
            f.write(json.dumps(rec) + "\n"); f.flush()
print("measured rounds", flush=True)
PY
}

# geometric-mean ratio a/b from jsonl files (out_tok_s). args: a.jsonl b.jsonl
geom_ratio(){
  "$PY" - "$1" "$2" <<'PY'
import json, math, sys
def gm(path):
    vals = [float(json.loads(l)["out_tok_s"]) for l in open(path) if l.strip()]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0
a, b = gm(sys.argv[1]), gm(sys.argv[2])
print(f"{a:.3f} {b:.3f} {b/a:.5f}")
PY
}

# combine all *_r*.jsonl for a state into one file (for geom mean across cycles)
cat_state(){
  local state="$1" dir="$2" out="$3"
  : > "$out"
  local f
  for f in "$dir"/${state}_r*.jsonl; do
    [ -e "$f" ] && cat "$f" >> "$out"
  done
}
