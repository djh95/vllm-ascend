#!/usr/bin/env bash
# C4 identity rerun for product tip b1b58921f+2ca3cdab5 (TP0 due-broadcast).
# Four states, temp=0 outputs must be bit-identical across T0-T3:
#   T0 merge-base 8a2c3182c (no runtime_guard)   -> worktree rg-t0-b1b58
#   T1 product HEAD, no reload                    -> worktree rg-tip15-8f5e3
#   T2 product HEAD, reload=3, detectors off
#   T3 product HEAD, reload=3, 4 detectors on (report-only, no dump)
# v2 ModelRunner, TP=2, Qwen2.5-7B, cards 4,5. NPU required.
# SAFETY: waits idle cards; pgid-scoped cleanup; residual ignores defunct.
set -uo pipefail
PY=/opt/slime/venv/bin/python
V030=/data0/test-mrv2-cann91/vllm030_pkgs
TREE=/data0/test-mrv2-cann91/rg-tip15-8f5e3
T0TREE=/data0/test-mrv2-cann91/rg-t0-b1b58
ROOT=/data0/test-mrv2-cann91/rg_c4_tip15_tip
LOG=$ROOT/master.log
OUT=$ROOT/c4_identity.jsonl
SUMMARY=$ROOT/c4_summary.txt
PORT=8181
MODEL=/data0/weights/Qwen2.5-7B-Instruct
PARGS="--tensor-parallel-size 2"
CARDS=4,5
mkdir -p "$ROOT"
: > "$OUT"; : > "$SUMMARY"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }
log "=== tip15 C4 identity start (v2 TP2 7B) PRODUCT=$TREE T0=$T0TREE ==="

hbm(){ npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+"; }
wait_idle_cards(){
  local i c ok
  for i in $(seq 1 90); do
    ok=1; IFS=',' read -ra _cs <<< "$CARDS"
    for c in "${_cs[@]}"; do [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null || { ok=0; break; }; done
    [ "$ok" = 1 ] && return 0; sleep 20
  done
  return 1
}
wait_health(){
  local i; for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done; echo 0
}
stop_pgid(){ kill -- -"$1" 2>/dev/null || true; sleep 3; kill -9 -- -"$1" 2>/dev/null || true; sleep 2; }
residual_check(){
  local label=$1 pgid=$2 out; sleep 3
  out=$(ps -eo pid,ppid,pgid,etime,args | awk -v g="$pgid" '$3==g && $1!=g' | grep -v defunct | head -10)
  if [ -n "$out" ]; then log "RESIDUAL_LIVE $label:"; echo "$out" | tee -a "$LOG"; kill -9 -- -"$pgid" 2>/dev/null || true
  else log "residual $label: none"; fi
}

write_cfg(){
  local kind=$1 cfg=$ROOT/cfg_$1.json
  if [ "$kind" = t2 ]; then
    cat > "$cfg" <<EOF
{"reload_interval_seconds": 3, "dump": {"dump_dir": "$ROOT/dump_t2", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "detector": {"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}}
EOF
  else
    cat > "$cfg" <<EOF
{"reload_interval_seconds": 3, "dump": {"dump_dir": "$ROOT/dump_t3", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true}, "spec_acceptance": {"enabled": true}}}
EOF
  fi
  echo "$cfg"
}

boot(){
  local tree=$1 cfg=$2 reload=$3
  local add=()
  if [ -n "$cfg" ]; then
    add=(--additional-config "{\"runtime_config_path\": \"$cfg\", \"runtime_config_reload_interval\": $reload, \"runtime_report_dir\": \"$ROOT/report_state\"}")
  fi
  ( cd "$tree"
    env PYTHONPATH="$V030:$tree:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=$CARDS \
      VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name dsv2 --port "$PORT" $PARGS \
      --gpu-memory-utilization 0.85 --enforce-eager "${add[@]}" \
      > "$ROOT/serve_$4.log" 2>&1 & echo $! )
}

capture(){
  local state=$1 tree=$2 cfg=$3 reload=$4 pid
  wait_idle_cards || { log "ABORT: cards busy"; exit 1; }
  pid=$(boot "$tree" "$cfg" "$reload" "$state")
  log "state=$state boot pid=$pid tree=$tree cfg=${cfg:-none}"
  [ "$(wait_health)" = 1 ] || { log "state=$state HEALTH FAIL"; stop_pgid "$pid"; exit 1; }
  log "state=$state health=200, capturing"
  RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" STATE="$state" \
    "$PY" - <<'PY' >> "$OUT"
import json, os, urllib.request
url = os.environ["RG_PERF_URL"]; state = os.environ["STATE"]
prompts = ["请介绍一下长城的历史和主要关口。",
           "请介绍一下李白的人生经历和代表作品。",
           "请围绕人工智能的发展写一段话。"]
for p in prompts:
    body = json.dumps({"model": "dsv2", "prompt": p, "max_tokens": 64,
                       "temperature": 0, "seed": 42}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    print(json.dumps({"state": state, "prompt": p,
                      "text": out["choices"][0]["text"]}, ensure_ascii=False))
PY
  stop_pgid "$pid"; residual_check "$state" "$pid"
  log "state=$state captured $(grep -c "\"state\": \"$state\"" "$OUT") prompts"
}

CFG_T2=$(write_cfg t2); CFG_T3=$(write_cfg t3)
capture T0 "$T0TREE" "" 0
capture T1 "$TREE" "" 0
capture T2 "$TREE" "$CFG_T2" 3
capture T3 "$TREE" "$CFG_T3" 3

"$PY" - "$OUT" <<'PY' | tee -a "$SUMMARY" | tee -a "$LOG"
import json, sys
from collections import defaultdict
by_prompt = defaultdict(dict)
for l in open(sys.argv[1]):
    if not l.strip(): continue
    r = json.loads(l)
    by_prompt[r["prompt"]][r["state"]] = r["text"]
ok = True
for p, states in sorted(by_prompt.items()):
    texts = set(states.values())
    identical = len(texts) == 1
    ok = ok and identical
    print(f"C4 prompt={p[:14]}... identical={identical} states={sorted(states)}")
print(f"C4 {'PASS' if ok else 'FAIL'} (temp=0 bit-identical across T0-T3, product tip b1b58921f+2ca3cdab5)")
PY
log "=== tip15 C4 done ==="
