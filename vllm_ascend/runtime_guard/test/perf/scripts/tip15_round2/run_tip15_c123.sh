#!/usr/bin/env bash
# C1/C2/C3 cross-rotation for product tip b1b58921f+2ca3cdab5 (TP0 due broadcast).
# T0 -> T1 -> T2 -> T3 interleaved, N=3 rounds (cross-rotation protocol).
#   t0: merge-base 8a2c3182c, no runtime_guard (worktree rg-t0-b1b58)
#   t1: product tip, reload off, detectors off
#   t2: product tip, reload=3s, detectors off   <- merged due-broadcast hot path
#   t3: product tip, reload=3s, detectors on
# Gates: C1 = gm(t1)/gm(t0) >= 0.999 (infra zero-cost)
#        C2 = gm(t2)/gm(t1) >= 0.999 (hot-reload overhead)
#        C3 = gm(t3)/gm(t2) >= 0.990 (detector overhead)
# RUNNER=v2|v1 (default v2). Model Qwen2.5-7B TP=2, cards 4,5, port 8182.
# SAFETY: waits idle cards; pgid-scoped cleanup; residual ignores defunct.
set -uo pipefail
RUNNER=${RUNNER:-v2}
PY=/opt/slime/venv/bin/python
V030=/data0/test-mrv2-cann91/vllm030_pkgs
TREE=/data0/test-mrv2-cann91/rg-tip15-8f5e3
T0TREE=/data0/test-mrv2-cann91/rg-t0-b1b58
PERF_DIR=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/test/perf
ROOT=/data0/test-mrv2-cann91/rg_c123_tip15_${RUNNER}
LOG=$ROOT/master.log
OUT=$ROOT/perf.jsonl
PORT=8182
MODEL=/data0/weights/Qwen2.5-7B-Instruct
PARGS="--tensor-parallel-size 2"
CARDS=4,5
N=3
POINT=tip15_b1b58_vllm030_${RUNNER}
mkdir -p "$ROOT"
: > "$OUT"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

hbm(){ npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+"; }
wait_idle_cards(){
  local i c ok
  for i in $(seq 1 180); do
    ok=1; IFS=',' read -ra _cs <<< "$CARDS"
    for c in "${_cs[@]}"; do [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null || { ok=0; break; }; done
    [ "$ok" = 1 ] && return 0; sleep 20
  done
  return 1
}
wait_health(){
  local i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done
  echo 0
}
stop_pgid(){
  kill -- -"$1" 2>/dev/null || true; sleep 3
  kill -9 -- -"$1" 2>/dev/null || true; sleep 2
}
residual_check(){
  local label=$1 pgid=$2 out
  sleep 3
  out=$(ps -eo pid,ppid,pgid,etime,args | awk -v g="$pgid" '$3==g && $1!=g' | grep -v defunct | head -20)
  if [ -n "$out" ]; then
    log "RESIDUAL_LIVE $label: pgid=$pgid survivors:"; echo "$out" | tee -a "$LOG"
    kill -9 -- -"$pgid" 2>/dev/null || true
  else
    log "residual $label: none"
  fi
}
write_cfg_t2(){
  cat > "$ROOT/cfg_t2.json" <<EOF
{"reload_interval_seconds": 3, "dump": {"dump_dir": "$ROOT/dump_t2", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "detector": {"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}}
EOF
}
write_cfg_t3(){
  cat > "$ROOT/cfg_t3.json" <<EOF
{"reload_interval_seconds": 3, "dump": {"dump_dir": "$ROOT/dump_t3", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true}, "spec_acceptance": {"enabled": true}}}
EOF
}

boot(){
  local tree=$1 state=$2 cfg=$3 reload=$4
  local add=()
  if [ -n "$cfg" ]; then
    add=(--additional-config "{\"runtime_config_path\": \"$cfg\", \"runtime_config_reload_interval\": $reload, \"runtime_report_dir\": \"$ROOT/report_${state}\"}")
  fi
  ( cd "$tree"
    env PYTHONPATH="$V030:$tree:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=$CARDS \
      VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=$([ "$RUNNER" = v2 ] && echo 1 || echo 0) \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name dsv2 --port "$PORT" $PARGS \
      --gpu-memory-utilization 0.85 --enforce-eager "${add[@]}" \
      > "$ROOT/serve_${state}.log" 2>&1 & echo $! )
}

measure(){
  local state=$1 round=$2
  ( cd "$PERF_DIR"
    RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" RG_PERF_OUT="$OUT" \
    RG_POINT="$POINT" RG_STATE="$state" RG_ROUND="$round" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY'
import json, math, os
from perf_lib import warmup, post, REQS
out, point = os.environ["RG_PERF_OUT"], os.environ["RG_POINT"]
state, rnd = os.environ["RG_STATE"], os.environ["RG_ROUND"]
warmup(rounds=1)
tps = []
with open(out, "a") as f:
    for tag, p, mt in REQS:
        wall, pt, ct, t = post(p, mt)
        tps.append(t)
        f.write(json.dumps({"point": point, "state": state, "round": int(rnd), "tag": tag,
                            "wall_s": round(wall, 2), "compl_tok": ct, "out_tok_s": round(t, 2)}) + "\n")
gm = math.exp(sum(math.log(t) for t in tps) / len(tps))
print(f"ROUND_GM state={state} round={rnd} gm={gm:.3f} tok/s", flush=True)
PY
  )
}

report(){
  "$PY" - "$OUT" "$LOG" "$POINT" <<'PY'
import json, math, sys
from collections import defaultdict
point = sys.argv[3]
per = defaultdict(lambda: defaultdict(dict))
for line in open(sys.argv[1]):
    d = json.loads(line)
    per[(d["point"], d["state"])].setdefault(d["round"], []).append(d["out_tok_s"])
gm, lines = {}, []
for (pt, state), rounds in sorted(per.items()):
    for r, vals in rounds.items():
        rounds[r] = math.exp(sum(math.log(v) for v in vals) / len(vals))
    gm[state] = math.exp(sum(math.log(v) for v in rounds.values()) / len(rounds))
    spread = max(rounds.values()) / min(rounds.values()) - 1
    lines.append(f"state={state} gm={gm[state]:.3f} tok/s rounds=" +
                 ",".join(f"r{r}:{v:.2f}" for r, v in sorted(rounds.items())) + f" round_spread={spread:+.2%}")
c1 = gm["t1"] / gm["t0"]
c2 = gm["t2"] / gm["t1"]
c3 = gm["t3"] / gm["t2"]
v1_ = "PASS" if c1 >= 0.999 else "FAIL"
v2_ = "PASS" if c2 >= 0.999 else "FAIL"
v3_ = "PASS" if c3 >= 0.990 else "FAIL"
overall = "PASS" if "PASS" in (v1_, v2_, v3_) and "FAIL" not in (v1_, v2_, v3_) else "FAIL"
lines.append(f"C1 T1/T0={c1:.5f} (>=0.999) {v1_}")
lines.append(f"C2 T2/T1={c2:.5f} (>=0.999) {v2_}")
lines.append(f"C3 T3/T2={c3:.5f} (>=0.990) {v3_}")
msg = f"TIP15_C123_VERDICT RUNNER={point.split('_')[-1]} " + " | ".join(lines) + f" | OVERALL={overall}"
print(msg)
open(sys.argv[2], "a").write(msg + "\n")
PY
}

log "=== C123 tip15 e2e: point=$POINT T0->T1->T2->T3 cross-rotation N=$N cards=$CARDS port=$PORT ==="
write_cfg_t2
write_cfg_t3
CFG_T2=$ROOT/cfg_t2.json
CFG_T3=$ROOT/cfg_t3.json
log "cfg_t2=$CFG_T2 cfg_t3=$CFG_T3"
for r in $(seq 1 "$N"); do
  for state in t0 t1 t2 t3; do
    log "--- round $r $state: waiting idle cards"
    if ! wait_idle_cards; then log "ABORT round $r $state: cards never idle"; exit 1; fi
    case $state in
      t0) pgid=$(boot "$T0TREE" t0 "" 0) ;;
      t1) pgid=$(boot "$TREE" t1 "" 0) ;;
      t2) pgid=$(boot "$TREE" t2 "$CFG_T2" 3) ;;
      t3) pgid=$(boot "$TREE" t3 "$CFG_T3" 3) ;;
    esac
    log "round $r $state: serve pgid=$pgid port=$PORT"
    if [ "$(wait_health)" != 1 ]; then
      log "round $r $state: HEALTH_FAIL (see serve_${state}.log)"
      grep -iE "error|exception|Traceback|not support" "$ROOT/serve_${state}.log" | head -5 | tee -a "$LOG"
      stop_pgid "$pgid"; residual_check "round $r $state" "$pgid"
      continue
    fi
    log "round $r $state: health=200, measuring"
    measure "$state" "$r" 2>&1 | tee -a "$LOG"
    stop_pgid "$pgid"; residual_check "round $r $state" "$pgid"
    sleep 10
  done
done
log "MERGED_BUS_UNFINISHED_T1=$(grep -c 'merged bus not finished' "$ROOT/serve_t1.log" 2>/dev/null) MERGED_BUS_UNFINISHED_T2=$(grep -c 'merged bus not finished' "$ROOT/serve_t2.log" 2>/dev/null)"
mis=0
for s in t0 t1 t2 t3; do
  m=$(grep -c "wave misalignment" "$ROOT/serve_${s}.log" 2>/dev/null || true)
  mis=$((mis + ${m:-0}))
done
log "WAVE_MISALIGNMENT_TOTAL=$mis"
report
log "=== C123 TIP15 E2E DONE RUNNER=$RUNNER ==="
