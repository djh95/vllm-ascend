#!/usr/bin/env bash
# P0-2 topology smoke: product tip b1b58921f+2ca3cdab5 (TP0 single-source
# due broadcast). PP=2 TP=2 on the v2 runner proves BOTH paths of the
# wave-head task bus at once:
#   - last-PP x TP  -> merged due-broadcast (sync_due_bits_from_src + wave_idx)
#   - non-last PP   -> local JSON file poll (no collectives)
# Model Qwen2.5-7B (no custom-op package needed), 4 idle cards, port 8192.
# PASS requires: health, sane output, manual dump .pt ONLY from last-PP TP
# shards (rank tags dp0_tp{0,1}_pp1_cp0), no wave misalignment, no Traceback,
# clean residual + HBM release. pgid-scoped cleanup only.
set -uo pipefail
PY=/opt/slime/venv/bin/python
V030=/data0/test-mrv2-cann91/vllm030_pkgs
PRODUCT=/data0/test-mrv2-cann91/rg-tip15-8f5e3
MODEL=/data0/weights/Qwen2.5-7B-Instruct
ROOT=/data0/test-mrv2-cann91/rg_tip15_pp2_smoke
LOG=$ROOT/master.log
CARDS=${PP2_CARDS:-4,5,6,7}
PORT=8192
mkdir -p "$ROOT" "$ROOT/report"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }
PASS=1
fail(){ PASS=0; log "FAIL: $*"; }

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
stop_pgid(){ kill -- -"$1" 2>/dev/null || true; sleep 5; kill -9 -- -"$1" 2>/dev/null || true; sleep 3; }
residual_check(){
  local out; sleep 3
  out=$(ps -eo pid,ppid,pgid,etime,args | awk -v g="$1" '$3==g && $1!=g' | grep -v defunct | head -10)
  if [ -n "$out" ]; then fail "residual processes alive after kill"; echo "$out" | tee -a "$LOG"
    kill -9 -- -"$1" 2>/dev/null || true
  else log "residual: none"; fi
}
hbm_released(){
  local c ok=1; IFS=',' read -ra _cs <<< "$CARDS"
  for c in "${_cs[@]}"; do [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null || { ok=0; log "HBM card $c still held ($(hbm "$c"))"; }; done
  [ "$ok" = 1 ] && log "HBM released on all cards" || fail "HBM not released"
}
write_cfg(){
  cat > "$ROOT/runtime_config.json" <<CFG
{"dump": {"dump_dir": "$ROOT/dump", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": $1}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true}, "spec_acceptance": {"enabled": true}}}
CFG
}

log "=== tip15 P0-2 PP2xTP2 smoke start PRODUCT=$PRODUCT cards=$CARDS port=$PORT ==="
if ! wait_idle_cards; then log "ABORT: cards busy"; log "TIP15_PP2_SMOKE_VERDICT FAIL"; exit 1; fi
write_cfg 0
pid=$( cd "$PRODUCT"
  env PYTHONPATH="$V030:$PRODUCT:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=$CARDS \
    VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name t5 --port "$PORT" \
    --pipeline-parallel-size 2 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 --enforce-eager \
    --additional-config "{\"runtime_config_path\": \"$ROOT/runtime_config.json\", \"runtime_config_hot_reload\": true, \"runtime_report_dir\": \"$ROOT/report\"}" \
    > "$ROOT/serve.log" 2>&1 & echo $! )
log "boot pid=$pid"
if [ "$(wait_health)" != 1 ]; then
  fail "health timeout"; tail -40 "$ROOT/serve.log" | tee -a "$LOG"
  stop_pgid "$pid"; residual_check "$pid"
  log "TIP15_PP2_SMOKE_VERDICT FAIL"; exit 1
fi
log "health=200"

out=$(curl -s --max-time 180 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"t5","prompt":"Once upon a time in a distant land","max_tokens":24,"temperature":0}')
echo "$out" > "$ROOT/short.json"
if "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); t=r["choices"][0]["text"]; sys.exit(0 if len(t.strip())>0 and "error" not in r else 1)' "$ROOT/short.json" 2>/dev/null; then
  log "short completion OK"
else fail "short completion bad: $(head -c 300 "$ROOT/short.json")"; fi

curl -s --max-time 300 http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"t5","prompt":"Write a long story about the sea, ships and sailors crossing the ocean","max_tokens":400,"temperature":0}' > "$ROOT/long.json" &
cpid=$!
sleep 6
write_cfg 1
sleep 18
"$PY" - "$ROOT/dump" "$ROOT/report" "$ROOT/dump_ranks.json" <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, os, sys
dumpdir, reportdir, outjson = sys.argv[1], sys.argv[2], sys.argv[3]
man_pt = sorted(set(
    glob.glob(f"{dumpdir}/**/*.pt", recursive=True) +
    glob.glob(f"{reportdir}/kv_cache/**/*.pt", recursive=True)))
ranks = sorted({os.path.basename(os.path.dirname(p)) for p in man_pt})
print(f"DUMP_CHECK manual_pt={len(man_pt)} ranks={ranks}")
json.dump({"manual_pt": len(man_pt), "ranks": ranks}, open(outjson, "w"))
PYEOF
man_pt=$("$PY" -c "import json; print(json.load(open('$ROOT/dump_ranks.json'))['manual_pt'])" 2>/dev/null || echo 0)
ranks=$("$PY" -c "import json; print(' '.join(json.load(open('$ROOT/dump_ranks.json'))['ranks']))" 2>/dev/null || true)
log "manual dump: pt=$man_pt ranks=[$ranks]"
if [ "${man_pt:-0}" -lt 1 ] 2>/dev/null; then fail "manual dump produced no .pt files"; fi
if [ -n "$ranks" ]; then
  bad_rank=$(echo "$ranks" | tr ' ' '\n' | grep -v "_pp1_" | head -1 || true)
  [ -n "$bad_rank" ] && fail "non-last-PP rank dumped: $bad_rank"
  n_ranks=$(echo "$ranks" | wc -w | tr -d ' ')
  [ "${n_ranks:-0}" -lt 2 ] && fail "expected 2 last-PP TP shards, got ${n_ranks:-0} rank dirs"
else
  fail "no rank dirs found"
fi

wait "$cpid" || true
log "long resp_len=$(wc -c < "$ROOT/long.json" 2>/dev/null)"
write_cfg 0
sleep 5

started=$(grep -c "bus worker started" "$ROOT/serve.log" || true)
stopped=$(grep -c "bus worker stopped" "$ROOT/serve.log" || true)
log "bus_worker started=${started:-0} stopped=${stopped:-0} (stopped==0 is normal under pgid kill flow)"
[ "${started:-0}" -ge 1 ] 2>/dev/null || fail "no bus worker started lines"
mis=$(grep -c "wave misalignment" "$ROOT/serve.log" || true)
[ "${mis:-0}" = 0 ] || fail "wave misalignment detected: $mis"
tb=$(grep -c "Traceback" "$ROOT/serve.log" || true)
[ "${tb:-0}" = 0 ] || { fail "Traceback in serve.log: $tb"; grep -A8 "Traceback" "$ROOT/serve.log" | head -20 | tee -a "$LOG"; }
inc=$(find "$ROOT/report" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
log "incident_report_json=${inc:-0}"

stop_pgid "$pid"
residual_check "$pid"
hbm_released
if [ "$PASS" = 1 ]; then log "TIP15_PP2_SMOKE_VERDICT PASS"; else log "TIP15_PP2_SMOKE_VERDICT FAIL"; fi
log "=== tip15 P0-2 PP2xTP2 smoke done ==="
