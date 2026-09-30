#!/usr/bin/env bash
# tip15 stage-3 performance matrix runner (one config per invocation).
# Env contract (all required unless default):
#   MODEL=<weights dir>  NAME=<short tag>  TP=<n>  PP=<n>  DP=<n>
#   STATE=t1|t3  (t1: guard infra only, no config; t3: hot reload + 3 detectors on)
#   PY= V030= PRODUCT= ROOT= PORT= HBM_TOTAL= CARDS="0 1 ..." (idle pool to pick from)
# Output: appends one JSON line to $ROOT/matrix.jsonl + timeline to master.log.
# Fault model: boot/health failure -> verdict=BOOT_FAIL recorded, exit 0 (matrix continues).
set -uo pipefail
PY=${PY:?PY required} V030=${V030:?V030 required} PRODUCT=${PRODUCT:?PRODUCT required}
MODEL=${MODEL:?MODEL required} NAME=${NAME:?NAME required}
TP=${TP:-1} PP=${PP:-1} DP=${DP:-1} STATE=${STATE:-t1}
ROOT=${ROOT:?ROOT required} PORT=${PORT:-8250}
HBM_TOTAL=${HBM_TOTAL:-65536} CARDS=${CARDS:?CARDS required}
LOG=$ROOT/master.log JSONL=$ROOT/matrix.jsonl
mkdir -p "$ROOT"
log(){ echo "$(date '+%F %H:%M:%S') [matrix] $*" | tee -a "$LOG"; }

hbm(){
  local v
  v=$(npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*$HBM_TOTAL" | tail -1 | grep -oE "^[0-9]+")
  if [ -z "$v" ]; then
    v=$(npu-smi info 2>/dev/null | awk -v n="$1" -v t="$HBM_TOTAL" \
      '$1=="|" && $2 ~ /^[0-9]+$/ && $3==n { if (match($0, "[0-9]+[ ]*/[ ]*" t)) { s=substr($0, RSTART, RLENGTH); sub(/[ ].*/, "", s); print s; exit }; if ((getline) > 0 && match($0, "[0-9]+[ ]*/[ ]*" t)) { s=substr($0, RSTART, RLENGTH); sub(/[ ].*/, "", s); print s; exit } }')
  fi
  echo "$v"
}
need=$(( TP * PP * DP ))
pick_cards(){
  local picked=() c
  for c in $CARDS; do
    [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null && picked+=("$c")
    [ ${#picked[@]} -ge $need ] && break
  done
  [ ${#picked[@]} -ge $need ] && echo "${picked[*]}" && return 0
  return 1
}
wait_cards(){
  local i; for i in $(seq 1 60); do
    local cs; cs=$(pick_cards) && { echo "$cs"; return 0; }
    sleep 30
  done
  return 1
}
wait_health(){
  local i; for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done; echo 0
}
stop_pgid(){ kill -- -"$1" 2>/dev/null || true; sleep 5; kill -9 -- -"$1" 2>/dev/null || true; sleep 3; }
emit(){ # key=value pairs -> one json line
  "$PY" - "$@" <<'PYEOF'
import json, sys
rec = {}
for kv in sys.argv[1:]:
    k, _, v = kv.partition("=")
    rec[k] = v
def numify(d):
    for k in list(d):
        try: d[k] = float(d[k]) if "." in d[k] else int(d[k])
        except ValueError: pass
numify(rec)
print(json.dumps(rec, ensure_ascii=False))
PYEOF
}

CFG=$ROOT/cfg_matrix_${NAME}_${STATE}.json
if [ "$STATE" = t3 ]; then
  cat > "$CFG" <<CFGJSON
{"dump": {"dump_dir": "$ROOT/dump_${NAME}_${STATE}", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true}, "spec_acceptance": {"enabled": true}}}
CFGJSON
  ADDCFG="{\"runtime_config_path\": \"$CFG\", \"runtime_config_hot_reload\": true, \"runtime_report_dir\": \"$ROOT/report_${NAME}_${STATE}\"}"
else
  ADDCFG="{}"
fi

log "=== RUN name=$NAME state=$STATE tp=$TP pp=$PP dp=$DP need=$need model=$MODEL ==="
CS=$(wait_cards) || { log "no idle cards for $NAME $STATE"; emit "host=$(hostname 2>/dev/null)" "name=$NAME" "state=$STATE" "tp=$TP" "pp=$PP" "dp=$DP" "verdict=NO_CARDS" | grep -v '^$' >> "$JSONL"; exit 0; }
DEV=$(echo "$CS" | tr ' ' ',')
log "cards=$DEV"

T3ARGS=()
[ "$STATE" = t3 ] && T3ARGS=(--additional-config "$ADDCFG")
DPARGS=()
[ "${DP:-1}" -gt 1 ] && DPARGS=(--data-parallel-size "$DP")

T0=$(date +%s)
( cd "$PRODUCT" && env PYTHONPATH="$V030:$PRODUCT:${PYTHONPATH:-}" \
    ASCEND_RT_VISIBLE_DEVICES=$DEV VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name m --port "$PORT" \
    --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
    "${DPARGS[@]}" \
    --gpu-memory-utilization 0.85 --enforce-eager \
    "${T3ARGS[@]}" \
    > "$ROOT/serve_${NAME}_${STATE}.log" 2>&1 & )
sleep 5
pgid=$(pgrep -f "api_server.*--port $PORT" | head -1)
[ -n "$pgid" ] || pgid=$(grep -oE "APIServer pid=[0-9]+" "$ROOT/serve_${NAME}_${STATE}.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
if [ -z "$pgid" ] || [ "$(wait_health)" != 1 ]; then
  log "BOOT_FAIL name=$NAME state=$STATE"
  grep -iE "error|not support|Traceback" "$ROOT/serve_${NAME}_${STATE}.log" 2>/dev/null | head -4 | tee -a "$LOG"
  T1=$(date +%s)
  emit "host=$(hostname 2>/dev/null)" "name=$NAME" "state=$STATE" "tp=$TP" "pp=$PP" "dp=$DP" "boot_s=$((T1-T0))" "verdict=BOOT_FAIL" | grep -v '^$' >> "$JSONL"
  [ -n "$pgid" ] && stop_pgid "$pgid"
  exit 0
fi
T1=$(date +%s); BOOT=$((T1-T0))
log "health=200 boot_s=$BOOT pid=$pgid"

"$PY" - "$PORT" "$JSONL" "$NAME" "$STATE" "$TP" "$PP" "$DP" "$BOOT" "$CS" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, math, os, sys, time, urllib.request
port, out, name, state, tp, pp, dp, boot, cs = sys.argv[1:10]
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPTS = [
    ("short",  "用两句话说明什么是快速排序。", 96),
    ("medium", "写一段200字左右的文字，介绍随机梯度下降与批量梯度下降的区别。", 320),
    ("long",   "写一篇约500字的短文，主题：分布式训练中数据并行与模型并行的取舍。", 1024),
]
def post(prompt, mt):
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": mt, "temperature": 0, "seed": 42}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(op.open(req, timeout=300).read())
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:120]}"}
    wall = time.time() - t0
    ct = r.get("usage", {}).get("completion_tokens", 0)
    return {"wall": round(wall, 2), "ct": ct, "tps": round(ct / wall, 2) if wall > 0 else 0.0}
post("warmup", 8)  # one tiny warmup
rows, errs = [], []
for rnd in (1, 2):
    for tag, p, mt in PROMPTS:
        res = post(p, mt)
        if "err" in res:
            errs.append(f"{tag}:{res['err']}"); continue
        rows.append(res["tps"])
        with open(out, "a") as f:
            f.write(json.dumps({"host": os.uname().nodename, "name": name, "state": state,
                "tp": int(tp), "pp": int(pp), "dp": int(dp), "tag": tag, "round": rnd,
                "boot_s": int(boot), **res}, ensure_ascii=False) + "\n")
gm = math.exp(sum(math.log(t) for t in rows) / len(rows)) if rows else 0.0
with open(out, "a") as f:
    f.write(json.dumps({"host": os.uname().nodename, "name": name, "state": state,
        "tp": int(tp), "pp": int(pp), "dp": int(dp), "boot_s": int(boot), "cards": cs,
        "gm_tok_s": round(gm, 3), "n_ok": len(rows),
        "verdict": "MEASURE_FAIL" if errs and not rows else ("PARTIAL" if errs else "OK"),
        "errs": ";".join(errs)[:200]}, ensure_ascii=False) + "\n")
print(f"MEASURE name={name} state={state} gm={gm:.3f} n={len(rows)} errs={len(errs)}")
PYEOF

stop_pgid "$pgid"
sleep 3
PEAK=0
for c in $CS; do v=$(hbm "$c"); [ -n "$v" ] && [ "$v" -gt "$PEAK" ] && PEAK=$v; done
log "peak_hbm=$PEAK verdict recorded"
exit 0
