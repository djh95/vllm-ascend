#!/usr/bin/env bash
# Shared P0 lifecycle base for live NPU harness (analysis branch).
# Lifecycle extracted from /data0/test-mrv2-cann91/run_tip15_tp1_smoke.sh (v4/v5):
#   pick idle card -> boot (v2/v1 worktree per RUNNER) -> health -> case logic
#   -> assert -> kill -> residual -> HBM check -> trap EXIT removes temp dumps
#   (FTL 0.4) + final df line.
# Config schema: NEW style only — "runtime_config_hot_reload" (bool) inside
# --additional-config; the legacy "reload_interval_seconds" is NOT written.
# Env overrides: PY V030 PRODUCT MODEL T0_PRODUCT SNAME ROOT PORT* HBM_TOTAL
# CARDS RUNNER KEEP_DUMP P0_IDLE_MB.
# NOTE: legacy vars (URL/REPORT_DIR/DUMP_ROOT/CONFIG_DIR/GOLDEN_DIR/df_check/
# require_cmds/cleanup_dumps) are kept for older g*/capture skeletons that
# still source this file; they are superseded by the p0_* helpers below.
set -uo pipefail

# ---- legacy compat (older skeletons source this file) ----
RUNNER="${RUNNER:-v2}"
case "$RUNNER" in
  v2) export VLLM_USE_V2_MODEL_RUNNER=1 ;;
  v1) export VLLM_USE_V2_MODEL_RUNNER=0 ;;
  *) echo "RUNNER must be v1 or v2, got: $RUNNER" >&2; exit 2 ;;
esac
HOST_="${HOST:-127.0.0.1}"
URL="${RG_LIVE_URL:-http://${HOST_}:${PORT:-8017}/v1/completions}"
REPORT_DIR="${RG_REPORT_DIR:-./runtime/report}"
DUMP_ROOT="${DUMP_ROOT:-${REPORT_DIR}/kv_cache}"
KEEP_DUMP="${KEEP_DUMP:-0}"
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../configs" && pwd)"
GOLDEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../golden" && pwd)"
df_check() { mkdir -p "$(dirname "${1:-$DUMP_ROOT}")" 2>/dev/null || true; df -h "$(dirname "${1:-$DUMP_ROOT}")" 2>/dev/null || df -h .; }
require_cmds() { command -v curl >/dev/null; command -v python3 >/dev/null; }
cleanup_dumps() { :; }  # superseded by p0_exit (P0_DUMP_DIRS list)

# ---- P0 lifecycle base ----
PY=${PY:-/opt/slime/venv/bin/python}
V030=${V030:-/data0/test-mrv2-cann91/vllm030_pkgs}
P0_PRODUCT_V2=/data0/test-mrv2-cann91/rg-tip15-8f5e3
P0_PRODUCT_V1=/data0/test-mrv2-cann91/rg-v1-9343
if [ -z "${PRODUCT:-}" ]; then
  if [ "$RUNNER" = v1 ]; then PRODUCT=$P0_PRODUCT_V1; else PRODUCT=$P0_PRODUCT_V2; fi
fi
T0_PRODUCT=${T0_PRODUCT:-/data0/test-mrv2-cann91/rg-t0-b1b58}
MODEL=${MODEL:-/data0/weights/Qwen2.5-7B-Instruct}
SNAME=${SNAME:-p0m}
HBM_TOTAL=${HBM_TOTAL:-32768}
CARDS=${CARDS:-"0 1 2 3 4 5 6 7"}
P0_IDLE_MB=${P0_IDLE_MB:-5000}
VERIFY_TOOL=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/analysis/scripts/verify_request_kv.py

ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_p0_xxx_${RUNNER}}
mkdir -p "$ROOT"
LOG=$ROOT/master.log
PASS=1
PGID=""
P0_TAG="pre"
P0_DUMP_DIRS=()

log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }
fail(){ PASS=0; log "FAIL: $*"; }

hbm(){
  local v
  v=$(npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*$HBM_TOTAL" | tail -1 | grep -oE "^[0-9]+")
  if [ -z "$v" ]; then
    v=$(npu-smi info 2>/dev/null | awk -v n="$1" -v t="$HBM_TOTAL" \
      '$1=="|" && $2 ~ /^[0-9]+$/ && $3==n { if (match($0, "[0-9]+[ ]*/[ ]*" t)) { s=substr($0, RSTART, RLENGTH); sub(/[ ].*/, "", s); print s; exit }; if ((getline) > 0 && match($0, "[0-9]+[ ]*/[ ]*" t)) { s=substr($0, RSTART, RLENGTH); sub(/[ ].*/, "", s); print s; exit } }')
  fi
  echo "$v"
}
pick_idle_card(){
  local c; for c in $CARDS; do
    if [ "$(hbm "$c")" -lt "$P0_IDLE_MB" ] 2>/dev/null; then echo "$c"; return 0; fi
  done
  return 1
}
wait_idle_card(){
  local i c; for i in $(seq 1 90); do
    c=$(pick_idle_card) && { echo "$c"; return 0; }
    sleep 20
  done
  return 1
}
wait_health(){
  local port=$1 i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return 0; }
    sleep 10
  done; echo 0
}
stop_pgid(){ kill -- -"$1" 2>/dev/null || true; sleep 5; kill -9 -- -"$1" 2>/dev/null || true; sleep 3; }
residual_check(){
  local label=$1 pgid=$2 out
  sleep 3
  out=$(ps -eo pid,ppid,pgid,etime,args | awk -v g="$pgid" '$3==g && $1!=g' | grep -v defunct | head -20)
  if [ -n "$out" ]; then
    log "RESIDUAL_AFTER $label: survivors (force kill):"; echo "$out" | tee -a "$LOG"
    kill -9 -- -"$pgid" 2>/dev/null || true
  else
    log "residual $label: none"
  fi
}
hbm_release_check(){  # $1=card $2=label
  local h; h=$(hbm "$1")
  if [ -n "$h" ] && [ "$h" -lt "$P0_IDLE_MB" ] 2>/dev/null; then
    log "HBM released on card $1 (${h}MB / ${HBM_TOTAL}MB) after $2"
  else
    fail "HBM not released on card $1 (${h:-NA}MB) after $2"
  fi
}
p0_df_kb(){  # $1=tag ; echoes ONLY the used-KB number (log goes to file, keeps $() clean)
  local u
  u=$(df -k "$ROOT" 2>/dev/null | tail -1 | awk '{print $(NF-3)}')
  echo "$(date '+%F %H:%M:%S') DF_USED_KB $1 ${u:-NA}" >> "$LOG"
  echo "${u:-0}"
}

# Write a NEW-schema runtime_config.json. $1=file $2=manual_dump(true|false)
# $3=detector json fragment $4=on_trigger (default ["report"]).
# Registers the dump dir for trap-EXIT removal (FTL 0.4) unless KEEP_DUMP=1.
p0_write_cfg(){
  local f=$1 manual=$2 det=$3 trig=${4:-'["report"]'}
  mkdir -p "$(dirname "$f")"
  cat > "$f" <<CFG
{"dump": {"dump_dir": "$(dirname "$f")/dump", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": ${manual}}, "actions": {"defaults": {"on_trigger": ${trig}}}, "detector": ${det}}
CFG
  P0_DUMP_DIRS+=("$(dirname "$f")/dump")
}

# Boot a detached api_server. $1=tag $2=card $3=port [$4=product_override("-"=PRODUCT)]
# [$5=mode: "guard"(default)|"plain"]  — plain omits runtime_config keys (T0 baseline).
p0_boot(){
  local tag=$1 card=$2 port=$3 prod=${4:--} mode=${5:-guard}
  local d="$ROOT/$tag"
  mkdir -p "$d/report"
  [ "$prod" = "-" ] && prod=$PRODUCT
  local -a xtra=()
  if [ "$mode" != plain ]; then
    xtra=(--additional-config "{\"runtime_config_path\": \"$d/runtime_config.json\", \"runtime_config_hot_reload\": true, \"runtime_report_dir\": \"$d/report\"}")
  fi
  ( cd "$prod" && env PYTHONPATH="$V030:$prod:${PYTHONPATH:-}" \
      ASCEND_RT_VISIBLE_DEVICES=$card VLLM_BATCH_INVARIANT=1 \
      VLLM_USE_V2_MODEL_RUNNER=$([ "$RUNNER" = v2 ] && echo 1 || echo 0) \
      setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name "$SNAME" --port "$port" \
      --gpu-memory-utilization 0.85 --enforce-eager \
      ${xtra[@]+"${xtra[@]}"} \
      > "$d/serve.log" 2>&1 & )
}
p0_pid(){ pgrep -f "vllm.entrypoints.openai.api_server.*--port $1" | head -1; }

# Wait pid+health for tag/port. Sets PGID/P0_TAG. rc!=0 on failure (log dumped).
p0_serve_up(){
  local tag=$1
  local port=$2
  local d="$ROOT/$tag"
  sleep 5
  PGID=$(p0_pid "$port"); P0_TAG=$tag
  [ -n "$PGID" ] || PGID=$(grep -oE "APIServer pid=[0-9]+" "$d/serve.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  log "[$tag] boot pid=${PGID:-none} port=$port sha=$(git -C "$PRODUCT" log --oneline -1 2>/dev/null | head -1) runner=$RUNNER"
  if [ -z "$PGID" ]; then
    log "[$tag] ERROR: no api_server pid"; grep -iE "error|exception|Traceback" "$d/serve.log" 2>/dev/null | head -8 | tee -a "$LOG"; return 1
  fi
  if [ "$(wait_health "$port")" != 1 ]; then
    log "[$tag] ERROR: health timeout"; grep -iE "error|exception|Traceback" "$d/serve.log" 2>/dev/null | head -8 | tee -a "$LOG"; return 1
  fi
  log "[$tag] health=200"
  return 0
}
p0_serve_down(){
  if [ -n "${PGID:-}" ]; then
    stop_pgid "$PGID"
    residual_check "${P0_TAG:-case}" "$PGID"
    PGID=""
  fi
}
p0_ask(){  # $1=port $2=prompt $3=max_tokens $4=outfile ; echoes http_code
  curl -s -o "$4" -w '%{http_code}' "http://127.0.0.1:$1/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SNAME\",\"prompt\":\"$2\",\"max_tokens\":$3,\"temperature\":0,\"seed\":42}"
}
p0_wait_pt(){  # $1=dir $2=timeout_s : rc0 once a *.pt appears under dir
  local i n=$(( ${2:-90} / 3 )) c
  for i in $(seq 1 "$n"); do
    c=$(find "$1" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
    [ "${c:-0}" -gt 0 ] && return 0
    sleep 3
  done
  return 1
}
p0_find_report(){  # $1=report_root [$2=incident_type] : first report_*.json path
  if [ -n "${2:-}" ]; then
    find "$1/$2" -maxdepth 1 -name 'report_*.json' 2>/dev/null | head -1
  else
    find "$1" -name 'report_*.json' 2>/dev/null | head -1
  fi
}
p0_body_cmp(){  # $1 $2 : byte evidence + deterministic-payload compare (id/created stripped)
  "$PY" - "$1" "$2" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
ra = open(sys.argv[1], 'rb').read(); rb = open(sys.argv[2], 'rb').read()
print(f"BODY_CMP_RAW_BYTE_IDENTICAL={ra == rb} (len {len(ra)} vs {len(rb)})")
def norm(p):
    d = json.loads(open(p, 'rb').read())
    # non-deterministic metadata, not model output: per-request random id,
    # epoch ts, and the serving build fingerprint (differs across builds by
    # design, e.g. T0 ffcfd709 vs product 9df27ce1)
    d.pop('id', None); d.pop('created', None); d.pop('system_fingerprint', None)
    return json.dumps(d, sort_keys=True, ensure_ascii=False)
na, nb = norm(sys.argv[1]), norm(sys.argv[2])
print(f"BODY_CMP_DETERMINISTIC_IDENTICAL={na == nb} (id/created/system_fingerprint stripped; byte compare on canonical form)")
if na != nb:
    da, db = json.loads(na), json.loads(nb)
    for k in sorted(set(da) | set(db)):
        if da.get(k) != db.get(k):
            print(f"  BODY_CMP_DIFF_FIELD {k}: {str(da.get(k))[:160]!r} != {str(db.get(k))[:160]!r}")
    sys.exit(1)
PYEOF
  return "${PIPESTATUS[0]}"
}
p0_report_check(){  # $1=report.json $2=expected incident_type : prints key fields; rc0 iff match
  "$PY" - "$1" "$2" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
r = json.load(open(sys.argv[1]))
exp = sys.argv[2]
det = r.get("detail") if isinstance(r.get("detail"), dict) else {}
stage = r.get("stage") or det.get("stage") or "-"
ill = r.get("ill_type", det.get("ill_type", "-"))
dump_att = r.get("dump_attempted", r.get("dump_armed", "-"))
print(f"REPORT_CHECK path={sys.argv[1]}")
print(f"REPORT_CHECK incident_type={r.get('incident_type')} req_id={r.get('req_id')} stage={stage} ill_type={ill} dump_attempted={dump_att}")
sys.exit(0 if str(r.get("incident_type") or "") == exp else 1)
PYEOF
  return "${PIPESTATUS[0]}"
}
p0_verdict(){  # $1=case token (e.g. P0_01)
  if [ "$PASS" = 1 ]; then log "$1_${RUNNER}_VERDICT PASS"; exit 0
  else log "$1_${RUNNER}_VERDICT FAIL"; exit 1; fi
}
p0_skip(){ log "$1_${RUNNER}_VERDICT SKIP ($2)"; exit 3; }

p0_exit(){
  local rc=$?
  if [ -n "${PGID:-}" ]; then
    log "trap EXIT: stopping server pgid=$PGID (unexpected-exit path)"
    stop_pgid "$PGID" || true
    residual_check "${P0_TAG:-case}" "$PGID" || true
    PGID=""
  fi
  if [ "${KEEP_DUMP:-0}" != 1 ]; then
    local d
    for d in ${P0_DUMP_DIRS[@]+"${P0_DUMP_DIRS[@]}"}; do
      if [ -d "$d" ]; then
        log "trap EXIT: FTL 0.4 removing temp dump $d"
        rm -rf "$d"
      fi
    done
  fi
  p0_df_kb trap_exit_final >/dev/null
  exit "$rc"
}
trap p0_exit EXIT

log "[live] P0 base ready RUNNER=$RUNNER PRODUCT=$PRODUCT MODEL=$MODEL ROOT=$ROOT HBM_TOTAL=$HBM_TOTAL CARDS='$CARDS' KEEP_DUMP=$KEEP_DUMP"
