#!/usr/bin/env bash
# tip9 multi-model / multi-parallel matrix on latest config branch (70d219d68).
# Per combo: T1->T2->T3 one rotation, perf via perf_lib REQS, plus a manual
# kv-cache dump + rank-dir structure check on the first T3 serve.
# T0 SKIPPED: rg-tip9-t0 cannot boot in this container (aclnnAddRmsNormBias
# missing from CANN 9.1 libopapi.so). C1 is covered globally by the tip7 clean
# rerun (guard src identical to tip9, zero diff).
set -uo pipefail
PY=/opt/slime/venv/bin/python
V029=/data0/test-mrv2-cann91/vllm029_pkgs
PRODUCT=/data0/test-mrv2-cann91/rg-tip9-verify
T0TREE=/data0/test-mrv2-cann91/rg-tip9-t0
PERF_DIR=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/test/perf
ROOT=/data0/test-mrv2-cann91/rg_tip9_matrix
LOG=$ROOT/master.log
PORT=8145
mkdir -p "$ROOT"
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

MATRIX=(
  "m1_qwen7b_tp2|/data0/weights/Qwen2.5-7B-Instruct|--tensor-parallel-size 2|0,3"
  "m2_qwen7b_pp2|/data0/weights/Qwen2.5-7B-Instruct|--pipeline-parallel-size 2|0,3"
  "m3_qwen7b_dp2|/data0/weights/Qwen2.5-7B-Instruct|--data-parallel-size 2|0,3"
  "m4_dsv2lite_tp2|/data0/weights/DeepSeek-V2-Lite|--tensor-parallel-size 2|0,3"
  "m5_qwen3_8b_tp2|/data0/weights/Qwen3-8B|--tensor-parallel-size 2|0,3"
  "m6_qwen3_32b_tp4|/data0/weights/Qwen3-32B|--tensor-parallel-size 4|0,3,4,5"
)

hbm(){ npu-smi info 2>/dev/null | grep -A1 "^| $1     910" | grep -oE "[0-9]+[ ]*/[ ]*32768" | tail -1 | grep -oE "^[0-9]+"; }
wait_idle_cards(){
  local cards="$1" i c ok
  for i in $(seq 1 60); do
    ok=1
    IFS=',' read -ra _cs <<< "$cards"
    for c in "${_cs[@]}"; do
      [ "$(hbm "$c")" -lt 5000 ] 2>/dev/null || { ok=0; break; }
    done
    [ "$ok" = 1 ] && return 0
    sleep 20
  done
  return 1
}
wait_health(){
  local port=$1 i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done
  echo 0
}
stop_pid(){
  kill -- -"$1" 2>/dev/null || true; sleep 3
  kill -9 -- -"$1" 2>/dev/null || true; sleep 2
}

write_t2_cfg(){
  cat > "$1" <<EOF
{"reload_interval_seconds": 3, "dump": {"dump_dir": "$2", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "detector": {"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "output_substring": {"enabled": false}, "spec_acceptance": {"enabled": false}}}
EOF
}
write_t3_cfg(){
  cat > "$1" <<EOF
{"reload_interval_seconds": 3, "dump": {"dump_dir": "$2", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "actions": {"defaults": {"on_trigger": ["report"]}}, "report": {"save_sensitive_info": false, "max_per_req": 1}, "runtime_report_dir": "$3", "detector": {"logits_finite": {"enabled": true}, "token_repeat": {"enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 32, "consecutive_hits": 1}, "output_substring": {"enabled": true, "patterns": [], "add_special_tokens": false, "match_prefix": false}, "spec_acceptance": {"enabled": true, "window": 10, "low_threshold": 0.3, "len_low_threshold": 1.4, "high_threshold": 0.96, "len_high_threshold": 2.8}}}
EOF
}

serve_state(){
  # tree label model pargs cards cfgpath reload
  local tree=$1 label=$2 model=$3 pargs=$4 cards=$5 cfgpath=$6 reload=$7
  local add=()
  if [ -n "$cfgpath" ]; then
    add=(--additional-config "{\"runtime_config_path\": \"$cfgpath\", \"runtime_config_reload_interval\": $reload, \"runtime_report_dir\": \"$mdir/report\"}")
  fi
  ( cd "$tree"
    env PYTHONPATH="$V029:$tree:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=$cards VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$model" --served-model-name dsv2 --port "$PORT" $pargs \
      --gpu-memory-utilization 0.85 --enforce-eager "${add[@]}" \
      > "$mdir/serve_${label}.log" 2>&1 & echo $! )
}

measure(){
  local state=$1 out=$2
  ( cd "$PERF_DIR"
    RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" RG_PERF_OUT="$out" \
    RUNNER=v2 RG_STATE="$state" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY'
import json, os
from perf_lib import warmup, post, REQS
out = os.environ["RG_PERF_OUT"]; state = os.environ["RG_STATE"]
warmup(rounds=1)
with open(out, "a") as f:
    for tag, p, mt in REQS:
        wall, pt, ct, tps = post(p, mt)
        f.write(json.dumps({"state": state, "tag": tag, "wall_s": round(wall, 2),
                            "compl_tok": ct, "out_tok_s": round(tps, 2)}) + "\n")
print("measured", flush=True)
PY
  )
}

manual_dump_check(){
  # Arm manual_dump, wait for the reload thread to pick it up, then fire a
  # long generation so the dump catches a live decode wave (mirrors the
  # smoke sequence that reliably produced dumps).
  "$PY" - "$mdir/cfg_t3.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["dump"]["manual_dump"] = True
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PY
  sleep 6
  curl -s --max-time 180 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d '{"model":"dsv2","prompt":"Write a long story about the sea, ships and sailors crossing the ocean","max_tokens":300,"temperature":0}' > "$mdir/manual_dump_req.json" 2>/dev/null &
  local rpid=$!
  sleep 20
  "$PY" - "$mdir" <<'PY'
import sys, glob, os, json
import torch
mdir = sys.argv[1]
pt = sorted(glob.glob(f"{mdir}/dump/**/*.pt", recursive=True))
ranks = sorted({os.path.basename(os.path.dirname(p)) for p in pt})
others = [f for f in glob.glob(f"{mdir}/dump/**/*", recursive=True) if not f.endswith(".pt") and os.path.isfile(f)]
print(f"DUMP_CHECK files={len(pt)} ranks={ranks} info_files={len(others)}")
for f in others[:6]: print("  info:", os.path.relpath(f, mdir))
if pt:
    d = torch.load(pt[0], map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        ks = {k: (tuple(v.shape), str(v.dtype)) for k, v in d.items() if hasattr(v, "shape")}
        print("  sample keys:", json.dumps({k: list(map(str, s)) for k, s in list(ks.items())[:4]}, ensure_ascii=False))
    else:
        print("  sample type:", type(d).__name__)
PY
  wait "$rpid" 2>/dev/null || true
  "$PY" - "$mdir/cfg_t3.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["dump"]["manual_dump"] = False
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PY
}

summary_ratio(){
  "$PY" - "$mdir/matrix.jsonl" <<'PY'
import json, math, sys
def gm(st):
    v = [json.loads(l)["out_tok_s"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["state"] == st]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else 0.0
t0, t1, t2, t3 = gm("T0"), gm("T1"), gm("T2"), gm("T3")
parts = []
if t0 > 0 and t1 > 0: parts.append(f"C1 T1/T0={t1/t0:.5f} (>=0.999)")
if t1 > 0 and t2 > 0: parts.append(f"C2 T2/T1={t2/t1:.5f} (>=0.999)")
if t2 > 0 and t3 > 0: parts.append(f"C3 T3/T2={t3/t2:.5f} (>=0.990)")
print("  ".join(parts))
print(f"geom T0={t0:.1f} T1={t1:.1f} T2={t2:.1f} T3={t3:.1f} tok/s")
PY
}

for entry in "${MATRIX[@]}"; do
  IFS='|' read -r name model pargs cards <<< "$entry"
  mdir="$ROOT/$name"
  mkdir -p "$mdir/dump" "$mdir/report"
  : > "$mdir/matrix.jsonl"
  log "=== [$name] model=$model pargs='$pargs' cards=$cards ==="
  wait_idle_cards "$cards" || { log "[$name] ABORT cards busy"; continue; }

  for st in T1 T2 T3; do
    case $st in
      T0) tree=$T0TREE; cfg=""; reload=0 ;;
      T1) tree=$PRODUCT; cfg=""; reload=0 ;;
      T2) tree=$PRODUCT; cfg="$mdir/cfg_t2.json"; reload=3; write_t2_cfg "$cfg" "$mdir/dump" ;;
      T3) tree=$PRODUCT; cfg="$mdir/cfg_t3.json"; reload=3; write_t3_cfg "$cfg" "$mdir/dump" "$mdir/report" ;;
    esac
    pid=$(serve_state "$tree" "t${st}" "$model" "$pargs" "$cards" "$cfg" "$reload")
    ok=$(wait_health "$PORT")
    if [ "$ok" != 1 ]; then
      log "[$name/$st] HEALTH_TIMEOUT"
      grep -iE "error|exception|Traceback|not support" "$mdir/serve_t${st}.log" | head -5 | tee -a "$LOG"
      stop_pid "$pid"; wait_idle_cards "$cards" || true
      log "[$name] SKIP remaining states for this combo"
      break
    fi
    log "[$name/$st] health=200"
    measure "$st" "$mdir/matrix.jsonl"
    log "[$name/$st] measured"
    if [ "$st" = "T3" ]; then manual_dump_check | tee -a "$LOG"; fi
    stop_pid "$pid"
    wait_idle_cards "$cards" || true
  done
  if grep -q '"state": "T3"' "$mdir/matrix.jsonl" 2>/dev/null; then
    summary_ratio | tee -a "$LOG"
  fi
  log "[$name] done"
done

log "=== MATRIX DONE (T0 skipped: tip9-t0 cannot boot in CANN9.1 container) ==="
