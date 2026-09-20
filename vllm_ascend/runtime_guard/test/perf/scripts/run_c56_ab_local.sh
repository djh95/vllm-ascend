#!/usr/bin/env bash
set -uo pipefail
# Local sequential variant of run_c56_ab.sh for test-mrv2-cann91 (910B4 A2,
# only cards 2+3 free): arm A (guard, all detectors on + dump timing shim)
# runs the full pipeline (C5 dump timing -> stress -> leakback), then arm B
# (bare base, no runtime_guard) repeats stress -> leakback on the same cards.
# Identical stress prompts (driver seq restarts at 1 per arm). Verdict =
# differential A-vs-B, replacing the old absolute 30MB bar; arm A's window
# includes post-dump residue by construction.
PY=${RG_C56_PY:-/opt/slime/venv/bin/python}
PRODUCT=${RG_C56_PRODUCT:-/data0/test-mrv2-cann91/vllm-ascend}
BASE=${RG_C56_BASE:-/data0/test-mrv2-cann91/rg-perf-t0}
ENVDIR=${RG_C56_ENVDIR:-/data0/test-mrv2-cann91/rg_c56}
SCRIPTS=$ENVDIR/scripts
CORPUS=$ENVDIR/data/longbench_corpus.jsonl
MODEL=${RG_C56_MODEL:-/data0/weights/DeepSeek-V2-Lite}
CARDS=${RG_C56_CARDS:-2,3}
NPU_A=${CARDS/,/+}
RUNNER=${RG_C56_RUNNER:-0}
ROOT=$ENVDIR/run
RES=${RG_C56_RES:-$ENVDIR/results/c56_local_$(date +%m%d_%H%M)}
STRESS_SEC=${RG_C56_STRESS_SEC:-300}
SAMPLE_SEC=${RG_C56_SAMPLE_SEC:-300}
SETTLE_SEC=${RG_C56_SETTLE_SEC:-30}
C5_TIERS=${RG_C56_C5_TIERS:-1024,4096,16384,65536,131072}
mkdir -p "$RES" "$ROOT"
LOG=$ROOT/master.log
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

wait_idle(){
  local i out ok
  for i in $(seq 1 60); do
    out=$(npu-smi info 2>/dev/null || true); ok=1
    for c in ${CARDS//,/ }; do
      grep -q "No running processes found in NPU $c" <<< "$out" || ok=0
    done
    [ "$ok" = 1 ] && return 0
    sleep 5
  done
  return 1
}
wait_health(){
  local port=$1 i
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done
  echo 0
}
kill_arm(){ kill -- -"${1:-0}" 2>/dev/null || true; sleep 3; kill -9 -- -"${1:-0}" 2>/dev/null || true; }

boot_guard(){
  mkdir -p "$ROOT/A/report" "$ROOT/A/dump"
  ( cd "$PRODUCT" &&
    PYTHONPATH="$SCRIPTS/c5_shim:$PRODUCT" \
    RG_C5_TIMING=1 RG_C5_TIMING_OUT=$ROOT/c5_timing.jsonl \
    ASCEND_RT_VISIBLE_DEVICES="$CARDS" \
    VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER="$RUNNER" \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name c56 --port 8130 \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --enforce-eager \
      --additional-config "{\"runtime_config\": {\"detector\": {\"logits_finite\": {\"enabled\": true}, \"token_repeat\": {\"enabled\": true, \"window\": 32, \"repeat_sum_threshold\": 64, \"min_tokens\": 32, \"consecutive_hits\": 1}, \"output_substring\": {\"enabled\": true, \"patterns\": [\"ZZQQXXYY_NEVER_MATCH\"]}, \"spec_acceptance\": {\"enabled\": true}}, \"actions\": {\"defaults\": {\"on_trigger\": [\"report\"]}}, \"dump\": {\"dump_dir\": \"$ROOT/A/dump\", \"auto_max_times\": 0, \"manual_dump\": 0}}, \"runtime_config_reload_interval\": 2, \"runtime_config_path\": \"$ROOT/A/runtime_config.json\", \"runtime_report_dir\": \"$ROOT/A/report\"}" \
    > "$ROOT/A/serve.log" 2>&1 & echo $! )
}
boot_base(){
  mkdir -p "$ROOT/B"
  ( cd "$BASE" &&
    PYTHONPATH="$BASE" \
    ASCEND_RT_VISIBLE_DEVICES="$CARDS" \
    VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER="$RUNNER" \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name c56 --port 8131 \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --enforce-eager \
    > "$ROOT/B/serve.log" 2>&1 & echo $! )
}
stress_arm(){
  RG_C56_CORPUS="$CORPUS" RG_C56_MODEL="$MODEL" "$PY" "$SCRIPTS/c56_driver.py" stress \
    --port "$1" --duration "$STRESS_SEC" > "$RES/stress_$2.json"
}

log "=== C5+C6 local AB start model=$MODEL cards=$CARDS runner=$RUNNER ==="

log "--- arm A boot (guard) ---"
wait_idle || { log "cards busy, abort"; exit 1; }
PID_A=$(boot_guard)
OK_A=$(wait_health 8130)
log "arm A health=$OK_A pid=$PID_A"
if [ "$OK_A" != 1 ]; then
  log "arm A HEALTH TIMEOUT"; tail -30 "$ROOT/A/serve.log" >> "$LOG" 2>/dev/null
  kill_arm "$PID_A"; exit 1
fi

log "--- phase 1: C5 dump timing (arm A) ---"
RG_C56_CORPUS="$CORPUS" RG_C56_MODEL="$MODEL" "$PY" "$SCRIPTS/c56_driver.py" c5 \
  --port 8130 --config "$ROOT/A/runtime_config.json" \
  --timing "$ROOT/c5_timing.jsonl" --tiers "$C5_TIERS" \
  | tee "$RES/c5_groups.jsonl"
log "C5 done"

log "--- phase 2: arm A stress ${STRESS_SEC}s ---"
stress_arm 8130 A
log "arm A stress: $(cat "$RES/stress_A.json")"

log "--- phase 3: arm A leakback ---"
"$PY" "$SCRIPTS/c56_driver.py" sample \
  --arms "A:$PID_A:$NPU_A" --settle "$SETTLE_SEC" --duration "$SAMPLE_SEC" \
  --interval 10 --phase leakback --out "$RES/leakback.jsonl"
kill_arm "$PID_A"; sleep 5
pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true
wait_idle || true

log "--- arm B boot (bare base) ---"
PID_B=$(boot_base)
OK_B=$(wait_health 8131)
log "arm B health=$OK_B pid=$PID_B"
if [ "$OK_B" != 1 ]; then
  log "arm B HEALTH TIMEOUT"; tail -30 "$ROOT/B/serve.log" >> "$LOG" 2>/dev/null
  kill_arm "$PID_B"; exit 1
fi
log "--- phase 2b: arm B stress ${STRESS_SEC}s ---"
stress_arm 8131 B
log "arm B stress: $(cat "$RES/stress_B.json")"
log "--- phase 3b: arm B leakback ---"
"$PY" "$SCRIPTS/c56_driver.py" sample \
  --arms "B:$PID_B:$NPU_A" --settle "$SETTLE_SEC" --duration "$SAMPLE_SEC" \
  --interval 10 --phase leakback --out "$RES/leakback.jsonl"
kill_arm "$PID_B"; sleep 5
pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true

"$PY" - "$RES" <<'PYVERDICT' | tee "$RES/verdict.txt"
import json, sys, os
res = sys.argv[1]
def load(p):
    try:
        return [json.loads(l) for l in open(os.path.join(res, p)) if l.strip()]
    except Exception:
        return []
rows = load("leakback.jsonl")
arms = {}
for r in rows:
    if r.get("phase") != "leakback":
        continue
    arms.setdefault(r["arm"], []).append(r)
print("== C6 leakback (differential, sequential arms) ==")
deltas = {}
for arm, rs in sorted(arms.items()):
    if len(rs) < 3:
        print(f"{arm}: insufficient samples"); continue
    def rss(x): return x["rss_kb"]
    def hbm(x): return sum(x.get("hbm_mb", {}).values())
    rss_first, rss_last = rss(rs[0]), sum(map(rss, rs[-3:])) / 3
    hbm_first, hbm_last = hbm(rs[0]), sum(map(hbm, rs[-3:])) / 3
    d_rss = (rss_last - rss_first) / 1024.0
    d_hbm = hbm_last - hbm_first
    deltas[arm] = (d_rss, d_hbm)
    print(f"{arm}: rss {rss_first/1024:.0f}MB -> {rss_last/1024:.0f}MB (delta {d_rss:+.1f}MB), "
          f"hbm {hbm_first:.0f}MB -> {hbm_last:.0f}MB (delta {d_hbm:+.1f}MB), n={len(rs)}")
if "A" in deltas and "B" in deltas:
    diff_rss = deltas["A"][0] - deltas["B"][0]
    diff_hbm = deltas["A"][1] - deltas["B"][1]
    verdict = "PASS" if (diff_rss < 10 and diff_hbm < 512) else "FLAG"
    print(f"guard-vs-base differential: rss {diff_rss:+.1f}MB hbm {diff_hbm:+.1f}MB -> {verdict}")
    print("(bar: dump+detector residue within +10MB RSS / +512MB HBM of bare base)")
print()
print("== C5 dump timing ==")
for r in load("c5_groups.jsonl"):
    if "error" in r:
        print(f"tier={r['tier']} rep={r['rep']} ERROR {r['error']}"); continue
    ranks = r.get("ranks") or {}
    if not ranks:
        print(f"tier={r['tier']} rep={r['rep']} NO DUMP EVENTS")
    for tag, v in sorted(ranks.items()):
        print(f"tier={r['tier']:>7} rep={r['rep']} rank={tag} prompt_tok={r['prompt_tokens']} "
              f"D2H={v.get('d2h_ms')}ms save={v.get('save_ms', 0)}ms "
              f"layers={v.get('layers')} bytes={v.get('d2h_bytes', 0)/2**20:.0f}MiB "
              f"files={v.get('save_files', 0)}")
PYVERDICT
cp "$ROOT/c5_timing.jsonl" "$RES/c5_timing_raw.jsonl" 2>/dev/null || true
du -sh "$ROOT/A/dump" >> "$RES/verdict.txt" 2>/dev/null || true
log "=== DONE results in $RES ==="
