#!/usr/bin/env bash
set -uo pipefail
# C5+C6 combined A/B (2026-09-20 redesign), tuned for 162 (A3, 8x64GB, bare metal
# python inside the nightly-main-a3 container, /home bind-mounted).
#   Arm A = product guard tree, ALL detectors on, reload 2s, dump timing shim on:
#     phase 1  C5: manual_dump per corpus tier (1k..128k) -> D2H + save ms per rank
#     phase 2  C6 stress: 3 legacy tiers + 6 LongBench tiers, seq-numbered prompts
#   Arm B = bare base tree (no runtime_guard at all): same stress, no C5 phase.
#   phase 3  both arms: settle 30s -> sample RSS(process tree)+HBM 300s.
# Verdict = differential (arm A leak minus arm B leak), replacing the old
# absolute 30MB bar. Post-dump residue lands in arm A's window by construction.
# Reusable: override ROOT/PORT/CARDS/RUNNER env to run a second pass (e.g. v2).
PY=${RG_C56_PY:-/usr/local/python3.12.13/bin/python}
ENVDIR=${RG_C56_ENVDIR:-/home/d00824595/rg162_env}
PRODUCT=$ENVDIR/product
BASE=$ENVDIR/base
SCRIPTS=$ENVDIR/scripts
CORPUS=$ENVDIR/data/longbench_corpus.jsonl
MODEL=${RG_C56_MODEL:-/home/d00824595/rg162_weights/Qwen3-Coder-30B-A3B-Instruct}
RUNNER=${RG_C56_RUNNER:-0}
A_CARDS=${RG_C56_A_CARDS:-0,1}
B_CARDS=${RG_C56_B_CARDS:-2,3}
PORT_A=${RG_C56_PORT_A:-8130}
PORT_B=${RG_C56_PORT_B:-8131}
ROOT=${RG_C56_ROOT:-/tmp/rg_c56}
RES=${RG_C56_RES:-$ENVDIR/results/c56_ab_$(date +%m%d_%H%M)}
STRESS_SEC=${RG_C56_STRESS_SEC:-300}
SAMPLE_SEC=${RG_C56_SAMPLE_SEC:-300}
SETTLE_SEC=${RG_C56_SETTLE_SEC:-30}
C5_TIERS=${RG_C56_C5_TIERS:-1024,4096,16384,65536,131072}
mkdir -p "$RES" "$ROOT"
LOG=$ROOT/master.log
log(){ echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

wait_health(){
  local port=$1 i
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return; }
    sleep 10
  done
  echo 0
}

boot_a(){
  mkdir -p "$ROOT/A/report" "$ROOT/A/dump"
  ( cd "$PRODUCT" &&
    PYTHONPATH="$SCRIPTS/c5_shim:$PRODUCT:${PYTHONPATH:-}" \
    RG_C5_TIMING=1 RG_C5_TIMING_OUT=$ROOT/c5_timing.jsonl \
    ASCEND_RT_VISIBLE_DEVICES="$A_CARDS" \
    VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER="$RUNNER" \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name c56 --port "$PORT_A" \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --enforce-eager \
      --additional-config "{\"runtime_config\": {\"detector\": {\"logits_finite\": {\"enabled\": true}, \"token_repeat\": {\"enabled\": true, \"window\": 32, \"repeat_sum_threshold\": 64, \"min_tokens\": 32, \"consecutive_hits\": 1}, \"output_substring\": {\"enabled\": true, \"patterns\": [\"ZZQQXXYY_NEVER_MATCH\"]}, \"spec_acceptance\": {\"enabled\": true}}, \"actions\": {\"defaults\": {\"on_trigger\": [\"report\"]}}, \"dump\": {\"dump_dir\": \"$ROOT/A/dump\", \"auto_max_times\": 0, \"manual_dump\": 0}}, \"runtime_config_reload_interval\": 2, \"runtime_config_path\": \"$ROOT/A/runtime_config.json\", \"runtime_report_dir\": \"$ROOT/A/report\"}" \
    > "$ROOT/A/serve.log" 2>&1 & echo $! )
}

boot_b(){
  mkdir -p "$ROOT/B"
  ( cd "$BASE" &&
    PYTHONPATH="$BASE:${PYTHONPATH:-}" \
    ASCEND_RT_VISIBLE_DEVICES="$B_CARDS" \
    VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER="$RUNNER" \
    setsid "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name c56 --port "$PORT_B" \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --enforce-eager \
    > "$ROOT/B/serve.log" 2>&1 & echo $! )
}

kill_arm(){
  kill -- -"${1:-0}" 2>/dev/null || true
  sleep 3
  kill -9 -- -"${1:-0}" 2>/dev/null || true
}

log "=== C5+C6 AB start model=$MODEL runner=$RUNNER A_cards=$A_CARDS B_cards=$B_CARDS ==="
PID_A=$(boot_a); PID_B=$(boot_b)
log "boot A pid=$PID_A B pid=$PID_B"
OK_A=$(wait_health "$PORT_A")
OK_B=$(wait_health "$PORT_B")
log "health A=$OK_A B=$OK_B"
if [ "$OK_A" != 1 ] || [ "$OK_B" != 1 ]; then
  log "HEALTH TIMEOUT, aborting"; tail -30 "$ROOT/A/serve.log" "$ROOT/B/serve.log" >> "$LOG" 2>/dev/null
  kill_arm "$PID_A"; kill_arm "$PID_B"; exit 1
fi

log "--- phase 1: C5 dump timing (arm A only) ---"
RG_C56_CORPUS="$CORPUS" RG_C56_MODEL="$MODEL" "$PY" "$SCRIPTS/c56_driver.py" c5 \
  --port "$PORT_A" --config "$ROOT/A/runtime_config.json" \
  --timing "$ROOT/c5_timing.jsonl" --tiers "$C5_TIERS" \
  | tee "$RES/c5_groups.jsonl"
log "C5 phase done"

log "--- phase 2: stress both arms ${STRESS_SEC}s ---"
RG_C56_CORPUS="$CORPUS" RG_C56_MODEL="$MODEL" "$PY" "$SCRIPTS/c56_driver.py" stress \
  --port "$PORT_A" --duration "$STRESS_SEC" > "$RES/stress_A.json" &
SA=$!
RG_C56_CORPUS="$CORPUS" RG_C56_MODEL="$MODEL" "$PY" "$SCRIPTS/c56_driver.py" stress \
  --port "$PORT_B" --duration "$STRESS_SEC" > "$RES/stress_B.json" &
SB=$!
wait $SA $SB
log "stress done: A=$(cat "$RES/stress_A.json") B=$(cat "$RES/stress_B.json")"

log "--- phase 3: leakback settle ${SETTLE_SEC}s + sample ${SAMPLE_SEC}s ---"
"$PY" "$SCRIPTS/c56_driver.py" sample \
  --arms "A:$PID_A:${A_CARDS/,/+},B:$PID_B:${B_CARDS/,/+}" \
  --settle "$SETTLE_SEC" --duration "$SAMPLE_SEC" --interval 10 \
  --phase leakback --out "$RES/leakback.jsonl"

kill_arm "$PID_A"; kill_arm "$PID_B"
sleep 5
pkill -9 -f 'VLLM::EngineCor[e]' 2>/dev/null || true
pkill -9 -f '[V]LLMWorker' 2>/dev/null || true

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
print("== C6 leakback (differential) ==")
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
print("== C5 dump timing (from c5_groups.jsonl) ==")
for r in load("c5_groups.jsonl"):
    if "error" in r:
        print(f"tier={r['tier']} rep={r['rep']} ERROR {r['error']}"); continue
    ranks = r.get("ranks") or {}
    for tag, v in sorted(ranks.items()):
        print(f"tier={r['tier']:>7} rep={r['rep']} rank={tag} prompt_tok={r['prompt_tokens']} "
              f"D2H={v.get('d2h_ms')}ms save={v.get('save_ms', 0)}ms "
              f"layers={v.get('layers')} bytes={v.get('d2h_bytes', 0)/2**20:.0f}MiB "
              f"files={v.get('save_files', 0)}")
PYVERDICT
cp "$ROOT/c5_timing.jsonl" "$RES/c5_timing_raw.jsonl" 2>/dev/null || true
du -sh "$ROOT/A/dump" >> "$RES/verdict.txt" 2>/dev/null || true
log "=== DONE results in $RES ==="
