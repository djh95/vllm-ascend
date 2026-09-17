#!/usr/bin/env bash
# §7 混部 (mixed deployment): H-01..H-06 — two vllm instances sharing one node/NPU.
# Covers: dump_dir isolation, log distinguishability, manual-dump arm isolation,
# per-instance disk gate, per-instance hot reload, UCM log non-hijack.
set -uo pipefail
PRODUCT=/data0/test-mrv2-cann91/vllm-ascend
CFGDIR=/data0/test-mrv2-cann91/vllm-ascend/vllm_ascend/runtime_guard/test/live/configs
PY=/opt/slime/venv/bin/python
MODEL=/data0/weights/Qwen2.5-0.5B-Instruct
SNAME=qwen05
ROOT=/tmp/rg_h_mixed
rm -rf "$ROOT"
mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.txt"; : > "$SUMMARY"
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$SUMMARY"; }

stop_own(){ local pid=$1; kill -- -"$pid" 2>/dev/null; sleep 3; kill -9 -- -"$pid" 2>/dev/null; sleep 2; }
wait_health(){ local port=$1; for i in $(seq 1 150); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo 1; return; }; sleep 5; done; echo 0; }

# mkcfg <base_json> <dump_dir> <headroom_bytes|"">  -> prints full runtime_config JSON
mkcfg(){ "$PY" - "$1" "$2" "$3" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
c.setdefault("dump",{})["dump_dir"]=sys.argv[2]
if len(sys.argv)>3 and sys.argv[3] not in ("", None):
    c.setdefault("dump",{})["free_headroom_bytes"]=int(sys.argv[3])
print(json.dumps(c,ensure_ascii=False))
PY
}

# serve <label> <port> <card> <cfg_json> [inject]
serve(){
  local label=$1 port=$2 card=$3 cfg=$4 inject=${5:-}
  local D="$ROOT/$label"; mkdir -p "$D/report" "$D/dump"
  cd "$PRODUCT"
  export PYTHONPATH="$PRODUCT:${PYTHONPATH:-}"
  export ASCEND_RT_VISIBLE_DEVICES="$card"
  export VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1
  if [ -n "$inject" ]; then export RG_INJECT="$inject"; else unset RG_INJECT; fi
  setsid "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SNAME" --port "$port" \
    --gpu-memory-utilization 0.85 --enforce-eager \
    --additional-config "{\"runtime_config\": $cfg, \"runtime_config_reload_interval\": 3, \"runtime_config_path\": \"$D/runtime_config.json\", \"runtime_report_dir\": \"$D/report\"}" \
    > "$D/serve.log" 2>&1 &
  echo $!
}

req(){ local port=$1 prompt=$2 out=$3; curl -s "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SNAME\",\"prompt\":\"$prompt\",\"max_tokens\":16,\"temperature\":0,\"seed\":42}" > "$out" 2>/dev/null || true; }
ndump(){ find "$1" -name '*.pt' 2>/dev/null | wc -l; }
nrep(){ find "$1" -name 'report_*.json' 2>/dev/null | wc -l; }

# ============ L1: H-01 dump_dir isolation + H-02 log + H-06 UCM ============
log "===== H-01/H-02/H-06 (two instances, nan inject, auto dump) ====="
CFG_A=$(mkcfg "$CFGDIR/k02_on_trigger_dump.json" "$ROOT/h01a/dump" "")
CFG_B=$(mkcfg "$CFGDIR/k02_on_trigger_dump.json" "$ROOT/h01b/dump" "")
PID_A=$(serve h01a 8041 0 "$CFG_A" nan_logits)
PID_B=$(serve h01b 8042 1 "$CFG_B" nan_logits)
log "[h01] pids A=$PID_A B=$PID_B"
OKA=$(wait_health 8041); OKB=$(wait_health 8042)
log "[h01] health A=$OKA B=$OKB"
if [ "$OKA" = 1 ] && [ "$OKB" = 1 ]; then
  req 8041 "mixed a" "$ROOT/h01a/resp.json"
  req 8042 "mixed b" "$ROOT/h01b/resp.json"
  sleep 8
  DA=$(ndump "$ROOT/h01a/dump"); DB=$(ndump "$ROOT/h01b/dump")
  RA=$(nrep "$ROOT/h01a/report"); RB=$(nrep "$ROOT/h01b/report")
  log "[H-01] dumps A=$DA B=$DB (both expect >=1) reports A=$RA B=$RB"
  # cross-write: no .pt under A referencing B req, and vice versa. Reports carry req_id.
  cross_a=$("$PY" - "$ROOT/h01a/report" <<'PY'
import json,sys,glob,os
d=sys.argv[1]
ids=set()
for f in glob.glob(os.path.join(d,'**','report_*.json'), recursive=True):
    try:
        r=json.load(open(f)); ids.add(r.get('req_id') or r.get('request_id'))
    except Exception: pass
print(len(ids))
PY
  )
  log "[H-01] distinct req_ids in A reports=$cross_a"
  log "[H-01] PASS if A/B dumps>=1 and report dirs disjoint (A:${RA} B:${RB})"
  # H-02: logs distinguishable by instance dir + each has its own port/boot line
  ba=$(grep -cE "8041|port.*8041" "$ROOT/h01a/serve.log" 2>/dev/null || echo 0)
  bb=$(grep -cE "8042|port.*8042" "$ROOT/h01b/serve.log" 2>/dev/null || echo 0)
  log "[H-02] A log port-hits=$ba B log port-hits=$bb (both expect >0)"
  # H-06: no cross-instance log hijack — each log file only contains its own instance boot.
  log "[H-06] UCM/log: A log lines=$(wc -l < "$ROOT/h01a/serve.log") B log lines=$(wc -l < "$ROOT/h01b/serve.log")"
  grep -iE "runtime_guard|anomaly|INJECT|report" "$ROOT/h01a/serve.log" 2>/dev/null | tail -5 >> "$SUMMARY"
  grep -iE "runtime_guard|anomaly|INJECT|report" "$ROOT/h01b/serve.log" 2>/dev/null | tail -5 >> "$SUMMARY"
else
  log "[h01] HEALTH_TIMEOUT"; tail -15 "$ROOT/h01a/serve.log" >> "$SUMMARY"; tail -15 "$ROOT/h01b/serve.log" >> "$SUMMARY"
fi
stop_own "$PID_A"; stop_own "$PID_B"

# ============ L2: H-03 manual dump only armed instance ============
log "===== H-03 (A manual_dump=true, B guard_off) ====="
CFG_A2=$(mkcfg "$CFGDIR/p0_03_manual_dump.json" "$ROOT/h03a/dump" "")
CFG_B2=$(mkcfg "$CFGDIR/p0_01_guard_off.json" "$ROOT/h03b/dump" "")
PID_A=$(serve h03a 8043 0 "$CFG_A2")
PID_B=$(serve h03b 8044 1 "$CFG_B2")
log "[h03] pids A=$PID_A B=$PID_B"
OKA=$(wait_health 8043); OKB=$(wait_health 8044)
log "[h03] health A=$OKA B=$OKB"
if [ "$OKA" = 1 ] && [ "$OKB" = 1 ]; then
  req 8043 "arm me" "$ROOT/h03a/resp.json"
  req 8044 "not armed" "$ROOT/h03b/resp.json"
  sleep 8
  DA=$(ndump "$ROOT/h03a/dump"); DB=$(ndump "$ROOT/h03b/dump")
  log "[H-03] dumps A=$DA (expect>=1) B=$DB (expect 0)"
else
  log "[h03] HEALTH_TIMEOUT"; tail -15 "$ROOT/h03a/serve.log" >> "$SUMMARY"; tail -15 "$ROOT/h03b/serve.log" >> "$SUMMARY"
fi
stop_own "$PID_A"; stop_own "$PID_B"

# ============ L3: H-04 disk gate per-instance (auto-dump) + H-05 reload independence ============
# H-04 MUST use the AUTO-dump path (nan inject -> logits_finite -> on_trigger dump_kv).
# manual_dump arms with deferred (empty) block_ids => estimate=0 => disk gate skipped (actions.py:224).
log "===== H-04/H-05 (auto-dump: A huge headroom skips, B normal dumps; then reload A) ====="
HUGE=50000000000000
CFG_A3=$(mkcfg "$CFGDIR/k02_on_trigger_dump.json" "$ROOT/h04a/dump" "$HUGE")
CFG_B3=$(mkcfg "$CFGDIR/k02_on_trigger_dump.json" "$ROOT/h04b/dump" "")
PID_A=$(serve h04a 8045 0 "$CFG_A3" nan_logits)
PID_B=$(serve h04b 8046 1 "$CFG_B3" nan_logits)
log "[h04] pids A=$PID_A B=$PID_B"
OKA=$(wait_health 8045); OKB=$(wait_health 8046)
log "[h04] health A=$OKA B=$OKB"
if [ "$OKA" = 1 ] && [ "$OKB" = 1 ]; then
  req 8045 "disk tight" "$ROOT/h04a/resp.json"
  req 8046 "disk ok" "$ROOT/h04b/resp.json"
  sleep 8
  DA=$(ndump "$ROOT/h04a/dump"); DB=$(ndump "$ROOT/h04b/dump")
  skip_a=$(grep -cE "insufficient_free_space|skip: free=" "$ROOT/h04a/serve.log" 2>/dev/null || echo 0)
  log "[H-04] dumps A=$DA (expect 0, skipped) B=$DB (expect>=1) A skip-log=$skip_a (expect>0)"
  # H-05: hot reload A to normal headroom; both instances must stay healthy (no cross-interference).
  "$PY" - "$ROOT/h04a/runtime_config.json" "$ROOT/h04a/dump" <<'PY' > "$ROOT/h04a/runtime_config.json.new"
import json,sys
c=json.load(open(sys.argv[1]))
c.setdefault("dump",{})["dump_dir"]=sys.argv[2]
c["dump"]["free_headroom_bytes"]=5*1024*1024*1024
print(json.dumps(c,ensure_ascii=False))
PY
  mv "$ROOT/h04a/runtime_config.json.new" "$ROOT/h04a/runtime_config.json"
  sleep 6
  req 8045 "after reload" "$ROOT/h04a/resp2.json"
  req 8046 "b still ok" "$ROOT/h04b/resp2.json"
  sleep 4
  ha=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8045/health" 2>/dev/null || echo 000)
  hb=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8046/health" 2>/dev/null || echo 000)
  log "[H-05] after reload A: A health=$ha (expect 200) B health=$hb (expect 200) — reload independent, no crash"
else
  log "[h04] HEALTH_TIMEOUT"; tail -15 "$ROOT/h04a/serve.log" >> "$SUMMARY"; tail -15 "$ROOT/h04b/serve.log" >> "$SUMMARY"
fi
stop_own "$PID_A"; stop_own "$PID_B"

log "ALL H-01..H-06 DONE — see /tmp/rg_h_mixed/ for per-instance logs/dumps"
