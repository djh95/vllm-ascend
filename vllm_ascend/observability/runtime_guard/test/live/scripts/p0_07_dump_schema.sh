#!/usr/bin/env bash
# P0-7 manual dump schema gate (K-11/12/13).
# Boot detectors OFF, health, then hot-write manual_dump=true (post-startup
# toggle — startup persist would wipe a pre-written true). Send prompts so
# live rows exist, manual dump fires end-of-wave.
# Verify with analysis verify_request_kv.py + inline K checks:
#   K-11 dump .pt schema keys + report schema (req_id/incident_type/block_ids)
#   K-12 block coverage: len(block_ids)*block_size >= N tokens
#   K-13 report<->dump consistency: req_id / block_ids / dump_dir
# FTL 0.4: trap EXIT removes the temp dump (KEEP_DUMP=1 overrides); df before/
# after recorded for the P0-8 reclaim embedding.
set -uo pipefail
RUNNER=${RUNNER:-v2}
ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_p0_07_${RUNNER}}
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PORT=${PORT:-8317}
BS=${BLOCK_SIZE:-128}
DET_OFF='{"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}'
DUMP_DIR="$ROOT/manual/dump"
REP_DIR="$ROOT/manual/report"

log "=== P0-7 dump schema (K-11/12/13) start runner=$RUNNER ==="
CARD=$(wait_idle_card) || { fail "no idle card"; p0_verdict P0_07; }
log "[manual] picked card=$CARD"
p0_write_cfg "$ROOT/manual/runtime_config.json" false "$DET_OFF"
p0_boot manual "$CARD" "$PORT"
if ! p0_serve_up manual "$PORT"; then
  p0_serve_down; hbm_release_check "$CARD" manual; p0_verdict P0_07
fi
sleep 5   # ensure_persisted() startup-overwrite window
df0=$(p0_df_kb pre_dump)
p0_write_cfg "$ROOT/manual/runtime_config.json" true "$DET_OFF"   # manual_dump=true via hot reload
sleep 6

code1=$(p0_ask "$PORT" "用一句话介绍长城" 64 "$ROOT/body1.json"); log "[manual] prompt1 http=$code1"
[ "$code1" = 200 ] || fail "prompt1 http=$code1"
code2=$(p0_ask "$PORT" "请连续输出60个哈字：$(printf '哈%.0s' {1..60})" 96 "$ROOT/body2.json"); log "[manual] prompt2 http=$code2"
[ "$code2" = 200 ] || fail "prompt2 http=$code2"

if p0_wait_pt "$DUMP_DIR" 120; then
  log "DUMP_FOUND under $DUMP_DIR ($(find "$DUMP_DIR" -name '*.pt' | wc -l | tr -d ' ') .pt files)"
else
  fail "no manual dump .pt under $DUMP_DIR after 120s"
fi
df1=$(p0_df_kb post_dump)
DUMP_KB=$(du -sk "$DUMP_DIR" 2>/dev/null | awk '{print $1}')
log "DUMP_SIZE_KB ${DUMP_KB:-0} df_delta_kb=$(( ${df1:-0} - ${df0:-0} ))"

REP=$(p0_find_report "$REP_DIR" manual_trigger)
[ -z "$REP" ] && REP=$(p0_find_report "$REP_DIR")
[ -n "$REP" ] || fail "no report_*.json under $REP_DIR (manual trigger report expected)"
if [ -n "$REP" ]; then
  log "REPORT_PATH $REP"
  p0_report_check "$REP" manual_trigger || log "NOTE report incident_type=$(python3 -c "import json;print(json.load(open('$REP')).get('incident_type'))" 2>/dev/null) (manual_trigger expected)"
fi

# ---- verify_request_kv.py (analysis tool) ----
if [ -n "$REP" ]; then
  "$PY" "$VERIFY_TOOL" --report "$REP" --report-dir "$REP_DIR" --block-size "$BS" \
    > "$ROOT/verify_out.txt" 2>&1
  VRC=$?
  sed 's/^/[verify_request_kv] /' "$ROOT/verify_out.txt" | tee -a "$LOG"
  [ "$VRC" = 0 ] || fail "verify_request_kv rc=$VRC"
fi

# ---- inline K-11 / K-12 / K-13 ----
"$PY" - "$REP" "$DUMP_DIR" "$BS" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
rep = json.load(open(sys.argv[1]))
dump_dir = Path(sys.argv[2]); bs = int(sys.argv[3])
det = rep.get("detail") if isinstance(rep.get("detail"), dict) else {}
dd = rep.get("dump_dir") or det.get("dump_dir")
# K-13 must pair the report with ITS OWN dump dir (manual dump may cover
# several live requests -> multiple <req_id> roots under the dump tree).
if dd and Path(str(dd)).is_dir():
    pts = sorted(Path(str(dd)).rglob("*.pt"))
else:
    pts = sorted(dump_dir.rglob("*.pt"))
req_roots = sorted({p.parent.name for p in dump_dir.rglob("wave_*")}) or sorted({p.parent.parent.parent.name for p in dump_dir.rglob("*.pt")})
print(f"K CONTEXT report_req={rep.get('req_id')} report_dump_dir={dd} req_roots_under_dump={len(req_roots)}")
ok = True
def chk(name, cond, extra=""):
    global ok
    print(f"K {name}: {'PASS' if cond else 'FAIL'} {extra}")
    ok = ok and bool(cond)
rows = det.get("requests") or []
r0 = rows[0] if rows and isinstance(rows[0], dict) else {}
chk("K11_REPORT_SCHEMA", ("req_id" in rep) and ("incident_type" in rep),
    f"req_id={rep.get('req_id')} incident_type={rep.get('incident_type')}")
chk("K11_REPORT_DETAIL", ("block_ids" in det) or ("block_ids" in r0) or ("block_ids" in rep),
    f"detail_keys={sorted(list(det))[:10]}")
import torch
pt = torch.load(pts[0], map_location="cpu", weights_only=False) if pts else {}
need = {"req_id", "block_ids", "layer", "rank_tag", "num_kv_heads", "tensor"}
chk("K11_PT_SCHEMA", need <= set(pt.keys()), f"pt={pts[0] if pts else 'NONE'} keys={sorted(list(pt))[:12]}")
bids = list(det.get("block_ids") or r0.get("block_ids") or rep.get("block_ids") or pt.get("block_ids") or [])
pn = len(det.get("prompt_token_ids") or r0.get("prompt_token_ids") or [])
on = len(det.get("output_token_ids") or r0.get("output_token_ids") or [])
N = pn + on
if N == 0:
    N = int(det.get("prompt_token_count") or r0.get("prompt_token_count") or 0) + \
        int(det.get("output_token_count") or r0.get("output_token_count") or 0)
chk("K12_BLOCK_COVER", (N == 0) or (len(bids) * bs >= N),
    f"len(block_ids)={len(bids)} block_size={bs} cap={len(bids)*bs} N={N}")
chk("K13_REQ_ID", str(pt.get("req_id")) == str(rep.get("req_id")),
    f"pt.req_id={pt.get('req_id')} report.req_id={rep.get('req_id')}")
chk("K13_BLOCK_IDS", (not bids) or (list(pt.get("block_ids") or []) == list(bids)),
    f"pt={list(pt.get('block_ids') or [])[:8]} report={bids[:8]}")
chk("K13_DUMP_DIR", bool(dd) and Path(str(dd)).is_dir(), f"dump_dir={dd}")
if dd and pts:
    chk("K13_PT_UNDER_DUMP_DIR", str(pts[0]).startswith(str(Path(str(dd)).resolve())),
        f"pt={pts[0]}")
print("K_OVERALL", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYEOF
KRC=${PIPESTATUS[0]}
[ "$KRC" = 0 ] || fail "inline K-11/12/13 checks failed (rc=$KRC)"

tb=$(grep -c "Traceback" "$ROOT/manual/serve.log" 2>/dev/null || true)
[ "${tb:-0}" = 0 ] 2>/dev/null || fail "serve log has Traceback"
log "traceback_count=${tb:-0}"

p0_serve_down
hbm_release_check "$CARD" manual
log "=== P0-7 done (dump cleanup handed to trap EXIT; KEEP_DUMP=${KEEP_DUMP:-0}) ==="
p0_verdict P0_07
