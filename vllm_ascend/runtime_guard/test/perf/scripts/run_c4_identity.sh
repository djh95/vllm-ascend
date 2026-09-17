#!/usr/bin/env bash
# C4: temp=0 outputs bit-identical across T0-T3 (functional isolation / no divergence).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="$RG_PERF_ROOT/logs/$RUNNER/c4_identity.jsonl"
SUMMARY="$RG_PERF_ROOT/results/c4_${RUNNER}.txt"
: > "$SUMMARY"
CFG_T2=$(write_cfg_t2); CFG_T3=$(write_cfg_t3)
export PROMPTS='["请介绍一下长城的历史和主要关口。","请介绍一下李白的人生经历和代表作品。","请围绕人工智能的发展写一段话。"]'

capture(){
  local state="$1" tree="$2" cfg="$3" reload="$4" pid
  wait_idle "$CARD" || { log "ABORT: card busy"; exit 1; }
  pid=$(serve_and_wait "$tree" "c4_$state" "$cfg" "$reload")
  cd "$PERF_DIR"
  RG_PERF_URL="http://127.0.0.1:$PORT/v1/completions" RUNNER="$RUNNER" STATE="$state" \
    PROMPTS="$PROMPTS" PYTHONPATH="$PERF_DIR" "$PY" - <<'PY'
import json, os, urllib.request
url = os.environ["RG_PERF_URL"]; state = os.environ["STATE"]; runner = os.environ["RUNNER"]
prompts = json.loads(os.environ["PROMPTS"])
for p in prompts:
    body = json.dumps({"model": "dsv2", "prompt": p, "max_tokens": 64,
                       "temperature": 0, "seed": 42}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    text = out["choices"][0]["text"]
    print(json.dumps({"state": state, "runner": runner, "prompt": p, "text": text},
                     ensure_ascii=False))
PY
  stop_own "$pid"; wait_idle "$CARD" || true
}

capture T0 "$T0_ROOT" "" 0 >> "$OUT"
capture T1 "$PRODUCT_ROOT" "" 0 >> "$OUT"
capture T2 "$PRODUCT_ROOT" "$CFG_T2" 3 >> "$OUT"
capture T3 "$PRODUCT_ROOT" "$CFG_T3" 3 >> "$OUT"

"$PY" - "$OUT" <<'PY' | tee -a "$SUMMARY"
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
print(f"C4 {'PASS' if ok else 'FAIL'} (temp=0 bit-identical across T0-T3)")
PY
log "C4 done; summary=$SUMMARY"
