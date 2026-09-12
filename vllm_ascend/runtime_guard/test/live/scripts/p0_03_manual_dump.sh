#!/usr/bin/env bash
# P0-3 / K-01: manual_dump — must produce wave_N/<rank_tag>/*.pt ; then verify schema.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds
CFG="${CONFIG_DIR}/p0_03_manual_dump.json"
echo "[p0_03] config=$CFG — trigger manual_dump / manual_trigger per product docs"
curl -sS "$URL" -H 'Content-Type: application/json' -d "{
  \"model\": \"${SERVED_MODEL_NAME:-default}\",
  \"prompt\": \"hello dump\",
  \"max_tokens\": 16,
  \"temperature\": 0,
  \"seed\": 1
}" >/tmp/rg_p0_03_body.json || true

# Pick newest report if any and verify KV layout.
REPORT="$(find "$REPORT_DIR" -name 'report_*.json' 2>/dev/null | head -1 || true)"
if [[ -n "$REPORT" ]]; then
  python3 -m vllm_ascend.runtime_guard.analysis.scripts.verify_request_kv \
    --report "$REPORT" --report-dir "$REPORT_DIR" || true
  python3 "$(cd "$(dirname "$0")" && pwd)/diff_report_golden.py" \
    --report "$REPORT" --golden "$GOLDEN_DIR/reports/k_manual_dump_schema.json" \
    --schema-only || true
else
  echo "[p0_03] WARN: no report_*.json yet — ensure manual_dump armed on product server"
fi
# cleanup via trap unless KEEP_DUMP=1
