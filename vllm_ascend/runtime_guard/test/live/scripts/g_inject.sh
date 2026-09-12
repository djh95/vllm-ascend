#!/usr/bin/env bash
# Generic inject case: set CASE_ID, RG_INJECT, INCIDENT_TYPE, CFG_NAME, GOLDEN_NAME, PROMPT.
# Example: CASE_ID=g02 RG_INJECT=inf_logits INCIDENT_TYPE=logits_finite ...
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds

CASE_ID="${CASE_ID:?}"
INCIDENT_TYPE="${INCIDENT_TYPE:?}"
CFG_NAME="${CFG_NAME:?}"
GOLDEN_NAME="${GOLDEN_NAME:?}"
PROMPT="${PROMPT:-inject probe}"
MAX_TOKENS="${MAX_TOKENS:-16}"
export RG_INJECT="${RG_INJECT:?}"
export PROMPT

CFG="${CONFIG_DIR}/${CFG_NAME}"
GOLDEN="${GOLDEN_DIR}/reports/${GOLDEN_NAME}"
echo "[$CASE_ID] RG_INJECT=$RG_INJECT config=$CFG golden=$GOLDEN"
echo "[$CASE_ID] skills: detector-sweep + analysis; update SKILL.md if drift"

curl -sS "$URL" -H 'Content-Type: application/json' -d "{
  \"model\": \"${SERVED_MODEL_NAME:-default}\",
  \"prompt\": $(python3 -c 'import json,os; print(json.dumps(os.environ["PROMPT"]))'),
  \"max_tokens\": ${MAX_TOKENS},
  \"temperature\": 0,
  \"seed\": 7
}" -o "/tmp/rg_${CASE_ID}_body.json" || true

python3 -m vllm_ascend.runtime_guard.analysis.scripts.summarize_reports \
  --report-dir "$REPORT_DIR" --incident-type "$INCIDENT_TYPE" --limit 5 || true

DIFF_RC=0
python3 "$(cd "$(dirname "$0")" && pwd)/diff_report_golden.py" \
  --report-dir "$REPORT_DIR" --incident-type "$INCIDENT_TYPE" \
  --golden "$GOLDEN" || DIFF_RC=$?

echo "[$CASE_ID] localize:"
echo "  1) expect incident_type=$INCIDENT_TYPE"
echo "  2) dump if on_trigger includes dump_kv → verify_request_kv"
echo "  3) one-liner suspicion from inject scenario"
echo "  4) refresh golden: python3 .../refresh_golden.py --report <path> --out $GOLDEN"
exit "$DIFF_RC"
