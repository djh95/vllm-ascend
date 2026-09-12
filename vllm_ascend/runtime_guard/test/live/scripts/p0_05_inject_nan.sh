#!/usr/bin/env bash
# P0-5 / G-01: RG_INJECT=nan_logits → logits_finite hit → report vs golden → brief localize note.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds
CFG="${CONFIG_DIR}/p0_05_inject_nan.json"
export RG_INJECT="${RG_INJECT:-nan_logits}"
echo "[p0_05/G-01] RG_INJECT=$RG_INJECT config=$CFG"
echo "[p0_05] skill: runtime-guard-detector-sweep + runtime-guard-analysis (update skill if paths drift)"

curl -sS "$URL" -H 'Content-Type: application/json' -d "{
  \"model\": \"${SERVED_MODEL_NAME:-default}\",
  \"prompt\": \"inject nan probe\",
  \"max_tokens\": 8,
  \"temperature\": 0,
  \"seed\": 7
}" >/tmp/rg_p0_05_body.json || true

python3 -m vllm_ascend.runtime_guard.analysis.scripts.summarize_reports \
  --report-dir "$REPORT_DIR" --incident-type logits_finite --limit 5 || true
python3 "$(cd "$(dirname "$0")" && pwd)/diff_report_golden.py" \
  --report-dir "$REPORT_DIR" --incident-type logits_finite \
  --golden "$GOLDEN_DIR/reports/g01_nan_logits.json"

echo "[p0_05] localize checklist:"
echo "  1) incident_type=logits_finite"
echo "  2) dump optional; if on_trigger has dump_kv → verify_request_kv"
echo "  3) one-liner: logits non-finite pre-sample (inject), not KV crosstalk"
echo "  4) if skill wrong → edit analysis/skill/*/SKILL.md"
