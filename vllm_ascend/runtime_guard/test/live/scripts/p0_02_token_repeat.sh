#!/usr/bin/env bash
# P0-2: token_repeat + report; HTTP body unchanged vs guard-off when possible.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds
CFG="${CONFIG_DIR}/p0_02_token_repeat.json"
echo "[p0_02] use config=$CFG ; enable RG_INJECT=token_loop:5:8 for reliable hit"
export RG_INJECT="${RG_INJECT:-token_loop:5:8}"
curl -sS "$URL" -H 'Content-Type: application/json' -d "{
  \"model\": \"${SERVED_MODEL_NAME:-default}\",
  \"prompt\": \"请连续输出相同汉字：哈哈\",
  \"max_tokens\": 64,
  \"temperature\": 0,
  \"seed\": 42
}" -o /tmp/rg_p0_02_body.json
python3 -m vllm_ascend.runtime_guard.analysis.scripts.summarize_reports \
  --report-dir "$REPORT_DIR" --incident-type token_repeat --limit 5 || true
python3 "$(cd "$(dirname "$0")" && pwd)/diff_report_golden.py" \
  --report-dir "$REPORT_DIR" --incident-type token_repeat \
  --golden "$GOLDEN_DIR/reports/g04_token_loop.json" || true
echo "[p0_02] body saved /tmp/rg_p0_02_body.json — compare to guard-off baseline manually"
