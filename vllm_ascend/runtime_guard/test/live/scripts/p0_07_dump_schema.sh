#!/usr/bin/env bash
# P0-7: dump schema gate — verify_request_kv + inspect one .pt
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds
REPORT="${1:-}"
if [[ -z "$REPORT" ]]; then
  REPORT="$(find "$REPORT_DIR" -name 'report_*.json' 2>/dev/null | head -1 || true)"
fi
[[ -n "$REPORT" ]] || { echo "usage: $0 <report.json>  (or have reports under $REPORT_DIR)"; exit 2; }
python3 -m vllm_ascend.runtime_guard.analysis.scripts.verify_request_kv \
  --report "$REPORT" --report-dir "$REPORT_DIR"
PT="$(find "$DUMP_ROOT" -name '*.pt' 2>/dev/null | head -1 || true)"
if [[ -n "$PT" ]]; then
  python3 -m vllm_ascend.runtime_guard.analysis.scripts.inspect_kv_dump --path "$PT"
fi
