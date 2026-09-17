#!/usr/bin/env bash
# P0-1: reload=0 / detectors off — output matches ungated baseline intent (temp=0).
# Requires: already-running server OR set START_CMD to start one.
# shellcheck source=_common.sh
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds

CFG="${CONFIG_DIR}/p0_01_guard_off.json"
echo "[p0_01] config=$CFG (detectors off; attach via product --additional-config if starting server)"
echo "[p0_01] TODO: start product build with this config, then:"
curl -sS "$URL" -H 'Content-Type: application/json' -d "{
  \"model\": \"${SERVED_MODEL_NAME:-default}\",
  \"prompt\": \"ping\",
  \"max_tokens\": 8,
  \"temperature\": 0,
  \"seed\": 42
}" | python3 -m json.tool | head -40
echo "[p0_01] expect: HTTP 200; no new anomaly reports under $REPORT_DIR"
