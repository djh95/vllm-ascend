#!/usr/bin/env bash
# G-06: bad JSON hot-reload soft-fail (no inject). Server must already run with reload>0.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds
CFG_LIVE="${RG_LIVE_CFG:?set RG_LIVE_CFG to the hot-reloaded runtime_config path on the server}"
BACKUP="${CFG_LIVE}.bak.$$"
cp "$CFG_LIVE" "$BACKUP"
echo '[g06] writing intentionally bad JSONC'
printf '{ this is not valid json\n' >"$CFG_LIVE" || true
sleep "${RG_RELOAD_WAIT:-5}"
echo '[g06] expect: service alive; old config kept; grep worker for reject/unknown'
curl -sS -o /dev/null -w "http=%{http_code}\n" "$URL" -H 'Content-Type: application/json' -d "{
  \"model\": \"${SERVED_MODEL_NAME:-default}\",
  \"prompt\": \"g06\",
  \"max_tokens\": 4,
  \"temperature\": 0
}" || echo "[g06] WARN: curl failed (service down?)"
mv "$BACKUP" "$CFG_LIVE"
echo "[g06] restored $CFG_LIVE"
