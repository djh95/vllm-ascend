#!/usr/bin/env bash
# P0-8: disk reclaim smoke — create then delete a dummy dump tree.
set -euo pipefail
DUMP_ROOT="${DUMP_ROOT:-./runtime/report/kv_cache}"
KEEP_DUMP="${KEEP_DUMP:-0}"
mkdir -p "$DUMP_ROOT/_p0_08_probe"
echo probe >"$DUMP_ROOT/_p0_08_probe/x.txt"
df -h "$(dirname "$DUMP_ROOT")" || df -h .
if [[ "$KEEP_DUMP" != "1" ]]; then
  rm -rf "$DUMP_ROOT/_p0_08_probe"
  echo "[p0_08] PASS — probe dir removed"
else
  echo "[p0_08] KEEP_DUMP=1 left probe in place"
fi
