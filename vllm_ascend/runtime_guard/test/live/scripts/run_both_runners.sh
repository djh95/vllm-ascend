#!/usr/bin/env bash
# Dual-runner wrapper: runs a live script under v2 then v1.
# Usage: ./run_both_runners.sh p0_08_disk_reclaim.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:?pass script name under live/scripts/}"
shift || true
for r in v2 v1; do
  echo "======== RUNNER=$r $TARGET ========"
  RUNNER="$r" bash "${SCRIPT_DIR}/${TARGET}" "$@"
done
