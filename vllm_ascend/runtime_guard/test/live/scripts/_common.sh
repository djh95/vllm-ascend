#!/usr/bin/env bash
# Shared env for live NPU harness (analysis branch only).
# shellcheck disable=SC2034
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# Prefer product checkout if RG_PRODUCT_ROOT set (config branch / install).
PRODUCT_ROOT="${RG_PRODUCT_ROOT:-$ROOT}"
export PYTHONPATH="${PRODUCT_ROOT}:${PYTHONPATH:-}"

RUNNER="${RUNNER:-v2}"
case "$RUNNER" in
  v2) export VLLM_USE_V2_MODEL_RUNNER=1 ;;
  v1) export VLLM_USE_V2_MODEL_RUNNER=0 ;;
  *) echo "RUNNER must be v1 or v2, got: $RUNNER" >&2; exit 2 ;;
esac

MODEL="${MODEL:-}"
PORT="${PORT:-8017}"
HOST="${HOST:-127.0.0.1}"
URL="${RG_LIVE_URL:-http://${HOST}:${PORT}/v1/completions}"
REPORT_DIR="${RG_REPORT_DIR:-./runtime/report}"
DUMP_ROOT="${DUMP_ROOT:-${REPORT_DIR}/kv_cache}"
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../configs" && pwd)"
GOLDEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../golden" && pwd)"
KEEP_DUMP="${KEEP_DUMP:-0}"

df_check() {
  local target="${1:-$DUMP_ROOT}"
  mkdir -p "$(dirname "$target")" 2>/dev/null || true
  df -h "$(dirname "$target")" || df -h .
}

cleanup_dumps() {
  if [[ "$KEEP_DUMP" == "1" ]]; then
    echo "[live] KEEP_DUMP=1 — leaving $DUMP_ROOT"
    return 0
  fi
  if [[ -d "$DUMP_ROOT" ]]; then
    echo "[live] removing temporary dumps under $DUMP_ROOT"
    rm -rf "$DUMP_ROOT"
  fi
}

trap cleanup_dumps EXIT

require_cmds() {
  command -v curl >/dev/null
  command -v python3 >/dev/null
}

echo "[live] RUNNER=$RUNNER MODEL=$MODEL URL=$URL REPORT_DIR=$REPORT_DIR"
df_check
