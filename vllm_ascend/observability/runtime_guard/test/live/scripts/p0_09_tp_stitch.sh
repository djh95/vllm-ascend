#!/usr/bin/env bash
# P0-9 / K-14 / K-24: TP stitch completeness.
# Verifies every tp_rank shard is present for each layer and that the KV head dim
# stitches back to full heads (no missing rank → silent partial dump).
# Run after a manual_dump / detector dump under TP>=2.
#   WAVE=path/wave_N   (default: newest wave_N under $DUMP_ROOT)
#   TP_SIZE=n          (expected TP size; default: auto-detect from rank tags)
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds

WAVE="${WAVE:-$(find "$DUMP_ROOT" -type d -name 'wave_*' 2>/dev/null | sort | tail -1)}"
if [[ -z "$WAVE" || ! -d "$WAVE" ]]; then
  echo "usage: WAVE=<wave_N> [TP_SIZE=<n>] $0  (no wave_N under $DUMP_ROOT)" >&2
  exit 2
fi

ARGS=(--dump-dir "$WAVE")
[[ -n "${TP_SIZE:-}" ]] && ARGS+=(--tp-size "$TP_SIZE")
echo "[p0_09] TP stitch completeness: WAVE=$WAVE TP_SIZE=${TP_SIZE:-auto}"
python3 -m vllm_ascend.runtime_guard.analysis.scripts.stitch_kv "${ARGS[@]}"
