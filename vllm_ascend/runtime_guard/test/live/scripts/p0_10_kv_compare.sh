#!/usr/bin/env bash
# P0-10 / K-22 / K-24: KV content correctness vs a reference baseline.
# Stitches both dumps and compares per-layer full-tensor cosine (>= COS_THRESH)
# and shape. Proves the dumped KV is the one the model actually used, not
# garbage / cross-request contamination.
#   TARGET=path/wave_N       (default: newest wave_N under $DUMP_ROOT)
#   REF=path/wave_N_ref      (required, or auto-match under RG_REF_ROOT)
#   COS_THRESH=0.9999
#   RG_JSON_OUT=out.json     (default: /tmp/rg_p0_10_compare.json)
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds

COS_THRESH="${COS_THRESH:-0.9999}"
TARGET="${TARGET:-$(find "$DUMP_ROOT" -type d -name 'wave_*' 2>/dev/null | sort | tail -1)}"
REF="${REF:-}"
if [[ -z "$REF" && -n "${RG_REF_ROOT:-}" ]]; then
  REF="$(find "$RG_REF_ROOT" -type d -name 'wave_*' 2>/dev/null | sort | tail -1)"
fi

if [[ -z "$TARGET" || ! -d "$TARGET" ]]; then
  echo "usage: TARGET=<wave_N> REF=<ref_wave> [COS_THRESH=0.9999] $0" >&2
  exit 2
fi
if [[ -z "$REF" || ! -d "$REF" ]]; then
  echo "REF wave dir required — set REF=... or RG_REF_ROOT=... (capture via capture_kv_ref.sh)" >&2
  exit 2
fi

OUT="${RG_JSON_OUT:-/tmp/rg_p0_10_compare.json}"
echo "[p0_10] KV compare: TARGET=$TARGET REF=$REF COS_THRESH=$COS_THRESH"
python3 -m vllm_ascend.runtime_guard.analysis.scripts.stitch_kv \
  --target "$TARGET" --ref "$REF" --cos-thresh "$COS_THRESH" --json-out "$OUT"
