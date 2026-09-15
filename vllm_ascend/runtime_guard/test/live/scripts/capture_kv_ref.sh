#!/usr/bin/env bash
# K-20: capture a KV reference baseline (golden 入库).
# 1) Writes a git-safe meta JSON (rank layout + per-layer shape, NO tensor bytes)
#    into $GOLDEN_DIR/dump_schema/kv_ref_meta.json.
# 2) Copies the FULL wave dir into $RG_REF_ROOT (machine-local, NOT git) so
#    p0_10_kv_compare.sh can later compare content cosine against it.
#   WAVE=path/wave_N       (default: newest wave_N under $DUMP_ROOT)
#   RG_REF_ROOT=/tmp/rg_kv_ref   (machine-local ref store; never commit this)
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
require_cmds

WAVE="${WAVE:-$(find "$DUMP_ROOT" -type d -name 'wave_*' 2>/dev/null | sort | tail -1)}"
if [[ -z "$WAVE" || ! -d "$WAVE" ]]; then
  echo "usage: WAVE=<wave_N> $0  (no wave_N under $DUMP_ROOT)" >&2
  exit 2
fi

REF_ROOT="${RG_REF_ROOT:-/tmp/rg_kv_ref}"
STAMP="$(basename "$(dirname "$WAVE")")_$(basename "$WAVE")"   # <req_id>_wave_N

# 1) git-safe meta — dump_schema/kv_ref_meta.json (no tensor data)
python3 - "$WAVE" "$GOLDEN_DIR/dump_schema/kv_ref_meta.json" <<'PY'
import json, sys
from pathlib import Path
from vllm_ascend.runtime_guard.analysis.scripts._lib import discover_rank_dirs, load_kv_dir

wave = Path(sys.argv[1]); out = Path(sys.argv[2])
ranks = discover_rank_dirs(wave)
meta = {"wave": str(wave), "rank_dirs": sorted(ranks), "layers": []}
for tag in sorted(ranks):
    for layer, d in load_kv_dir(ranks[tag]).items():
        meta["layers"].append({
            "rank_tag": tag, "layer": layer, "req_id": d.req_id,
            "block_ids": d.block_ids, "shape": list(d.tensor.shape),
            "num_kv_heads": d.num_kv_heads, "tp_rank": d.tp_rank,
            "pp_rank": d.pp_rank, "cp_rank": d.cp_rank, "dp_rank": d.dp_rank,
        })
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[capture_kv_ref] meta -> {out} ({len(meta['layers'])} layer shards)")
PY

# 2) machine-local full ref (NOT git)
mkdir -p "$REF_ROOT"
DEST="$REF_ROOT/$STAMP"
rm -rf "$DEST"
cp -a "$WAVE" "$DEST"
echo "[capture_kv_ref] full ref -> $DEST"
echo "[capture_kv_ref] next: RG_REF_ROOT=$REF_ROOT p0_10_kv_compare.sh (TARGET=<new wave>)"
