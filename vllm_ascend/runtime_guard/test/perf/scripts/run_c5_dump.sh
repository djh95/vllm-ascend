#!/usr/bin/env bash
# C5: dump_kv on_trigger cost vs T3 (manual lab compare).
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
CFG="${RG_PERF_ROOT}/config/t3_with_dump_kv.json"
cat >"$CFG" <<'EOF'
{
  "reload_interval_seconds": 3,
  "dump": { "auto_max_times": 2, "manual_dump": false, "auto_cooldown_seconds": 60 },
  "actions": { "defaults": { "on_trigger": ["report", "dump_kv"] } },
  "detector": {
    "token_repeat": { "enabled": true, "min_tokens": 8, "window": 32, "repeat_sum_threshold": 64 },
    "logits_finite": { "enabled": false },
    "output_substring": { "enabled": false },
    "spec_acceptance": { "enabled": false }
  }
}
EOF
echo "[perf] C5 runner=$RUNNER cfg=$CFG"
echo "1) baseline T3 (no dump_kv) TPS"
echo "2) switch to this cfg; inject/trigger dumps sparsely; measure TPS + df"
echo "3) KEEP_DUMP=0 delete kv_cache after round"
echo "Record ratio dump/T3 separately for v2 and v1"
