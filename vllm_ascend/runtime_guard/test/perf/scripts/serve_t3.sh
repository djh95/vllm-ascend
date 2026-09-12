#!/usr/bin/env bash
# T3: reload=3, four detectors on.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
MODEL="${MODEL:?set MODEL}"
CFG="${RG_PERF_ROOT}/config/t3_reload_detectors_on.json"
cat >"$CFG" <<'EOF'
{
  "reload_interval_seconds": 3,
  "dump": { "auto_max_times": 0, "manual_dump": false },
  "actions": { "defaults": { "on_trigger": ["report"] } },
  "detector": {
    "logits_finite": { "enabled": true },
    "token_repeat": { "enabled": true },
    "output_substring": { "enabled": false },
    "spec_acceptance": { "enabled": false }
  }
}
EOF
export RG_PERF_CFG="$CFG"
echo "[perf] T3 cfg=$CFG runner=$RUNNER (enable output_substring/spec as needed)"
if [[ -n "${START_CMD:-}" ]]; then eval "$START_CMD"; fi
