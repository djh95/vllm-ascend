#!/usr/bin/env bash
# T2: reload=3, detectors off. Writes a config under RG_PERF_ROOT/config.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
MODEL="${MODEL:?set MODEL}"
CFG="${RG_PERF_ROOT}/config/t2_reload_detectors_off.json"
cat >"$CFG" <<'EOF'
{
  "reload_interval_seconds": 3,
  "dump": { "auto_max_times": 0, "manual_dump": false },
  "detector": {
    "logits_finite": { "enabled": false },
    "token_repeat": { "enabled": false },
    "output_substring": { "enabled": false },
    "spec_acceptance": { "enabled": false }
  }
}
EOF
export RG_PERF_CFG="$CFG"
echo "[perf] T2 cfg=$CFG runner=$RUNNER"
echo "Attach via product --additional-config / runtime_config_path=$CFG (lab-specific flags)."
echo "example START_CMD with additional-config pointing at $CFG"
if [[ -n "${START_CMD:-}" ]]; then eval "$START_CMD"; fi
