#!/usr/bin/env bash
# T2: product HEAD + runtime_config_path + reload=3, detectors off.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
CFG=$(write_cfg_t2)
log "[T2] cfg=$CFG reload=3 detectors off"
serve_and_wait "$PRODUCT_ROOT" "t2" "$CFG" 3
