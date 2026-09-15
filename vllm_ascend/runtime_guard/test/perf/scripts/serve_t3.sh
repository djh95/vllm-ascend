#!/usr/bin/env bash
# T3: product HEAD + reload=3, four detectors on (logits_finite, token_repeat,
# output_substring, spec_acceptance).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
CFG=$(write_cfg_t3)
log "[T3] cfg=$CFG reload=3 detectors on (4)"
serve_and_wait "$PRODUCT_ROOT" "t3" "$CFG" 3
