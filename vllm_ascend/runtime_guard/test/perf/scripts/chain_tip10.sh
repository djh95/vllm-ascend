#!/usr/bin/env bash
# tip10 chain: smoke (v2) -> multi-model matrix (v2). No global perf rerun:
# hot path semantically unchanged vs tip9/tip7 (refactors + additive knobs);
# T0 still cannot boot in this container. Cards 0,3,4,5,6,7 free -> m6 tp4 runs.
set -uo pipefail
S=/data0/test-mrv2-cann91/rg_c56/scripts

echo "$(date '+%F %T') tip10 chain start"
bash "$S/run_tip10_smoke.sh"
echo "$(date '+%F %T') smoke exit=$?; matrix next"
bash "$S/run_tip10_matrix.sh"
echo "$(date '+%F %T') === TIP10_CHAIN_DONE ==="
