#!/usr/bin/env bash
# tip12 chain (1992d6b71): wait for tip10 baseline chain, then smoke (v2) -> matrix (v2).
# tip12 = branch rewrite: feature(078331827) + BugFix(3a479639a zombie/quota/print fix)
# + observability nesting + shim drop. Config interface unchanged (runtime_config_path etc).
set -uo pipefail
LOG=/data0/test-mrv2-cann91/rg_tip12_chain.log
S=/data0/test-mrv2-cann91/rg_c56/scripts
echo "$(date '+%F %T') tip12 chain waiting for tip10 chain" >> "$LOG"
while pgrep -f "chain_tip10.sh" >/dev/null 2>&1; do sleep 60; done
echo "$(date '+%F %T') tip10 finished; tip12 chain start" >> "$LOG"
bash "$S/run_tip12_smoke.sh" >> "$LOG" 2>&1
bash "$S/run_tip12_matrix.sh" >> "$LOG" 2>&1
echo "$(date '+%F %T') === TIP12_CHAIN_DONE ===" >> "$LOG"
