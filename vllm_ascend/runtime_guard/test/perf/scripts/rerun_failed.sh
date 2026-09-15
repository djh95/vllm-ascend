#!/usr/bin/env bash
# Re-run the parts that failed/contaminated during the first pass:
#   v2: C4 (c4_T2 died), C6 (leakback PID swap)  -> re-run
#   v1: C1-C6 (servers killed by external signals) -> full re-run
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_one(){
  local runner="$1" c="$2"
  export RUNNER="$runner"
  echo "==== $runner :: $c ===="
  if bash "$SCRIPT_DIR/$c.sh"; then
    echo "OK   $runner :: $c"
  else
    echo "FAIL $runner :: $c"
  fi
}

# v2 re-runs (short)
run_one v2 run_c4_identity
run_one v2 run_c6_leakback

# v1 full pass
run_one v1 run_c1_c2_cross_rotate
run_one v1 run_c3_ab
run_one v1 run_c4_identity
run_one v1 run_c5_dump
run_one v1 run_c6_leakback

echo "ALL RERUN DONE"
