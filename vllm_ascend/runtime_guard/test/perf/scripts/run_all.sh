#!/usr/bin/env bash
# Master: run C1-C6 under BOTH runners (v2 then v1), results in logs/{v2,v1}/.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for RUNNER in v2 v1; do
  export RUNNER
  echo "##############################################"
  echo "########## RUNNER=$RUNNER ##########"
  echo "##############################################"
  for c in run_c1_c2_cross_rotate run_c3_ab run_c4_identity run_c5_dump run_c6_leakback; do
    echo "==== $RUNNER :: $c ===="
    if bash "$SCRIPT_DIR/$c.sh"; then
      echo "OK   $RUNNER :: $c"
    else
      echo "FAIL $RUNNER :: $c"
    fi
  done
done
echo "ALL DONE"
