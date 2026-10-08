#!/usr/bin/env bash
# G-05: RG_INJECT=spec_all_reject (needs MTP / spec decode on)
export CASE_ID=g05 RG_INJECT="${RG_INJECT:-spec_all_reject}" INCIDENT_TYPE=spec_acceptance \
  CFG_NAME=g05_spec_all_reject.json GOLDEN_NAME=g05_spec_all_reject.json \
  PROMPT="spec reject probe" MAX_TOKENS=32
exec "$(cd "$(dirname "$0")" && pwd)/g_inject.sh"
