#!/usr/bin/env bash
# G-02: RG_INJECT=inf_logits
export CASE_ID=g02 RG_INJECT="${RG_INJECT:-inf_logits}" INCIDENT_TYPE=logits_finite \
  CFG_NAME=g02_inf_logits.json GOLDEN_NAME=g02_inf_logits.json \
  PROMPT="inject inf probe" MAX_TOKENS=8
exec "$(cd "$(dirname "$0")" && pwd)/g_inject.sh"
