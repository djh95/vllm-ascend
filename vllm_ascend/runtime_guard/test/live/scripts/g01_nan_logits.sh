#!/usr/bin/env bash
# G-01 alias → same as p0_05
export CASE_ID=g01 RG_INJECT="${RG_INJECT:-nan_logits}" INCIDENT_TYPE=logits_finite \
  CFG_NAME=p0_05_inject_nan.json GOLDEN_NAME=g01_nan_logits.json \
  PROMPT="inject nan probe" MAX_TOKENS=8
exec "$(cd "$(dirname "$0")" && pwd)/g_inject.sh"
