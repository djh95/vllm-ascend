#!/usr/bin/env bash
# G-04: RG_INJECT=token_loop
export CASE_ID=g04 RG_INJECT="${RG_INJECT:-token_loop:5:8}" INCIDENT_TYPE=token_repeat \
  CFG_NAME=p0_02_token_repeat.json GOLDEN_NAME=g04_token_loop.json \
  PROMPT="loop probe" MAX_TOKENS=64
exec "$(cd "$(dirname "$0")" && pwd)/g_inject.sh"
