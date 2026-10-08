#!/usr/bin/env bash
# G-03: RG_INJECT=forbidden_substring (DEFAULT_TEXT=李白; config patterns match)
export CASE_ID=g03 \
  RG_INJECT="${RG_INJECT:-forbidden_substring:5}" \
  INCIDENT_TYPE=output_substring \
  CFG_NAME=g03_forbidden_substring.json GOLDEN_NAME=g03_forbidden_substring.json \
  PROMPT="please continue" MAX_TOKENS=32
exec "$(cd "$(dirname "$0")" && pwd)/g_inject.sh"
