#!/usr/bin/env bash
# P0-6 async scheduling: check_after_sample still runs after get_output.
# v2: default async scheduling wraps AsyncOutput via AscendAsyncOutput
#     (hooks.py) — check_after_sample executes inside get_output() after the
#     inner D2H trim. v1: ModelRunner v1 wiring (9343c9859 restore) calls the
#     same guard hook on its own path.
# Evidence (either one satisfies the assert):
#   (a) serve log lines matching after_sample / check_after / AscendAsyncOutput
#   (b) a token_repeat incident report — that detector runs in stage
#       after_sample (detector/token_repeat.py), so a hit proves the
#       after-sample chain executed end to end.
set -uo pipefail
RUNNER=${RUNNER:-v2}
ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_p0_06_${RUNNER}}
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PORT=${PORT:-8316}
DET_ON='{"token_repeat": {"enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 8, "consecutive_hits": 1}, "logits_finite": {"enabled": false}, "output_substring": {"enabled": false}, "spec_acceptance": {"enabled": false}}'
REPEAT_PROMPT="请连续输出60个哈字：$(printf '哈%.0s' {1..60})"

log "=== P0-6 async after_sample start runner=$RUNNER ==="
[ "$RUNNER" = v2 ] && log "[wiring] v2 default async scheduling: AscendAsyncOutput.get_output -> _safe_check_after_sample (hooks.py)"
[ "$RUNNER" = v1 ] && log "[wiring] v1: ModelRunner v1 runtime_guard wiring -> check_after_sample (9343c9859)"

CARD=$(wait_idle_card) || { fail "no idle card"; p0_verdict P0_06; }
log "[async] picked card=$CARD"
p0_write_cfg "$ROOT/async/runtime_config.json" false "$DET_ON"
p0_boot async "$CARD" "$PORT"
if ! p0_serve_up async "$PORT"; then
  p0_serve_down; hbm_release_check "$CARD" async; p0_verdict P0_06
fi
sleep 5
p0_write_cfg "$ROOT/async/runtime_config.json" false "$DET_ON"
sleep 6

code=$(p0_ask "$PORT" "$REPEAT_PROMPT" 96 "$ROOT/body.json")
log "[async] repeat prompt http=$code bytes=$(wc -c < "$ROOT/body.json" | tr -d ' ')"
[ "$code" = 200 ] || fail "repeat prompt http=$code"

REP=$(p0_find_report "$ROOT/async/report" token_repeat)
if [ -z "$REP" ]; then
  p0_ask "$PORT" "$REPEAT_PROMPT" 96 "$ROOT/body.json" >/dev/null; sleep 6
  REP=$(p0_find_report "$ROOT/async/report" token_repeat)
fi
if [ -n "$REP" ]; then
  log "REPORT_PATH $REP"
  p0_report_check "$REP" token_repeat || fail "incident_type mismatch (expect token_repeat, after_sample stage)"
  log "EVIDENCE after-sample chain live: token_repeat detector (stage=after_sample) fired -> check_after_sample ran after get_output"
else
  fail "no token_repeat incident — after-sample chain produced no hit"
fi

# (a) supplementary: grep hook traces in serve log
HOOKS=$(grep -inE "after_sample|check_after|AscendAsyncOutput|after-sample" "$ROOT/async/serve.log" 2>/dev/null | head -5)
if [ -n "$HOOKS" ]; then
  while IFS= read -r l; do log "HOOK_EVIDENCE $l"; done <<< "$HOOKS"
else
  log "HOOK_EVIDENCE none at INFO level (normal; detector-hit evidence above is authoritative)"
fi
RGN=$(grep -c "\[runtime_guard" "$ROOT/async/serve.log" 2>/dev/null || true)
log "RUNTIME_GUARD_LOG_LINES=${RGN:-0}"
grep "\[runtime_guard" "$ROOT/async/serve.log" 2>/dev/null | tail -6 | while IFS= read -r l; do log "RG_LOG $l"; done

tb=$(grep -c "Traceback" "$ROOT/async/serve.log" 2>/dev/null || true)
[ "${tb:-0}" = 0 ] 2>/dev/null || fail "serve log has Traceback"
log "traceback_count=${tb:-0}"

p0_serve_down
hbm_release_check "$CARD" async
log "=== P0-6 done ==="
p0_verdict P0_06
