#!/usr/bin/env bash
# P0-2 token_repeat report does not alter HTTP output (I7).
# One boot: detectors OFF -> baseline body; hot-reload detectors ON
# (token_repeat min_tokens=8, on_trigger=["report"]) -> same prompt again.
# Assert: incident report lands under report/token_repeat/report_*.json AND
# the deterministic payload (id/created stripped) is byte-identical to the
# guard-off body. Baseline is self-contained in-boot; the T0 equivalence of
# the guard-off state itself is covered by P0-1 (its counterpart is reusable).
# Fallback if hot toggle misses: reboot with detectors ON from the start and
# compare against the boot-1 guard-off body.
set -uo pipefail
RUNNER=${RUNNER:-v2}
ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_p0_02_${RUNNER}}
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PORT=${PORT:-8312}
DET_OFF='{"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}'
DET_ON='{"token_repeat": {"enabled": true, "window": 32, "repeat_sum_threshold": 64, "min_tokens": 8, "consecutive_hits": 1}, "logits_finite": {"enabled": false}, "spec_acceptance": {"enabled": false}}'
REPEAT_PROMPT="请连续输出60个哈字：$(printf '哈%.0s' {1..60})"
CFG_BASE="$ROOT/guard_on/runtime_config.json"

log "=== P0-2 token_repeat (I7) start runner=$RUNNER ==="
CARD=$(wait_idle_card) || { fail "no idle card"; p0_verdict P0_02; }
log "[guard_on] picked card=$CARD"

p0_write_cfg "$CFG_BASE" false "$DET_OFF"
p0_boot guard_on "$CARD" "$PORT"
if ! p0_serve_up guard_on "$PORT"; then
  p0_serve_down; hbm_release_check "$CARD" guard_on; p0_verdict P0_02
fi
sleep 5
p0_write_cfg "$CFG_BASE" false "$DET_OFF"   # steady guard-off state (re-assert)
sleep 6

code=$(p0_ask "$PORT" "$REPEAT_PROMPT" 96 "$ROOT/body_guard_off.json")
log "[phase off] repeat prompt http=$code bytes=$(wc -c < "$ROOT/body_guard_off.json" | tr -d ' ')"
[ "$code" = 200 ] || fail "guard-off repeat prompt http=$code"
sleep 3
n0=$(find "$ROOT/guard_on/report" -name 'report_*.json' 2>/dev/null | wc -l | tr -d ' ')
log "[phase off] reports_before_detector_on=${n0:-0} (expect 0)"

p0_write_cfg "$CFG_BASE" false "$DET_ON"    # hot toggle: token_repeat on
sleep 8                                    # 3s poll + margin
code=$(p0_ask "$PORT" "$REPEAT_PROMPT" 96 "$ROOT/body_guard_on.json")
log "[phase on] repeat prompt http=$code bytes=$(wc -c < "$ROOT/body_guard_on.json" | tr -d ' ')"
[ "$code" = 200 ] || fail "guard-on repeat prompt http=$code"

REP=$(p0_find_report "$ROOT/guard_on/report" token_repeat)
if [ -z "$REP" ]; then
  log "[phase on] no report yet — retry same prompt once"
  p0_ask "$PORT" "$REPEAT_PROMPT" 96 "$ROOT/body_guard_on.json" >/dev/null
  sleep 6
  REP=$(p0_find_report "$ROOT/guard_on/report" token_repeat)
fi

if [ -z "$REP" ]; then
  # fallback: hot toggle missed — reboot with detectors ON from the start
  log "[fallback] hot-reload toggle produced no hit; rebooting with detectors ON pre-boot"
  p0_serve_down
  hbm_release_check "$CARD" guard_on
  CARD=$(wait_idle_card) || { fail "no idle card (fallback)"; p0_verdict P0_02; }
  p0_write_cfg "$ROOT/guard_on2/runtime_config.json" false "$DET_ON"
  p0_boot guard_on2 "$CARD" "$PORT"
  if ! p0_serve_up guard_on2 "$PORT"; then
    p0_serve_down; hbm_release_check "$CARD" guard_on2; p0_verdict P0_02
  fi
  sleep 5
  p0_write_cfg "$ROOT/guard_on2/runtime_config.json" false "$DET_ON"
  sleep 6
  code=$(p0_ask "$PORT" "$REPEAT_PROMPT" 96 "$ROOT/body_guard_on.json")
  log "[fallback] repeat prompt http=$code"
  [ "$code" = 200 ] || fail "fallback repeat prompt http=$code"
  REP=$(p0_find_report "$ROOT/guard_on2/report" token_repeat)
fi

if [ -n "$REP" ]; then
  log "REPORT_PATH $REP"
  p0_report_check "$REP" token_repeat || fail "report incident_type mismatch (expect token_repeat)"
else
  fail "no token_repeat incident report under $ROOT/guard_on*/report/token_repeat/"
fi

log "[I7] body compare guard-off vs guard-on (same prompt, temp=0 seed=42)"
if ! p0_body_cmp "$ROOT/body_guard_off.json" "$ROOT/body_guard_on.json"; then
  fail "I7 violated: HTTP body changed when token_repeat detector+report active"
fi

tb=$(grep -c "Traceback" "$ROOT"/guard_on*/serve.log 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
[ "${tb:-0}" = 0 ] 2>/dev/null || fail "serve log has Traceback"
log "traceback_count=${tb:-0}"

p0_serve_down
hbm_release_check "$CARD" guard_on
log "=== P0-2 done ==="
p0_verdict P0_02
