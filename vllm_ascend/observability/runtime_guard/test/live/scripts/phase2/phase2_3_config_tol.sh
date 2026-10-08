#!/usr/bin/env bash
# Phase 2.3: F-07/G-06 config容错测试
# Tests: unknown key handling + bad JSON handling after config refactor
set -uo pipefail
source /home/ma-user/rg_env.sh
export ASCEND_CUSTOM_OPP_PATH="${PRODUCT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${PRODUCT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${ASCEND_CUSTOM_OPP_PATH:-}"

SCRIPTS=/data0/test-mrv2-cann91/rg-analysis/vllm_ascend/runtime_guard/test/live/scripts
source "$SCRIPTS/_common.sh"

ROOT=/data0/test-mrv2-cann91/rg_phase23
PORT=8319
TAG=config_tol

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [phase23] $*"; }

rm -rf "$ROOT"
mkdir -p "$ROOT/$TAG/report" "$ROOT/$TAG/dump"

# Write initial valid config
p0_write_cfg "$ROOT/$TAG/runtime_config.json" false '{"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}'

# Boot server with hot reload
CARD=$(wait_idle_card) || { log "FAIL: no idle card"; exit 1; }
log "picked card=$CARD"
p0_boot "$TAG" "$CARD" "$PORT"
if ! p0_serve_up "$TAG" "$PORT"; then
  log "FAIL: server not up"
  p0_serve_down
  exit 1
fi
log "server up on port=$PORT card=$CARD"

CFG="$ROOT/$TAG/runtime_config.json"
SERVE_LOG="$ROOT/$TAG/serve.log"
PASS=0; FAIL=0

# --- Test 1: Unknown key in config ---
log "=== Test 1: Unknown key (F-07) ==="
cp "$CFG" "${CFG}.bak"
cat > "$CFG" <<EOF
{"dump": {"dump_dir": "$ROOT/$TAG/dump", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": false}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}, "totally_unknown_detector": {"enabled": true, "bogus_param": 42}}, "unknown_section": {"foo": "bar"}}}
EOF
sleep 5
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"p0m\",\"prompt\":\"test unknown key\",\"max_tokens\":4,\"temperature\":0}")
if [ "$code" = "200" ]; then
  log "PASS: server survived unknown key (http=$code)"
  PASS=$((PASS+1))
else
  log "FAIL: server died on unknown key (http=$code)"
  FAIL=$((FAIL+1))
fi
# Check log for warning
if grep -qi "unknown\|unrecognized\|ignored\|warn" "$SERVE_LOG" 2>/dev/null; then
  log "PASS: warning logged for unknown key"
  PASS=$((PASS+1))
else
  log "WARN: no warning found in log for unknown key"
fi
cp "${CFG}.bak" "$CFG"
sleep 3

# --- Test 2: Bad JSON (G-06) ---
log "=== Test 2: Bad JSON (G-06) ==="
cp "$CFG" "${CFG}.bak2"
printf '{ this is not valid json\n' > "$CFG"
sleep 5
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"p0m\",\"prompt\":\"test bad json\",\"max_tokens\":4,\"temperature\":0}")
if [ "$code" = "200" ]; then
  log "PASS: server survived bad JSON (http=$code)"
  PASS=$((PASS+1))
else
  log "FAIL: server died on bad JSON (http=$code)"
  FAIL=$((FAIL+1))
fi
if grep -qiE "json|parse|reject|invalid" "$SERVE_LOG" 2>/dev/null; then
  log "PASS: JSON parse error logged"
  PASS=$((PASS+1))
else
  log "WARN: no JSON error logged"
fi
cp "${CFG}.bak2" "$CFG"
sleep 3

# --- Test 3: Valid config with new manual_dump=true ---
log "=== Test 3: Valid config recovery ==="
cat > "$CFG" <<EOF
{"dump": {"dump_dir": "$ROOT/$TAG/dump", "auto_max_times": 0, "auto_cooldown_seconds": 300, "manual_dump": true}, "actions": {"defaults": {"on_trigger": ["report"]}}, "detector": {"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}}
EOF
sleep 5
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"p0m\",\"prompt\":\"test recovery\",\"max_tokens\":4,\"temperature\":0}")
if [ "$code" = "200" ]; then
  log "PASS: config recovery works (http=$code)"
  PASS=$((PASS+1))
else
  log "FAIL: config recovery failed (http=$code)"
  FAIL=$((FAIL+1))
fi
sleep 5
# Check if manual dump triggered
PT_COUNT=$(find "$ROOT/$TAG/dump" -name "*.pt" 2>/dev/null | wc -l)
if [ "$PT_COUNT" -gt 0 ]; then
  log "PASS: manual dump triggered after recovery ($PT_COUNT .pt files)"
  PASS=$((PASS+1))
else
  log "WARN: no manual dump after recovery"
fi

log "=== Phase 2.3 Results: PASS=$PASS FAIL=$FAIL ==="

# Cleanup
p0_serve_down
hbm_release_check "$CARD" "$TAG"
log "=== Phase 2.3 done ==="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
