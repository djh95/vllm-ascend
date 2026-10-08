#!/usr/bin/env bash
# P0-5 inject NaN -> logits_finite incident.
# Plan: RG_INJECT=nan_logits env at serve boot. Before booting we verify the
# product build actually honours that switch: grep the product worktree + git
# HEAD for RG_INJECT / any nan_logits fault-injection implementation.
# If absent -> SKIP with evidence (per FTL P0-5 instructions).
# Known state at authoring time: RG_INJECT lives only in the legacy
# feat/runtime-guard-analysis layout (vllm_ascend/runtime_guard/inject.py,
# source-level master switch shipped False); product builds under test
# (v2 623bf2166 / v1 9343c9859) have no inject implementation.
set -uo pipefail
RUNNER=${RUNNER:-v2}
ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_p0_05_${RUNNER}}
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

log "=== P0-5 inject_nan start runner=$RUNNER product=$PRODUCT ==="
log "PRODUCT_SHA $(git -C "$PRODUCT" log --oneline -1 2>/dev/null | head -1)"

G_HEAD=$( { cd "$PRODUCT" && git grep -n "RG_INJECT" HEAD -- '*.py'; } 2>/dev/null | head -5 || true)
G_WT=$(grep -rn --include='*.py' "RG_INJECT" "$PRODUCT" 2>/dev/null | grep -v '/\.git/' | head -5 || true)
G_EV=$( { cd "$PRODUCT" && git grep -n "nan_logits\|fault_inject\|INJECT_ENV" HEAD -- 'vllm_ascend/**/*.py'; } 2>/dev/null | grep -viE "raise_if" | head -5 || true)
log "INJECT_GREP_GIT_HEAD out=[${G_HEAD:-<none>}]"
log "INJECT_GREP_WORKTREE out=[${G_WT:-<none>}]"
log "INJECT_GREP_FAULT_IMPL out=[${G_EV:-<none>}] (informational)"

if [ -z "$G_HEAD" ] && [ -z "$G_WT" ]; then
  log "SKIP reason: 产品仓 $PRODUCT ($(git -C "$PRODUCT" rev-parse --short HEAD 2>/dev/null)) 不认 RG_INJECT=nan_logits —"
  log "  git grep RG_INJECT HEAD 与 worktree grep 均无命中，亦无其它 fault-inject 实现；"
  log "  RG_INJECT 仅存在于 feat/runtime-guard-analysis 旧布局 vllm_ascend/runtime_guard/inject.py"
  log "  （source-level master switch，默认 shipped False，需源码改动），无法在本构建注入 NaN。"
  log "  => logits_finite 无法被真实触发，ill_type=nan 断言不可执行。"
  p0_skip P0_05 "no RG_INJECT/fault-inject switch in product build $(git -C "$PRODUCT" rev-parse --short HEAD 2>/dev/null)"
fi

# Switch exists in this build: boot with env and exercise the detector.
PORT=${PORT:-8315}
DET_ON='{"logits_finite": {"enabled": true}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}'
export RG_INJECT="${RG_INJECT:-nan_logits}"
log "[live] RG_INJECT=$RG_INJECT recognized by build — booting to exercise logits_finite"
CARD=$(wait_idle_card) || { fail "no idle card"; p0_verdict P0_05; }
p0_write_cfg "$ROOT/inject/runtime_config.json" false "$DET_ON"
p0_boot inject "$CARD" "$PORT"
if ! p0_serve_up inject "$PORT"; then
  p0_serve_down; hbm_release_check "$CARD" inject; p0_verdict P0_05
fi
sleep 5; p0_write_cfg "$ROOT/inject/runtime_config.json" false "$DET_ON"; sleep 6
code=$(p0_ask "$PORT" "inject nan probe" 32 "$ROOT/body.json")
log "[inject] prompt http=$code"; [ "$code" = 200 ] || fail "inject probe http=$code"
REP=$(p0_find_report "$ROOT/inject/report" logits_finite)
[ -n "$REP" ] || fail "no logits_finite report"
if [ -n "$REP" ]; then
  log "REPORT_PATH $REP"
  p0_report_check "$REP" logits_finite || fail "incident_type mismatch (expect logits_finite)"
  ill=$("$PY" -c "import json;r=json.load(open('$REP'));d=r.get('detail',{});print(r.get('ill_type',d.get('ill_type','NA')))" 2>/dev/null)
  log "ILL_TYPE ${ill:-NA} (expect nan semantics)"
  echo "${ill:-NA}" | grep -qi "nan" || fail "ill_type not nan: ${ill:-NA}"
fi
code2=$(p0_ask "$PORT" "server still alive" 16 "$ROOT/body2.json")
log "[inject] follow-up prompt http=$code2 (service must stay 200)"
[ "$code2" = 200 ] || fail "service crashed after NaN incident (follow-up http=$code2)"
p0_serve_down; hbm_release_check "$CARD" inject
log "=== P0-5 done ==="
p0_verdict P0_05
