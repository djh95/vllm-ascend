#!/usr/bin/env bash
# P0-1 guard_off vs T0 no-guard baseline (C4 simplified).
# Phase T0 : serve T0_PRODUCT (rg-t0-b1b58, no runtime_guard code) — plain boot.
# Phase GO : serve $PRODUCT (RUNNER=v2|v1 worktree) with ALL detectors off,
#            new schema (runtime_config_hot_reload=true), no legacy interval.
# Same 3 fixed prompts (great wall / Li Bai / AI), temp=0 seed=42.
# Assert: both phases HTTP 200; guard-off emits 0 incident reports; no
# Traceback; per-prompt deterministic payload byte-identical vs T0
# (raw-byte cmp recorded; id/created stripped for the verdict — vLLM mints a
# fresh request id/epoch ts per call, which is not model output).
set -uo pipefail
RUNNER=${RUNNER:-v2}
ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_p0_01_${RUNNER}}
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PORT_T0=${PORT_T0:-8301}
PORT_G=${PORT_G:-8302}
DET_OFF='{"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "output_substring": {"enabled": false}, "spec_acceptance": {"enabled": false}}'
P_NAMES=(great_wall li_bai ai_text)
P_TEXTS=("用一句话介绍长城" "写一篇150字的短文介绍李白" "用三句话说明什么是人工智能")
P_MT=(64 192 128)

log "=== P0-1 guard-off vs T0 start runner=$RUNNER t0=$T0_PRODUCT($(git -C "$T0_PRODUCT" log --oneline -1 2>/dev/null | head -1)) ==="

# ---------- Phase T0: no-guard baseline ----------
CARD=$(wait_idle_card) || { fail "no idle card for T0 phase"; p0_verdict P0_01; }
log "[t0] picked card=$CARD"
mkdir -p "$ROOT/t0"
p0_boot t0 "$CARD" "$PORT_T0" "$T0_PRODUCT" plain
if ! p0_serve_up t0 "$PORT_T0"; then
  p0_serve_down; hbm_release_check "$CARD" t0; p0_verdict P0_01
fi
sleep 5
for i in 0 1 2; do
  code=$(p0_ask "$PORT_T0" "${P_TEXTS[$i]}" "${P_MT[$i]}" "$ROOT/t0/${P_NAMES[$i]}.json")
  log "[t0] ${P_NAMES[$i]} http=$code bytes=$(wc -c < "$ROOT/t0/${P_NAMES[$i]}.json" | tr -d ' ')"
  [ "$code" = 200 ] || fail "T0 ${P_NAMES[$i]} http=$code"
done
tb0=$(grep -c "Traceback" "$ROOT/t0/serve.log" 2>/dev/null || true)
[ "${tb0:-0}" = 0 ] 2>/dev/null || fail "T0 serve log has Traceback"
log "[t0] traceback_count=${tb0:-0}"
p0_serve_down
hbm_release_check "$CARD" t0

# ---------- Phase guard-off on product build ----------
CARD=$(wait_idle_card) || { fail "no idle card for guard-off phase"; p0_verdict P0_01; }
log "[guard_off] picked card=$CARD"
p0_write_cfg "$ROOT/guard_off/runtime_config.json" false "$DET_OFF"
p0_boot guard_off "$CARD" "$PORT_G"
if ! p0_serve_up guard_off "$PORT_G"; then
  p0_serve_down; hbm_release_check "$CARD" guard_off; p0_verdict P0_01
fi
sleep 5   # ensure_persisted() startup-overwrite window (see run_tip15 v3 note)
p0_write_cfg "$ROOT/guard_off/runtime_config.json" false "$DET_OFF"   # re-assert via hot reload
sleep 6   # hot reload pickup (3s poll)
for i in 0 1 2; do
  code=$(p0_ask "$PORT_G" "${P_TEXTS[$i]}" "${P_MT[$i]}" "$ROOT/guard_off/${P_NAMES[$i]}.json")
  log "[guard_off] ${P_NAMES[$i]} http=$code bytes=$(wc -c < "$ROOT/guard_off/${P_NAMES[$i]}.json" | tr -d ' ')"
  [ "$code" = 200 ] || fail "guard-off ${P_NAMES[$i]} http=$code"
done
nrep=$(find "$ROOT/guard_off/report" -name 'report_*.json' 2>/dev/null | wc -l | tr -d ' ')
log "[guard_off] incident_reports=${nrep:-0} (expect 0, detectors off)"
[ "${nrep:-0}" = 0 ] 2>/dev/null || fail "guard-off produced ${nrep} incident reports (detectors should be off)"
tb1=$(grep -c "Traceback" "$ROOT/guard_off/serve.log" 2>/dev/null || true)
[ "${tb1:-0}" = 0 ] 2>/dev/null || fail "guard-off serve log has Traceback"
log "[guard_off] traceback_count=${tb1:-0}"

# ---------- diff vs T0 baseline (C4 simplified) ----------
for i in 0 1 2; do
  log "[cmp] ${P_NAMES[$i]}: T0 vs guard_off"
  if ! p0_body_cmp "$ROOT/t0/${P_NAMES[$i]}.json" "$ROOT/guard_off/${P_NAMES[$i]}.json"; then
    fail "prompt ${P_NAMES[$i]}: guard-off body diverged from T0 baseline"
  fi
done

p0_serve_down
hbm_release_check "$CARD" guard_off
log "=== P0-1 done ==="
p0_verdict P0_01
