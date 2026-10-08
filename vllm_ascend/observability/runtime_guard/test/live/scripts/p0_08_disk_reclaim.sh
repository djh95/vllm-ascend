#!/usr/bin/env bash
# P0-8 disk reclaim smoke.
# Standalone: df before -> boot -> health -> small manual dump -> df grew with
# dump size -> rm dump -> df falls back to baseline. Dump-producing paths in
# the other P0 scripts embed the same pattern (trap-EXIT removal + DF_USED_KB
# lines before/after; see _common.sh p0_exit / p0_df_kb).
set -uo pipefail
RUNNER=${RUNNER:-v2}
ROOT=${ROOT:-/data0/test-mrv2-cann91/rg_p0_08_${RUNNER}}
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PORT=${PORT:-8318}
DET_OFF='{"logits_finite": {"enabled": false}, "token_repeat": {"enabled": false}, "spec_acceptance": {"enabled": false}}'
DUMP_DIR="$ROOT/reclaim/dump"

df_used(){ df -k "$ROOT" 2>/dev/null | tail -1 | awk '{print $(NF-3)}'; }

log "=== P0-8 disk reclaim start runner=$RUNNER ==="
CARD=$(wait_idle_card) || { fail "no idle card"; p0_verdict P0_08; }
log "[reclaim] picked card=$CARD"
p0_write_cfg "$ROOT/reclaim/runtime_config.json" false "$DET_OFF"
p0_boot reclaim "$CARD" "$PORT"
if ! p0_serve_up reclaim "$PORT"; then
  p0_serve_down; hbm_release_check "$CARD" reclaim; p0_verdict P0_08
fi
sleep 5
p0_write_cfg "$ROOT/reclaim/runtime_config.json" false "$DET_OFF"
sleep 6

df0=$(p0_df_kb baseline_after_health)
p0_write_cfg "$ROOT/reclaim/runtime_config.json" true "$DET_OFF"   # small manual dump
sleep 6
code1=$(p0_ask "$PORT" "用一句话介绍长城" 64 "$ROOT/body1.json"); log "[reclaim] prompt1 http=$code1"
[ "$code1" = 200 ] || fail "prompt1 http=$code1"
code2=$(p0_ask "$PORT" "请连续输出60个哈字：$(printf '哈%.0s' {1..60})" 96 "$ROOT/body2.json"); log "[reclaim] prompt2 http=$code2"
[ "$code2" = 200 ] || fail "prompt2 http=$code2"

if p0_wait_pt "$DUMP_DIR" 120; then
  log "DUMP_FOUND $(find "$DUMP_DIR" -name '*.pt' | wc -l | tr -d ' ') .pt files under $DUMP_DIR"
else
  fail "no manual dump .pt under $DUMP_DIR after 120s"
fi
DUMP_KB=$(du -sk "$DUMP_DIR" 2>/dev/null | awk '{print $1}')
log "DUMP_SIZE_KB ${DUMP_KB:-0}"

# df grow (poll briefly: fs accounting can lag writeback)
GROW=-1
for i in $(seq 1 6); do
  u=$(df_used); GROW=$(( u - df0 ))
  [ "$GROW" -ge $(( DUMP_KB / 3 )) ] && break
  sleep 5
done
df1=$(( df0 + GROW ))
log "DF_DUMP_GROW_KB $GROW (dump=${DUMP_KB:-0}KB)"
[ "$GROW" -ge 0 ] || fail "df used decreased while dump written (grow=${GROW}KB)"
if [ "${DUMP_KB:-0}" -gt 0 ] && [ "$GROW" -lt $(( DUMP_KB / 3 )) ]; then
  log "WARN df grow ${GROW}KB below dump size ${DUMP_KB}KB (fs granularity/writeback) — reclaim assert below is authoritative"
fi

rm -rf "$DUMP_DIR"
sleep 3
df2=$(p0_df_kb after_cleanup)
RECLAIM=$(( df1 - df2 ))
DELTA_BASE=$(( df2 - df0 ))
SLACK=$(( DUMP_KB / 10 )); [ "$SLACK" -lt 102400 ] && SLACK=102400
log "DF_RECLAIM_KB $RECLAIM (df1=$df1 df2=$df2 df0=$df0)"
[ "$RECLAIM" -ge $(( DUMP_KB / 2 )) ] || fail "reclaim ${RECLAIM}KB < half of dump ${DUMP_KB}KB"
[ "$DELTA_BASE" -le "$SLACK" ] || fail "df did not fall back to baseline (delta=${DELTA_BASE}KB > slack ${SLACK}KB)"
log "DF_FALLBACK PASS: post-cleanup used within ${SLACK}KB of pre-dump baseline"

tb=$(grep -c "Traceback" "$ROOT/reclaim/serve.log" 2>/dev/null || true)
[ "${tb:-0}" = 0 ] 2>/dev/null || fail "serve log has Traceback"
log "traceback_count=${tb:-0}"

p0_serve_down
hbm_release_check "$CARD" reclaim
log "=== P0-8 done ==="
p0_verdict P0_08
