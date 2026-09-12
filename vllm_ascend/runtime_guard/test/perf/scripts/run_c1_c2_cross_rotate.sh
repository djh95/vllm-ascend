#!/usr/bin/env bash
# C1+C2 cross-rotation driver (documentation + hooks). Full automation needs lab START_CMD.
# Rule: T0→T1→T2→… N rounds; then repeat for the other RUNNER.
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
ROUNDS="${ROUNDS:-6}"
OUT_DIR="${RG_PERF_ROOT}/logs/${RUNNER}"
mkdir -p "$OUT_DIR"
echo "[perf] C1/C2 cross-rotate runner=$RUNNER rounds=$ROUNDS → $OUT_DIR"
echo "For each round r=1..$ROUNDS:"
echo "  1) serve_t0.sh → perf_baseline → save ${OUT_DIR}/t0_r\${r}.jsonl"
echo "  2) serve_t1.sh → perf_baseline → save ${OUT_DIR}/t1_r\${r}.jsonl"
echo "  3) serve_t2.sh → perf_baseline → save ${OUT_DIR}/t2_r\${r}.jsonl"
echo "  4) kill server (npu-smi worker PIDs); never serial all-T0-then-all-T1"
echo "Derive C1=geom_mean(T1)/geom_mean(T0), C2=geom_mean(T2)/geom_mean(T1)"
echo "Then: RUNNER=v1 $0   # other runner"
if [[ "${EXECUTE:-0}" != "1" ]]; then
  echo "[perf] dry-run only (set EXECUTE=1 after wiring START_CMD / kill helpers)"
  exit 0
fi
echo "[perf] EXECUTE=1 not fully automated yet — run the printed steps on lab"
exit 2
