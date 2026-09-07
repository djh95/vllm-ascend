"""6-round baseline perf + memory measurement on a baseline server.

Measures terminal-state T1 (guard code loaded, reload=0, all detectors off
— the production "set-and-forget" shape). The runtime_guard infrastructure IS
loaded (RuntimeGuardProcessor.bind runs unconditionally on vllm-ascend
platform), but every per-step path early-returns so cost should be ~0.

NOT T0. To measure T0 (no guard code at all), run from a git worktree at
merge-base 37e382498 — see tests/perf/runtime_guard/README.md.

After the 6 measured rounds, runs a 5-min idle leak-back phase to detect
per-req state container leaks. Pass: end-of-leakback rss_delta_kb <= 30MB.
"""

from perf_lib import (
    OUT_BASELINE, OUT_LEAKBACK_BASELINE, warmup, run_rounds, run_leakback,
)


def main():
    warmup(rounds=1)
    print("[warmup baseline, unmeasured]", flush=True)
    rounds = [("baseline", None)] * 6
    run_rounds(rounds, OUT_BASELINE, with_mem=True)
    print("[leakback baseline start]", flush=True)
    run_leakback("baseline", OUT_LEAKBACK_BASELINE)
    print("done", flush=True)


if __name__ == "__main__":
    main()
