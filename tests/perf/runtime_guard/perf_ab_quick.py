"""C3 perf comparison: T3 (B, detectors on) vs T2 (A, detectors off), ×3 cross-rotated.

IMPORTANT: A/B here mean detector off/on WITHIN reload=3 — NOT the same as
``dfx-perf-bench`` SKILL's A/B which mean no-DFX / DFX+reload>0+off.
See tests/perf/runtime_guard/README.md for the unified T-label terminology.

Server must be in T2 base shape (``run_rg.sh guard`` with reload_interval=3);
script hot-toggles detectors via JSON edits + ``trigger()`` to flip T2 ↔ T3.
Restores runtime_config.json in finally.
"""

import json
import time

from perf_lib import CFG, OUT_AB, set_detectors, trigger, warmup, run_rounds


def main():
    original = json.load(open(CFG))
    try:
        set_detectors(True)
        trigger()
        time.sleep(1)
        warmup(rounds=1)
        print("[warmup B-state, unmeasured]", flush=True)
        rounds = [("B", True), ("A", False)] * 3
        run_rounds(rounds, OUT_AB)
    finally:
        json.dump(original, open(CFG, "w"), indent=2)
        trigger()
    print("config restored", flush=True)


if __name__ == "__main__":
    main()
