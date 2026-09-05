"""B/A × 3 cross-rotated perf measurement on a guard server.

Restores runtime_config.json in finally. Run on a guard-mode server (``run_rg.sh guard``).
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
