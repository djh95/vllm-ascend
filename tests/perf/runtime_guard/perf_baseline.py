"""6-round baseline perf measurement on a baseline server (``run_rg.sh baseline``).

No runtime_config toggling — the baseline server has no runtime_config_path at all,
so guard stays unloaded. Pair with perf_ab_quick.py's A rounds to measure the overhead
of loading (but not activating) the guard infrastructure.
"""

from perf_lib import OUT_BASELINE, warmup, run_rounds


def main():
    warmup(rounds=1)
    print("[warmup baseline, unmeasured]", flush=True)
    rounds = [("baseline", None)] * 6
    run_rounds(rounds, OUT_BASELINE)
    print("done", flush=True)


if __name__ == "__main__":
    main()
