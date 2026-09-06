"""6-round perf measurement on a T1 baseline server (no ``--additional-config``).

Measures terminal-state T1 (guard code loaded, reload_interval=0, all detectors off —
the production "set-and-forget" shape). The runtime_guard infrastructure IS loaded
(``RuntimeGuardProcessor.bind`` runs unconditionally on vllm-ascend platform),
but every per-step path early-returns so cost should be ~0.

NOT T0 ("no guard code``). To measure T0, run from a ``git worktree``
at merge-base ``37e382498`` — see README.md C1.

Pair with perf_ab_quick.py A rounds (T2) to compare T1 vs T2 = hot-reload cost (C2).
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
