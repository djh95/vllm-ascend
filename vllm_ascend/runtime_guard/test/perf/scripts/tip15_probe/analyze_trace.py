#!/usr/bin/env python3
"""Analyze tip15 torch-profiler chrome traces: bus-thread AR vs rank pairing.

Answers, from per-rank traces (epoch-us timestamps, same host):
  1. Which op name the merged-bus AR shows up as (gloo vs hccl attribution).
  2. Per-rank AR duration distribution (p50/p90/max) and wave period.
  3. Cross-rank AR pairing: enter-time skew, exit-time skew, dur asymmetry
     -> does one rank enter ~one-wave early and block, while the other
        enters late and finishes immediately?
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TRACES = Path("/data0/test-mrv2-cann91/rg_prof_tip15/traces")
AR_RE = re.compile(r"all_?reduce|c10d|gloo|hccl", re.I)


def load_events(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(str(path), "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def analyze_one(path: Path) -> dict | None:
    ev = load_events(path)
    thread_names: dict[tuple[int, int], str] = {}
    proc_names: dict[int, str] = {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            thread_names[(e["pid"], e["tid"])] = e.get("args", {}).get("name", "")
        if e.get("ph") == "M" and e.get("name") == "process_name":
            proc_names[e["pid"]] = e.get("args", {}).get("name", "")
    # bus-thread AR events + per-tid op counts (to identify main thread)
    ar_events = []
    op_count: dict[tuple[int, int], int] = defaultdict(int)
    for e in ev:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        if AR_RE.search(name):
            ar_events.append(
                {
                    "ts": e["ts"],
                    "dur": e["dur"],
                    "tid": e["tid"],
                    "pid": e["pid"],
                    "name": name,
                    "cat": e.get("cat", ""),
                }
            )
            op_count[(e["pid"], e["tid"])] += 1
        else:
            op_count[(e["pid"], e["tid"])] += 1
    if not ar_events:
        return None
    ar_by_name = defaultdict(int)
    for a in ar_events:
        ar_by_name[(a["name"], a["cat"])] += 1
    # bus tid = tid with most AR events
    ar_tids = defaultdict(int)
    for a in ar_events:
        ar_tids[a["tid"]] += 1
    bus_tid = max(ar_tids, key=lambda t: ar_tids[t])
    bus_ar = sorted([a for a in ar_events if a["tid"] == bus_tid], key=lambda a: a["ts"])
    # main tid = tid with most non-AR ops in same pid
    pid = bus_ar[0]["pid"]
    main_tid = max(
        (t for t in op_count if t[0] == pid and t[1] != bus_tid),
        key=lambda t: op_count[t],
        default=None,
    )
    durs = [a["dur"] for a in bus_ar]
    periods = [b["ts"] - a["ts"] for a, b in zip(bus_ar, bus_ar[1:])]
    return {
        "path": path.name,
        "proc": proc_names.get(pid, f"pid{pid}"),
        "bus_tid": bus_tid,
        "bus_thread_name": thread_names.get((pid, bus_tid), "?"),
        "main_tid": main_tid,
        "main_thread_name": thread_names.get((pid, main_tid), "?") if main_tid else "?",
        "ar_names": dict(ar_by_name),
        "n_ar": len(bus_ar),
        "dur_us_p50": round(pct(durs, 0.5) / 1000, 1),
        "dur_us_p90": round(pct(durs, 0.9) / 1000, 1),
        "dur_us_max": round(max(durs) / 1000, 1),
        "period_us_p50": round(pct(periods, 0.5) / 1000, 1),
        "ar_list": [(a["ts"], a["dur"]) for a in bus_ar],
    }


def main():
    files = sorted(TRACES.glob("*.json.gz")) + sorted(TRACES.glob("*.json"))
    if not files:
        print("NO TRACE FILES in", TRACES)
        return
    ranks = []
    for p in files:
        r = analyze_one(p)
        if r:
            ranks.append(r)
            print(
                f"trace={r['path']} proc={r['proc']} bus_tid={r['bus_tid']} "
                f"({r['bus_thread_name']}) main_tid={r['main_tid']} ({r['main_thread_name']})"
            )
            print(f"  ar_names={r['ar_names']}")
            print(
                f"  n_ar={r['n_ar']} dur_us p50={r['dur_us_p50']} p90={r['dur_us_p90']} "
                f"max={r['dur_us_max']} period_us_p50={r['period_us_p50']}"
            )
        else:
            print(f"trace={p.name}: no AR-like events")
    if len(ranks) < 2:
        print("NEED 2 RANK TRACES for pairing; got", len(ranks))
        return
    # pair AR sequences by order
    a, b = ranks[0], ranks[1]
    n = min(len(a["ar_list"]), len(b["ar_list"]))
    # align on best offset: match by minimizing enter-time diff
    # (traces may cover slightly different windows; ARs are strictly ordered)
    ta = a["ar_list"]
    tb = b["ar_list"]
    # find offset that aligns: use first AR of the later-starting trace
    if ta[0][0] <= tb[0][0]:
        off = next((i for i, x in enumerate(ta) if x[0] >= tb[0][0]), 0)
        pairs = [(ta[off + k], tb[k]) for k in range(min(len(ta) - off, len(tb)))]
    else:
        off = next((i for i, x in enumerate(tb) if x[0] >= ta[0][0]), 0)
        pairs = [(ta[k], tb[off + k]) for k in range(min(len(ta), len(tb) - off))]
    enter_skew = [(y[0] - x[0]) / 1000 for x, y in pairs]
    exit_skew = [(y[0] + y[1] - x[0] - x[1]) / 1000 for x, y in pairs]
    dur_diff = [(y[1] - x[1]) / 1000 for x, y in pairs]
    print(f"\npaired ARs n={len(pairs)}")
    print(f"enter_skew ms (rank1-rank0): p50={pct(enter_skew,0.5):.2f} p90={pct(enter_skew,0.9):.2f} min={min(enter_skew):.2f} max={max(enter_skew):.2f}")
    print(f"exit_skew  ms: p50={pct(exit_skew,0.5):.3f} p90={pct(exit_skew,0.9):.3f} max={max(exit_skew):.3f}")
    print(f"dur_diff   ms (rank1-rank0): p50={pct(dur_diff,0.5):.2f} p90={pct(dur_diff,0.9):.2f} min={min(dur_diff):.2f} max={max(dur_diff):.2f}")
    print("\nsample pairs (rank0 ts,dur | rank1 ts,dur | enter_skew_ms):")
    for x, y in pairs[:12]:
        print(f"  {x[0]} {x[1]/1000:.1f}ms | {y[0]} {y[1]/1000:.1f}ms | {(y[0]-x[0])/1000:+.1f}ms")
    print("TRACE_ANALYSIS_DONE")


if __name__ == "__main__":
    sys.exit(main())
