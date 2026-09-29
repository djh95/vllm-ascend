#!/usr/bin/env python3
"""Analyze [RG-BUS-WAVE] per-wave timestamps, split by TP rank, with cross-rank pairing.

Answers the lockstep / misalignment question directly:
  - per-rank ar_call (collective time) and drain wait distributions
  - cross-rank pairing on |head diff| < threshold (same wave)
  - head / submit / ar_in / ar_out skews: is the main thread lockstep?
  - the misalignment fingerprint: slow rank's ar_out == fast rank's next-wave head

Usage: analyze_wave_pairs.py <serve_log> [pair_window_ms]
"""
import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/data0/test-mrv2-cann91/rg_probe_tip15/serve_t2.log")
WINDOW_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0

PAT = re.compile(
    r"(Worker_TP(\d)).*RG-BUS-WAVE\] head=([\d.]+) submit=\+([-\d.]+)ms "
    r"ar_in=\+([-\d.]+)ms ar_out=\+([-\d.]+)ms ar_call_us=([\d.]+) "
    r"ar_item_us=([\d.]+) gate=(\w+) wait_us=([\d.]+)"
)

by_rank = {}
for line in LOG.read_text(errors="ignore").splitlines():
    m = PAT.search(line)
    if m:
        by_rank.setdefault(f"TP{m.group(2)}", []).append({
            "head": float(m.group(3)),
            "submit": float(m.group(3)) + float(m.group(4)) / 1000,
            "ar_in": float(m.group(3)) + float(m.group(5)) / 1000,
            "ar_out": float(m.group(3)) + float(m.group(6)) / 1000,
            "call_us": float(m.group(7)),
            "item_us": float(m.group(8)),
            "gate": m.group(9),
            "wait_us": float(m.group(10)),
        })

if not by_rank:
    print("NO WAVE LINES FOUND")
    sys.exit(1)

for r, ws in by_rank.items():
    ws.sort(key=lambda w: w["head"])
    calls = sorted(w["call_us"] for w in ws)
    waits = sorted(w["wait_us"] for w in ws)
    n = len(ws)
    gates = {w["gate"] for w in ws}
    print(
        f"{r}: n={n} gate={gates} "
        f"ar_call p50={calls[n // 2] / 1000:.2f}ms p90={calls[9 * n // 10] / 1000:.2f}ms max={calls[-1] / 1000:.2f}ms | "
        f"drain_wait p50={waits[n // 2] / 1000:.3f}ms p90={waits[9 * n // 10] / 1000:.3f}ms max={waits[-1] / 1000:.3f}ms"
    )
    heads = [w["head"] for w in ws]
    periods = [round((b - a) * 1000) for a, b in zip(heads, heads[1:])]
    if periods:
        p = sorted(periods)
        print(f"  wave period ms: p50={p[len(p) // 2]} max={p[-1]}")

ranks = sorted(by_rank)
if len(ranks) == 2:
    a, b = by_rank[ranks[0]], by_rank[ranks[1]]
    pairs = []
    for x in a:
        cand = [y for y in b if abs(y["head"] - x["head"]) < WINDOW_MS / 1000]
        if cand:
            pairs.append((x, min(cand, key=lambda y: abs(y["head"] - x["head"]))))
    print(f"\ncross-rank same-wave pairs (window={WINDOW_MS}ms): {len(pairs)} of {len(a)}")
    if pairs:
        for f in ("head", "submit", "ar_in", "ar_out"):
            skews = sorted((y[f] - x[f]) * 1000 for x, y in pairs)
            n = len(skews)
            print(
                f"  {ranks[1]}-{ranks[0]} {f}_skew ms: p10={skews[n // 10]:.2f} "
                f"p50={skews[n // 2]:.2f} p90={skews[9 * n // 10]:.2f}"
            )
        slow0 = sum(1 for x, y in pairs if x["call_us"] > 20000)
        slow1 = sum(1 for x, y in pairs if y["call_us"] > 20000)
        print(f"  slow-AR (>20ms) waves: {ranks[0]}={slow0} {ranks[1]}={slow1}")
        print("  sample (paired waves):")
        for x, y in pairs[:8]:
            print(
                f"    head={x['head']:.3f} {ranks[0]}_call={x['call_us'] / 1000:.1f}ms "
                f"{ranks[1]}_call={y['call_us'] / 1000:.1f}ms | "
                f"ar_out {ranks[0]}=+{(x['ar_out'] - x['head']) * 1000:.1f}ms "
                f"{ranks[1]}=+{(y['ar_out'] - y['head']) * 1000:.1f}ms | "
                f"wait {ranks[0]}={x['wait_us'] / 1000:.3f}ms {ranks[1]}={y['wait_us'] / 1000:.3f}ms"
            )
        print("\nWAVE_PAIR_ANALYSIS_DONE")
