#!/usr/bin/env python3
"""Analyze [RG-BUS-WAVE] per-wave timestamps from the probe session.

Both ranks log into one file; head= is epoch seconds (same host) so ranks can
be aligned. Key questions answered:
  1. gate backend actually used (cpu=gloo / npu=hccl)
  2. ar_call (collective wait) vs ar_item (D2H) — where does the 99ms sit
  3. wave-phase structure of the two ranks: same-wave pairs (small head gap)
     vs a full-wave phase shift (head gap ~ wave period)
  4. if pairable: head/submit/ar_enter/ar_exit skews per wave
"""
import re
import sys
from collections import Counter
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/data0/test-mrv2-cann91/rg_probe_tip15/serve_t2.log")
PAT = re.compile(
    r"\[RG-BUS-WAVE\] head=([\d.]+) submit=\+([-\d.]+)ms ar_in=\+([-\d.]+)ms "
    r"ar_out=\+([-\d.]+)ms ar_call_us=([\d.]+) ar_item_us=([\d.]+) "
    r"gate=(\w+) wait_us=([\d.]+) drain_end=([\d.]+)"
)

waves = []
for line in LOG.read_text(errors="ignore").splitlines():
    m = PAT.search(line)
    if m:
        w = {
            "head": float(m.group(1)),
            "submit_off": float(m.group(2)),
            "ar_in_off": float(m.group(3)),
            "ar_out_off": float(m.group(4)),
            "ar_call_us": float(m.group(5)),
            "ar_item_us": float(m.group(6)),
            "gate": m.group(7),
            "wait_us": float(m.group(8)),
            "drain_end": float(m.group(9)),
        }
        w["submit"] = w["head"] + w["submit_off"] / 1000
        w["ar_in"] = w["head"] + w["ar_in_off"] / 1000
        w["ar_out"] = w["head"] + w["ar_out_off"] / 1000
        waves.append(w)

if not waves:
    print("NO WAVE LINES FOUND")
    sys.exit(1)

print(f"n_wave_lines={len(waves)}")
gates = Counter(w["gate"] for w in waves)
print(f"gate_backends={dict(gates)}")

calls = sorted(w["ar_call_us"] for w in waves)
items = sorted(w["ar_item_us"] for w in waves)


def stats(name, vals, scale=1000.0):
    if not vals:
        return
    n = len(vals)
    print(
        f"{name}: n={n} p10={vals[n // 10] / scale:.2f}ms p50={vals[n // 2] / scale:.2f}ms "
        f"p90={vals[9 * n // 10] / scale:.2f}ms max={vals[-1] / scale:.2f}ms"
    )


stats("ar_call(collective wait)", calls)
stats("ar_item(sync after AR)", items)
stats("wait_us(end-of-wave drain block)", sorted(w["wait_us"] for w in waves))

# wave-phase structure: sort by head, look at gaps
waves.sort(key=lambda w: w["head"])
gaps = [b["head"] - a["head"] for a, b in zip(waves, waves[1:])]
small = [g for g in gaps if g < 0.05]
big = [g for g in gaps if g >= 0.05]
print(
    f"\nhead-gap structure: n_gaps={len(gaps)} small(<50ms)={len(small)} big(>=50ms)={len(big)}"
)
if small:
    small.sort()
    print(f"  small gaps ms: p50={small[len(small)//2]*1000:.1f} max={small[-1]*1000:.1f}")
if big:
    big.sort()
    print(f"  big gaps ms: p10={big[len(big)//10]*1000:.1f} p50={big[len(big)//2]*1000:.1f} p90={big[9*len(big)//10]*1000:.1f}")

# classify waves into two rank groups by ar_call magnitude if bimodal
lo = [w for w in waves if w["ar_call_us"] < 20000]
hi = [w for w in waves if w["ar_call_us"] >= 20000]
print(f"\nrank split by ar_call: lo(<20ms)={len(lo)} hi(>=20ms)={len(hi)}")
for name, grp in (("fast-AR rank", lo), ("slow-AR rank", hi)):
    if not grp:
        continue
    stats(f"  {name} ar_call", sorted(w["ar_call_us"] for w in grp))
    stats(f"  {name} ar_item", sorted(w["ar_item_us"] for w in grp))
    stats(f"  {rank_wait}", sorted(w["wait_us"] for w in grp)) if (rank_wait := f"{name} wait_us") else None
    # ar_out relative to head: where in the wave does AR finish
    outs = sorted(w["ar_out_off"] for w in grp)
    n = len(outs)
    print(
        f"  {name} ar_out_off(head-relative): p10={outs[n//10]:.1f}ms p50={outs[n//2]:.1f}ms "
        f"p90={outs[9*n//10]:.1f}ms"
    )

# cross-rank pairing: same wave = |head diff| < 50ms (if small gaps exist)
pairs = []
used = set()
for i, a in enumerate(waves):
    if i in used:
        continue
    for j in range(i + 1, len(waves)):
        if j in used:
            continue
        b = waves[j]
        if abs(b["head"] - a["head"]) < 0.05:
            pairs.append((a, b))
            used.add(i)
            used.add(j)
            break
print(f"\ncross-rank same-wave pairs (|head diff|<50ms): {len(pairs)}")
if pairs:
    def skew(field):
        return [(b[field] - a[field]) * 1000 for a, b in pairs]

    for f in ("head", "submit", "ar_in", "ar_out"):
        vals = sorted(skew(f))
        n = len(vals)
        if n:
            print(
                f"  {f}_skew ms: p10={vals[n//10]:.2f} p50={vals[n//2]:.2f} p90={vals[9*n//10]:.2f}"
            )
    print("\nfirst 6 pairs (A then B):")
    for a, b in pairs[:6]:
        print(
            f"  A head={a['head']:.3f} call={a['ar_call_us']/1000:.1f}ms | "
            f"B head={b['head']:.3f} call={b['ar_call_us']/1000:.1f}ms | "
            f"ar_in_skew={(b['ar_in']-a['ar_in'])*1000:+.1f}ms ar_out_skew={(b['ar_out']-a['ar_out'])*1000:+.1f}ms"
        )
print("WAVE_ANALYSIS_DONE")
