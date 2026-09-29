#!/usr/bin/env python3
"""Analyze [RG-BUS-WV2] per-wave lines: rank wave-count pairing forensics.

Answers, with explicit wave indices ("rank 是第几次"):
  1. idx coverage per rank — any wave where one rank ran the bus AR and the
     other did not (count divergence = permanent Gloo seq offset origin)
  2. sync counter vs bus counter per rank — sync_for_step calls that skipped
     the merged bus (idle/static branch) between bus waves
  3. same-idx pairing skew: head / ar_in / ar_out / fwd_in / fwd_out
  4. adjacent-idx coupling fingerprint: slow side's ar_out(k) vs other side's
     head(k+1) — the one-wave-offset evidence, now with explicit idx
  5. first N waves detail (boot window)
  6. cross-check [RG-BUS-AR-IN]/[RG-BUS-AR-OUT] worker logs vs drain recall
"""
import re
import sys
from pathlib import Path

LOG = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/data0/test-mrv2-cann91/rg_probe_tip15/serve_t2.log"
)

WV2 = re.compile(
    r"\[RG-BUS-WV2\] idx=(-?\d+) sync=(-?\d+) tp=(-?\d+) head=([\d.]+) "
    r"submit=\+([-\d.]+)ms ar_in=\+([-\d.]+)ms ar_out=\+([-\d.]+)ms "
    r"fwd_in=\+([-\d.]+)ms fwd_out=\+([-\d.]+)ms "
    r"ar_call_us=([\d.]+) ar_item_us=([\d.]+) sched_us=([\d.]+) "
    r"wait_us=([\d.]+) drain_end=([\d.]+) gate=(\w+)"
)
AR_IN = re.compile(r"\[RG-BUS-AR-IN\] idx=(-?\d+) tp=(-?\d+) wall=([\d.]+)")
AR_OUT = re.compile(
    r"\[RG-BUS-AR-OUT\] idx=(-?\d+) tp=(-?\d+) wall=([\d.]+) "
    r"enter=([\d.]+) exit=([\d.]+) call_us=([\d.]+) item_us=([\d.]+)"
)
WORKER_STARTED = re.compile(r"\[runtime_guard\] bus worker started name=(\S+)")
# tp label fallback: log prefix "(Worker_TPn pid=...)" when tp= field is -1
PREFIX_TP = re.compile(r"\(Worker_TP(\d+) pid=")

waves: dict[int, dict[int, dict]] = {}
ar_in: dict[tuple[int, int], float] = {}
ar_out: dict[tuple[int, int], dict] = {}
worker_starts: list[float] = []

def tp_of(line: str, field_tp: int) -> int:
    if field_tp >= 0:
        return field_tp
    m = PREFIX_TP.search(line)
    return int(m.group(1)) if m else field_tp

for line in LOG.read_text(errors="ignore").splitlines():
    m = WV2.search(line)
    if m:
        tp = tp_of(line, int(m.group(3)))
        idx = int(m.group(1))
        w = {
            "idx": idx,
            "sync": int(m.group(2)),
            "head": float(m.group(4)),
            "submit": float(m.group(4)) + float(m.group(5)) / 1000.0,
            "ar_in": float(m.group(4)) + float(m.group(6)) / 1000.0,
            "ar_out": float(m.group(4)) + float(m.group(7)) / 1000.0,
            "fwd_in": float(m.group(4)) + float(m.group(8)) / 1000.0,
            "fwd_out": float(m.group(4)) + float(m.group(9)) / 1000.0,
            "has_fwd": float(m.group(8)) != 0.0,
            "ar_call_us": float(m.group(10)),
            "ar_item_us": float(m.group(11)),
            "sched_us": float(m.group(12)),
            "wait_us": float(m.group(13)),
            "drain_end": float(m.group(14)),
            "gate": m.group(15),
            "raw": line.strip(),
        }
        waves.setdefault(tp, {})[idx] = w
        continue
    m = AR_IN.search(line)
    if m:
        ar_in[(tp_of(line, int(m.group(2))), int(m.group(1)))] = float(m.group(3))
        continue
    m = AR_OUT.search(line)
    if m:
        ar_out[(tp_of(line, int(m.group(2))), int(m.group(1)))] = {
            "wall": float(m.group(3)),
            "enter": float(m.group(4)),
            "exit": float(m.group(5)),
            "call_us": float(m.group(6)),
            "item_us": float(m.group(7)),
        }
        continue
    m = WORKER_STARTED.search(line)
    if m:
        worker_starts.append(0.0)  # count only; wall ts not in this line

tps = sorted(waves)
if len(tps) < 2:
    print(f"NEED TWO TP GROUPS, got {tps} (waves={sum(len(v) for v in waves.values())})")
    sys.exit(1)
a, b = tps[0], tps[1]
wa, wb = waves[a], waves[b]
print(f"n_waves: tp{a}={len(wa)} tp{b}={len(wb)}; worker_starts={len(worker_starts)}")
print(f"idx range: tp{a}={min(wa)}..{max(wa)} tp{b}={min(wb)}..{max(wb)}")

# ---- 1. idx coverage divergence ----
sa, sb = set(wa), set(wb)
only_a = sorted(sa - sb)
only_b = sorted(sb - sa)
print(f"\n[idx coverage] only tp{a}: {only_a[:20]}{'...' if len(only_a) > 20 else ''}")
print(f"[idx coverage] only tp{b}: {only_b[:20]}{'...' if len(only_b) > 20 else ''}")
if not only_a and not only_b:
    print("[idx coverage] IDENTICAL idx sets — both ranks ran the AR every wave")

# ---- 2. sync counter continuity per rank ----
for tp, wv in ((a, wa), (b, wb)):
    items = sorted(wv.items())
    jumps = [
        (k, wv[k - 1]["sync"], wv[k]["sync"])
        for k, _ in items
        if k - 1 in wv and wv[k]["sync"] - wv[k - 1]["sync"] != 1
    ]
    print(f"\n[sync jumps] tp{tp}: {len(jumps)} waves with sync-for-step delta != 1")
    for k, s0, s1 in jumps[:10]:
        note = "skipped bus (sync advanced w/o AR)" if s1 - s0 > 1 else "repeated sync?"
        print(f"  idx {k}: sync {s0} -> {s1} ({note})")

# ---- 3. same-idx pairing skew ----
common = sorted(sa & sb)
if common:
    def skews(field):
        vals = []
        for k in common:
            if field == "head":
                vals.append(wb[k]["head"] - wa[k]["head"])
            else:
                va, vb = wa[k][field], wb[k][field]
                if va and vb:
                    vals.append(vb - va)
        return vals

    period_a = sorted(wb[k]["head"] - wa[k]["head"] for k in common)
    med = sorted(wa[k + 1]["head"] - wa[k]["head"] for k in common if k + 1 in wa)
    wave_ms = 1000 * med[len(med) // 2] if med else 0.0
    print(f"\n[wave period] median head-to-head = {wave_ms:.1f}ms")
    for field in ("head", "submit", "ar_in", "ar_out", "fwd_in", "fwd_out"):
        vals = sorted(skews(field))
        if not vals:
            continue
        n = len(vals)
        neg = sum(1 for v in vals if v < -0.02)
        pos = sum(1 for v in vals if v > 0.02)
        print(
            f"[same-idx skew tp{b}-tp{a}] {field}: n={n} p10={1000*vals[n//10]:.1f}ms "
            f"p50={1000*vals[n//2]:.1f}ms p90={1000*vals[9*n//10]:.1f}ms "
            f"max={1000*vals[-1]:.1f}ms | ~wave-aligned: {neg + pos}/{n}"
        )
    print(
        "  (skew ~0 = same-idx waves truly overlap; skew ~ 1 wave period = "
        "one rank's idx k runs a full wave ahead of the other's idx k)"
    )

# ---- 4. adjacent-idx coupling fingerprint ----
print("\n[coupling] slow-side ar_out(k) vs other-side head(k+1):")
for slow, fast in ((a, b), (b, a)):
    ws, wf = waves[slow], waves[fast]
    hits = 0
    diffs = []
    for k, w in ws.items():
        if w["ar_call_us"] < 20000 or (k + 1) not in wf:
            continue
        d = w["ar_out"] - wf[k + 1]["head"]
        if abs(d) < 0.01:
            hits += 1
            diffs.append(d * 1000)
    slow_n = sum(1 for w in ws.values() if w["ar_call_us"] >= 20000)
    if slow_n:
        print(
            f"  tp{slow} slow waves={slow_n}: ar_out(k)==head(k+1) of tp{fast} "
            f"in {hits} cases (median |diff|={sorted(abs(x) for x in diffs)[len(diffs)//2]:.1f}ms)"
            if diffs
            else f"  tp{slow} slow waves={slow_n}: no tight coupling to tp{fast} head(k+1)"
        )

# ---- 5. first N waves detail ----
print("\n[first waves]")
hdr = (
    f"{'idx':>5} {'tp':>2} {'sync':>6} {'head':>12} {'submit':>8} {'ar_in':>8} "
    f"{'ar_out':>8} {'fwd_in':>8} {'fwd_out':>8} {'ar_call':>8} {'wait':>7}"
)
print(hdr)
for k in range(0, 14):
    for tp, wv in ((a, wa), (b, wb)):
        w = wv.get(k)
        if not w:
            print(f"{k:>5} {tp:>2} ---missing---")
            continue
        fi = f"{(w['fwd_in'] - w['head']) * 1000:7.1f}" if w["has_fwd"] else "    n/a"
        fo = f"{(w['fwd_out'] - w['head']) * 1000:7.1f}" if w["has_fwd"] else "    n/a"
        print(
            f"{k:>5} {tp:>2} {w['sync']:>6} {w['head']:>12.3f} "
            f"{(w['submit'] - w['head']) * 1000:>7.1f}m {(w['ar_in'] - w['head']) * 1000:>7.1f}m "
            f"{(w['ar_out'] - w['head']) * 1000:>7.1f}m {fi} {fo} "
            f"{w['ar_call_us']:>7.1f}u {w['wait_us']:>6.1f}u"
        )

# ---- 6. worker AR-IN/AR-OUT cross-check ----
print("\n[worker AR log cross-check]")
for tp in (a, b):
    n_in = sum(1 for (t, _) in ar_in if t == tp)
    n_out = sum(1 for (t, _) in ar_out if t == tp)
    mismatch = 0
    for (t, k), rec in ar_out.items():
        if t != tp or k not in waves.get(tp, {}):
            continue
        if abs(rec["exit"] - waves[tp][k]["ar_out"]) > 0.002:
            mismatch += 1
    print(f"  tp{tp}: AR-IN={n_in} AR-OUT={n_out} wv2={len(waves.get(tp, {}))} exit-mismatch={mismatch}")

# AR enter walls per same idx across ranks — the seq pairing question
if common:
    enter_skews = sorted(ar_out[(b, k)]["enter"] - ar_out[(a, k)]["enter"] for k in common if (a, k) in ar_out and (b, k) in ar_out)
    if enter_skews:
        n = len(enter_skews)
        print(
            f"\n[AR enter skew same-idx tp{b}-tp{a}]: n={n} p10={1000*enter_skews[n//10]:.1f}ms "
            f"p50={1000*enter_skews[n//2]:.1f}ms p90={1000*enter_skews[9*n//10]:.1f}ms"
        )
        print("  (enter skew ~0: Gloo seq aligned but pairing blocked elsewhere;")
        print("   enter skew ~ 1 wave: seq misaligned — one rank's seq n pairs other's next wave)")

print("\nANALYZE_WV2_DONE")
