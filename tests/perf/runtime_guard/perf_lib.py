"""Shared helpers for runtime_guard perf + memory scripts.

T-label terminology (see tests/perf/runtime_guard/README.md):
- T0: merge-base 37e382498 worktree (no runtime_guard code)
- T1: HEAD with default startup (no --additional-config, reload=0, detectors off)
- T2: HEAD + reload=3 + detectors off
- T3: HEAD + reload=3 + 6 detectors on

perf_baseline.py    -> T1 (or T0 if run from merge-base worktree)
perf_ab_quick.py    -> T3 "B" vs T2 "A" x3 cross-rotated

IMPORTANT: A/B here mean detector off/on WITHIN reload=3 — NOT the same as
dfx-perf-bench SKILL's A/B which mean no-DFX / DFX+reload>0+off.

Memory sampling: every round records pre/post RSS of the vllm parent+worker
PIDs and per-NPU HBM-Usage. A 5-min idle leak-back phase runs after the
measured rounds: if RSS does not return to within 30MB of the start, a
per-req state container is leaking.
"""

import json
import os
import re
import subprocess
import time
import urllib.request

URL = os.environ.get("RG_PERF_URL", "http://127.0.0.1:8017/v1/completions")
CFG = os.environ.get("RG_PERF_CFG",
                    "/workspace/djh-testruntime/rg_test/config/runtime_config.json")
OUT_AB = os.environ.get("RG_PERF_OUT_AB",
                       "/workspace/djh-testruntime/rg_test/logs/perf_ab_quick.jsonl")
OUT_BASELINE = os.environ.get(
    "RG_PERF_OUT_BASELINE",
    "/workspace/djh-testruntime/rg_test/logs/perf_baseline.jsonl")
OUT_LEAKBACK_BASELINE = os.environ.get(
    "RG_PERF_OUT_LEAKBACK_BASELINE",
    "/workspace/djh-testruntime/rg_test/logs/leakback_baseline.jsonl")
OUT_LEAKBACK_AB = os.environ.get(
    "RG_PERF_OUT_LEAKBACK_AB",
    "/workspace/djh-testruntime/rg_test/logs/leakback_ab.jsonl")
LEAKBACK_SECONDS = int(os.environ.get("RG_PERF_LEAKBACK_SEC", "300"))
LEAKBACK_INTERVAL = int(os.environ.get("RG_PERF_LEAKBACK_INTERVAL", "60"))
NPU_INDICES = os.environ.get("RG_PERF_NPU", "6,7").split(",")

_para = (
    "盛唐诗人李白，字太白，号青莲居士，被后人誉为诗仙。"
    "李白的诗歌想象丰富，飘逸豪放，代表作有《将进酒》《蜀道难》《静夜思》。"
    "相传李白斗酒诗百篇，长安市上酒家眠。他的朋友杜甫写道：白也诗无敌，飘然思不群。"
)
LONG_PROMPT = (
    "请详细介绍一下李白及其诗歌风格，并谈谈李白对后世文学的影响。\n" + _para * 15
    + "\n请围绕以上内容写一篇长文："
)

REQS = [
    ("short", "请介绍一下长城的历史和主要关口。", 64),
    ("medium", "请介绍一下李白的人生经历和代表作品。", 128),
    ("long", LONG_PROMPT, 256),
]


def post(prompt, max_tokens):
    body = json.dumps(
        {"model": "dsv2", "prompt": prompt, "max_tokens": max_tokens,
         "temperature": 0, "seed": 42}
    ).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    usage = out.get("usage", {})
    wall = time.time() - t0
    ct = usage.get("completion_tokens", 0)
    return wall, usage.get("prompt_tokens", 0), ct, (ct / wall if wall > 0 else 0.0)


def set_detectors(enabled):
    """Flip every detector.<name>.enabled in runtime_config.json (hot-reloaded)."""
    cfg = json.load(open(CFG))
    for sec in (cfg.get("detector") or {}).values():
        if isinstance(sec, dict):
            sec["enabled"] = enabled
    json.dump(cfg, open(CFG, "w"), indent=2)


def trigger():
    """Small request to force a scheduler step so the config reload picks up."""
    post("你好", 2)


def warmup(rounds=1):
    for _ in range(rounds):
        for _, p, mt in REQS:
            post(p, mt)


def _vllm_pids():
    """Return list of live vllm-related PIDs (parent serve + TP workers).

    Strategy: find the parent "vllm serve" PID, then walk /proc/<pid>/task
    descendants to get TP workers. Skips defunct (state Z) processes.
    """
    pids = set()
    # Parent serve PID(s)
    for pattern in (r"[v]llm serve",):
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", pattern], text=True
            ).strip().split("\n")
            for p in out:
                if p:
                    pids.add(int(p))
        except subprocess.CalledProcessError:
            continue

    # Walk descendants of each parent (children of vllm serve = TP workers)
    def _children(parent_pid):
        try:
            out = subprocess.check_output(
                ["pgrep", "-P", str(parent_pid)], text=True
            ).strip().split("\n")
            return [int(p) for p in out if p]
        except subprocess.CalledProcessError:
            return []

    to_visit = list(pids)
    visited = set()
    while to_visit:
        p = to_visit.pop()
        if p in visited:
            continue
        visited.add(p)
        for child in _children(p):
            if child not in pids:
                pids.add(child)
                to_visit.append(child)

    # Filter out defunct (state Z) processes — they hold no RSS
    live = []
    for p in pids:
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("State:"):
                        state = line.split()[1]
                        if state != "Z":
                            live.append(p)
                        break
        except FileNotFoundError:
            continue
    return sorted(live)


def _sample_npu_hbm():
    """Return {npu_index: used_mb} from npu-smi info, for NPU_INDICES."""
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True)
    except Exception:
        return {}
    # npu-smi chip line layout (note: Memory-Usage and HBM-Usage share one
    # pipe-section, e.g. "0           0    / 0          28975/ 32768"). Match
    # the LAST "used/total" pair (HBM), not the first (NPUMemoryXel).
    hbm_re = re.compile(r"(\d+)\s*/\s*\d+\s*$")
    result = {}
    lines = out.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"\|\s+(\d+)\s+910B4", line)
        if not m:
            continue
        npu_idx = m.group(1)
        for j in range(i + 1, min(i + 4, len(lines))):
            if "0000:" in lines[j]:
                fields = [f.strip() for f in lines[j].split("|") if f.strip()]
                if len(fields) >= 2:
                    hm = hbm_re.search(fields[-1])
                    if hm:
                        result[npu_idx] = int(hm.group(1))
                break
    return {idx: result.get(idx, 0) for idx in NPU_INDICES}


def sample_mem():
    """Snapshot RSS (KB) of all vllm PIDs + per-NPU HBM (MB)."""
    pids = _vllm_pids()
    rss_kb = 0
    if pids:
        try:
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
                text=True,
            ).strip().split("\n")
            rss_kb = sum(int(x.strip()) for x in out if x.strip())
        except (subprocess.CalledProcessError, ValueError):
            pass
    hbm = _sample_npu_hbm()
    return {
        "rss_kb": rss_kb,
        "hbm_mb": hbm,
        "pids": pids,
    }


def make_rec(round_no, state, tag, wall, pt, ct, tps, mem_pre=None, mem_post=None):
    rec = {
        "round": round_no, "state": state, "tag": tag,
        "wall_s": round(wall, 2), "prompt_tok": pt,
        "compl_tok": ct, "out_tok_s": round(tps, 2),
    }
    if mem_pre is not None:
        rec["pre_rss_kb"] = mem_pre["rss_kb"]
        rec["pre_hbm_mb"] = mem_pre["hbm_mb"]
    if mem_post is not None:
        rec["post_rss_kb"] = mem_post["rss_kb"]
        rec["post_hbm_mb"] = mem_post["hbm_mb"]
    return rec


def run_rounds(rounds_spec, out_path, with_mem=True):
    """Run measured rounds, optionally sampling memory per-round.

    rounds_spec: list of (state_label, det_setting) tuples. det_setting is
    True/False to flip detectors before the round, or None to skip toggling
    (baseline server without a runtime_config).
    """
    with open(out_path, "w") as f:
        for rnd_no, (state, det_setting) in enumerate(rounds_spec, 1):
            if det_setting is not None:
                set_detectors(det_setting)
                trigger()
                time.sleep(1)
            for tag, p, mt in REQS:
                mem_pre = sample_mem() if with_mem else None
                wall, pt, ct, tps = post(p, mt)
                mem_post = sample_mem() if with_mem else None
                rec = make_rec(rnd_no, state, tag, wall, pt, ct, tps, mem_pre, mem_post)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(json.dumps(rec), flush=True)


def run_leakback(state, out_path, seconds=None, interval=None):
    """Idle leak-back phase: sample memory every interval for seconds.

    Writes one JSONL line per sample. If end rss_delta_kb > 30*1024 (30MB),
    a per-req state container is leaking (see SYSTEM_OBSERVABILITY_TEST.md 1.2).
    """
    seconds = seconds if seconds is not None else LEAKBACK_SECONDS
    interval = interval if interval is not None else LEAKBACK_INTERVAL
    start = sample_mem()
    start_ts = time.time()
    with open(out_path, "w") as f:
        rec = {"phase": "leakback_start", "state": state, "ts": start_ts}
        rec.update(start)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps(rec), flush=True)
        n_samples = max(1, int(seconds / interval))
        for i in range(n_samples):
            time.sleep(interval)
            s = sample_mem()
            rec = {
                "phase": "leakback_sample",
                "state": state,
                "ts": time.time(),
                "elapsed_s": (i + 1) * interval,
            }
            rec.update(s)
            rec["rss_delta_kb"] = s["rss_kb"] - start["rss_kb"]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)
        end = sample_mem()
        rec = {
            "phase": "leakback_end",
            "state": state,
            "ts": time.time(),
            "rss_delta_kb": end["rss_kb"] - start["rss_kb"],
        }
        rec.update(end)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps(rec), flush=True)
