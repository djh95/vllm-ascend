"""Shared helpers for runtime_guard perf scripts (B/A toggle and baseline)."""

import json
import os
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
    """Flip every ``detector.<name>.enabled`` in runtime_config.json (hot-reloaded by server)."""
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


def make_rec(round_no, state, tag, wall, pt, ct, tps):
    return {
        "round": round_no, "state": state, "tag": tag,
        "wall_s": round(wall, 2), "prompt_tok": pt,
        "compl_tok": ct, "out_tok_s": round(tps, 2),
    }


def run_rounds(rounds_spec, out_path):
    """Run measured rounds.

    ``rounds_spec``: list of ``(state_label, det_setting)`` tuples. ``det_setting``
    is ``True``/``False`` to flip detectors before the round, or ``None`` to
    skip toggling (used for baseline server without a runtime_config).
    """
    with open(out_path, "w") as f:
        for rnd_no, (state, det_setting) in enumerate(rounds_spec, 1):
            if det_setting is not None:
                set_detectors(det_setting)
                trigger()
                time.sleep(1)
            for tag, p, mt in REQS:
                wall, pt, ct, tps = post(p, mt)
                rec = make_rec(rnd_no, state, tag, wall, pt, ct, tps)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(json.dumps(rec), flush=True)
