#!/usr/bin/env python3
"""C5/C6 combined driver (2026-09-20 redesign).

Subcommands:
  c5      arm dump.manual_dump per corpus tier, wait for shim-timed dump events
  stress  duration-limited concurrent load; every prompt carries a unique
          zero-padded sequence prefix so prefix caching cannot reuse KV
  sample  RSS (process tree) + HBM (npu-smi, per card) sampling for leakback

Env: RG_C56_CORPUS (longbench_corpus.jsonl), RG_C56_MODEL (weights dir).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

_PARA = (
    "盛唐诗人李白，字太白，号青莲居士，被后人誉为诗仙。"
    "李白的诗歌想象丰富，飘逸豪放，代表作有《将进酒》《蜀道难》《静夜思》。"
    "相传李白斗酒诗百篇，长安市上酒家眠。他的朋友杜甫写道：白也诗无敌，飘然思不群。"
)
_LONG_PROMPT = (
    "请详细介绍一下李白及其诗歌风格，并谈谈李白对后世文学的影响。\n" + _PARA * 15
    + "\n请围绕以上内容写一篇长文："
)
LEGACY_TIERS = [
    ("short", "请介绍一下长城的历史和主要关口。", 64),
    ("medium", "请介绍一下李白的人生经历和代表作品。", 128),
    ("long", _LONG_PROMPT, 256),
]
LB_TIERS = [
    ("lb1k", 1024, 64),
    ("lb4k", 4096, 64),
    ("lb16k", 16384, 128),
    ("lb32k", 32768, 128),
    ("lb64k", 65536, 128),
    ("lb128k", 131072, 128),
]


def _post(port, payload, timeout):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % port,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _completions(port, prompt, max_tokens, timeout=900):
    return _post(port, {"model": "c56", "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0, "seed": 42}, timeout)


def _load_corpus_docs():
    path = os.environ["RG_C56_CORPUS"]
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line)["text"])
    if not docs:
        raise SystemExit("empty corpus: %s" % path)
    return docs


_tok = None
_corpus_cache = {}


def _tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(
            os.environ["RG_C56_MODEL"], trust_remote_code=True)
    return _tok


def corpus_prompt(target_tokens, seq):
    """Prompt of exactly target_tokens ids, built from concatenated corpus docs.

    Cached per target across calls; the per-request sequence prefix is
    zero-padded so its token count is constant.
    """
    body = _corpus_cache.get(target_tokens)
    if body is None:
        tok = _tokenizer()
        docs = _load_corpus_docs()
        ids = []
        di = 0
        while len(ids) < target_tokens:
            piece = tok(docs[di % len(docs)], add_special_tokens=False)["input_ids"]
            ids.extend(piece)
            di += 1
        body = tok.decode(ids[:target_tokens])
        _corpus_cache[target_tokens] = body
    return "[c56-%06d] " % seq + body


def _seq_iter():
    i = 0
    while True:
        i += 1
        yield i


# ---------------------------------------------------------------- stress ---
def cmd_stress(args):
    seq = _seq_iter()
    seq_lock = threading.Lock()
    stats = {"sent": 0, "ok": 0, "fail": 0, "tokens": 0}
    deadline = time.time() + args.duration

    def worker(offset):
        idx = offset
        while time.time() < deadline:
            with seq_lock:
                n = next(seq)
            tiers = LEGACY_TIERS + [
                (name, None, mt) for name, _, mt in LB_TIERS]
            name, prompt, mt = tiers[idx % len(tiers)]
            if prompt is None:
                tt = {n: t for n, t, _ in LB_TIERS}[name]
                prompt = corpus_prompt(tt, n)
            else:
                prompt = "[c56-%06d] %s" % (n, prompt)
            try:
                out = _completions(args.port, prompt, mt)
                u = out.get("usage") or {}
                stats["tokens"] += int(u.get("total_tokens") or 0)
                stats["ok"] += 1
            except Exception:
                stats["fail"] += 1
            stats["sent"] += 1
            idx += 1

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(json.dumps({"port": args.port, "duration": args.duration,
                      **stats}))


# --------------------------------------------------------------------- c5 ---
def _config_set(path, key, value):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("dump", {})[key] = value
    tmp = path + ".driver.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _timing_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def cmd_c5(args):
    reload_wait = args.reload_interval + 0.5
    for target in [int(x) for x in args.tiers.split(",")]:
        for rep in range(1, args.reps + 1):
            _config_set(args.config, "manual_dump", 0)
            time.sleep(reload_wait)
            offset = len(_timing_lines(args.timing))
            _config_set(args.config, "manual_dump", 1)
            seq = next(_seq_iter())
            prompt = corpus_prompt(target, seq)
            t0 = time.time()
            try:
                out = _completions(args.port, prompt, args.gen_tokens,
                                   timeout=args.req_timeout)
                usage = out.get("usage") or {}
                ptoks = int(usage.get("prompt_tokens") or 0)
                ctoks = int(usage.get("completion_tokens") or 0)
            except Exception as exc:
                print(json.dumps({"tier": target, "rep": rep,
                                  "error": repr(exc)}), flush=True)
                continue
            gen_s = round(time.time() - t0, 1)
            deadline = time.time() + args.dump_timeout
            while time.time() < deadline:
                lines = _timing_lines(args.timing)[offset:]
                if any('"d2h_summary"' in l for l in lines):
                    time.sleep(args.quiet_sec)  # trailing save events
                    lines = _timing_lines(args.timing)[offset:]
                    break
                time.sleep(1.0)
            ranks = {}
            for l in lines:
                try:
                    ev = json.loads(l)
                except Exception:
                    continue
                if ev.get("event") == "d2h_summary":
                    tag = ev.get("rank_tag") or "rank?"
                    ranks.setdefault(tag, {})["d2h_ms"] = ev.get("d2h_ms")
                    ranks[tag]["layers"] = ev.get("layers")
                    ranks[tag]["d2h_bytes"] = ev.get("bytes")
                    ranks[tag]["req_id"] = ev.get("req_id")
                elif ev.get("event") == "save":
                    tag = ev.get("rank_tag") or "rank?"
                    r = ranks.setdefault(tag, {})
                    r["save_ms"] = r.get("save_ms", 0) + ev.get("save_ms", 0)
                    r["save_files"] = r.get("save_files", 0) + ev.get("files", 0)
                    r["save_bytes"] = r.get("save_bytes", 0) + ev.get("bytes", 0)
            print(json.dumps({"tier": target, "rep": rep,
                              "prompt_tokens": ptoks, "completion_tokens": ctoks,
                              "gen_s": gen_s, "ranks": ranks}, ensure_ascii=False),
                  flush=True)
            _config_set(args.config, "manual_dump", 0)


# ---------------------------------------------------------------- sample ---
def _children_map():
    ppid = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry) as f:
                parts = f.read().rsplit(") ", 1)[1].split()
            ppid[int(entry)] = int(parts[1])
        except Exception:
            pass
    return ppid


def _rss_tree_kb(root):
    ppid = _children_map()
    seen, stack, total = set(), [root], 0
    pg = os.sysconf("SC_PAGE_SIZE") / 1024.0
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            with open("/proc/%d/statm" % pid) as f:
                total += int(f.read().split()[1]) * pg
        except Exception:
            pass
        stack.extend(p for p, q in ppid.items() if q == pid and p not in seen)
    return int(total)


_HBM_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _hbm_mb(cards):
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return {}
    res = {}
    want = {str(c) for c in cards}
    lines = out.splitlines()
    for i, line in enumerate(lines):
        head = [t.strip() for t in line.split("|")]
        # npu-smi 25.5: col-1 cell holds "NPU Name" together; the NPU-id row
        # carries no HBM — the chip row below it packs "AICore Mem/MB HBM/MB"
        # into one cell -> take the last X / Y pair of that row.
        if len(head) >= 2 and head[1].split() and head[1].split()[0] in want \
                and i + 1 < len(lines):
            pairs = _HBM_RE.findall(lines[i + 1])
            if pairs:
                nid = int(head[1].split()[0])
                res[nid] = int(pairs[-1][0])
    return res


def cmd_sample(args):
    arms = []
    for spec in args.arms.split(","):
        name, pid, cards = spec.split(":")
        arms.append((name, int(pid), [int(c) for c in cards.split("+")]))
    time.sleep(args.settle)
    deadline = time.time() + args.duration
    with open(args.out, "a", encoding="utf-8") as f:
        while time.time() < deadline:
            now = time.strftime("%F %H:%M:%S")
            for name, pid, cards in arms:
                hbm = _hbm_mb(cards)
                f.write(json.dumps({
                    "ts": now, "phase": args.phase, "arm": name,
                    "rss_kb": _rss_tree_kb(pid),
                    "hbm_mb": hbm}) + "\n")
            f.flush()
            time.sleep(args.interval)
    for name, pid, cards in arms:
        print(json.dumps({"summary_for": name, "arm": name}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stress")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=cmd_stress)
    p = sub.add_parser("c5")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--timing", required=True)
    p.add_argument("--tiers", default="1024,4096,16384,65536,131072")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--reload-interval", type=float, default=2.0)
    p.add_argument("--req-timeout", type=float, default=900)
    p.add_argument("--dump-timeout", type=float, default=300)
    p.add_argument("--quiet-sec", type=float, default=8)
    p.set_defaults(func=cmd_c5)
    p = sub.add_parser("sample")
    p.add_argument("--arms", required=True,
                   help="A:PID:C0+C1,B:PID:C2+C3")
    p.add_argument("--settle", type=float, default=30)
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--interval", type=float, default=10)
    p.add_argument("--phase", default="leakback")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_sample)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
