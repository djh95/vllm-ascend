#!/usr/bin/env python3
"""Active idle leak-back check against a running soak instance.

Burst N requests -> sample peak RSS -> idle -> sample final RSS.
Reports whether host RSS returns toward baseline (per-req state leak detector).

Usage (inside test-mrv2-cann91):
  /opt/slime/venv/bin/python leakback_check.py <served-model-name> <port> [N] [idle_sec]
"""
import json
import subprocess
import sys
import time


def instance_rss_kb(served_name):
    """Sum VmRSS (KB) of the api_server (matched by served-model-name) + its
    direct EngineCore/worker children."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,rss=,cmd="], capture_output=True, text=True
        ).stdout
    except Exception:
        return 0
    api_pids = []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, rss, cmd = parts[0], parts[1], parts[2], parts[3]
        if not rss.isdigit():
            continue
        rows.append((pid, ppid, int(rss), cmd))
        if served_name in cmd and "api_server" in cmd:
            api_pids.append(pid)
    if not api_pids:
        return 0
    total = 0
    for pid, ppid, rss, cmd in rows:
        if pid in api_pids or ppid in api_pids:
            total += rss
    return total


def fire_burst(port, served_name, n):
    ok = 0
    for i in range(n):
        body = json.dumps({"model": served_name, "prompt": f"Tell me a short fact about number {i}",
                           "max_tokens": 128, "temperature": 0.7})
        try:
            subprocess.run(["curl", "-s", f"http://127.0.0.1:{port}/v1/completions",
                            "-H", "Content-Type: application/json", "-d", body],
                           capture_output=True, timeout=60)
            ok += 1
        except Exception:
            pass
    return ok


def main():
    served, port = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    idle_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 180

    base = instance_rss_kb(served)
    print(f"[leakback] {served} baseline_rss={base}KB", flush=True)
    ok = fire_burst(port, served, n)
    print(f"[leakback] burst {ok}/{n} req done", flush=True)
    peak = instance_rss_kb(served)
    print(f"[leakback] peak_rss={peak}KB", flush=True)

    for waited in range(0, idle_sec, 30):
        time.sleep(min(30, idle_sec - waited))
        cur = instance_rss_kb(served)
        print(f"[leakback] idle {waited+30:>3}s rss={cur}KB", flush=True)

    final = instance_rss_kb(served)
    delta = final - base
    verdict = "PASS" if delta <= 30 * 1024 else "FAIL"
    print(f"[leakback] RESULT served={served} base={base}KB peak={peak}KB "
          f"final={final}KB delta={delta}KB -> {verdict} (threshold 30MB)", flush=True)


if __name__ == "__main__":
    main()
