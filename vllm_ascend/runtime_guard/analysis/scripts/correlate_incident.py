#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correlate a req_id's anomaly report(s) with kv_cache dump files (Step 5 / 5.5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def find_reports(
    report_dir: Path,
    req_id: str,
    incident_type: str | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    root = report_dir
    if incident_type:
        candidates = list((root / incident_type).glob("report_*.json"))
        candidates += list((root / incident_type).glob("anomaly_*.log"))
    else:
        candidates = list(root.glob("*/report_*.json"))
        candidates += list(root.glob("report_*.json"))
        candidates += list(root.glob("*/anomaly_*.log"))
        candidates += list(root.glob("anomaly_*.log"))

    matched: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        data = _load_json(path)
        if data is None:
            continue
        if str(data.get("req_id")) != req_id and req_id not in path.name:
            continue
        if incident_type:
            itype = str(data.get("incident_type") or path.parent.name)
            if itype != incident_type and path.parent.name != incident_type:
                continue
        matched.append((path, data))
    return matched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("runtime/report"))
    parser.add_argument("--req-id", type=str, required=True)
    parser.add_argument("--incident-type", type=str, default=None)
    args = parser.parse_args(argv)

    root = args.report_dir
    req_id = args.req_id
    matched = find_reports(root, req_id, args.incident_type)

    print(f"report_dir={root.resolve()} req_id={req_id} reports={len(matched)}")
    for path, data in matched:
        detail = data.get("detail") if isinstance(data.get("detail"), dict) else {}
        itype = str(data.get("incident_type") or path.parent.name)
        print(
            f"[report] {path}\n"
            f"  incident_type={itype} dump_armed={data.get('dump_armed')} "
            f"arm_wave={data.get('dump_arm_wave')} block_ids={detail.get('block_ids')}"
        )
        kv_dir = root / "kv_cache" / itype / req_id
        shared = root / "kv_cache" / itype / "shared"
        for label, d in (("req", kv_dir), ("shared", shared)):
            if not d.is_dir():
                print(f"  [kv:{label}] missing {d}")
                continue
            pts = sorted(d.glob("*.pt"))
            print(f"  [kv:{label}] dir={d} files={len(pts)}")
            for pt in pts[:12]:
                print(f"    - {pt.name}")
            if len(pts) > 12:
                print(f"    ... +{len(pts) - 12} more")
        print(
            "  next: python -m vllm_ascend.runtime_guard.analysis.scripts.verify_request_kv "
            f"--report {path} --report-dir {root}"
        )

    if not matched:
        kv_root = root / "kv_cache"
        if kv_root.is_dir():
            hits = list(kv_root.glob(f"*/{req_id}")) + list(kv_root.glob("*/shared"))
            print(f"No reports matched; kv hits={len(hits)}")
            for d in hits:
                print(f"  {d}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
