#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a live report against a golden summary (key fields only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Floating fields ignored in equality checks.
IGNORE = {
    "req_id",
    "timestamp",
    "time",
    "dump_dir",
    "rank",
    "dump_arm_wave",
    "path",
}


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"not an object: {path}")
    return data


def _pick_report(report_dir: Path, incident_type: str | None) -> Path:
    roots = [report_dir / incident_type] if incident_type else [report_dir]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(root.glob("report_*.json"))
            files.extend(root.glob("*/report_*.json"))
    if not files:
        raise SystemExit(f"no report_*.json under {report_dir}")
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _critical(report: dict[str, Any], schema_only: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "incident_type": report.get("incident_type"),
        "dump_attempted": report.get("dump_attempted", report.get("dump_armed")),
    }
    if schema_only:
        return out
    detail = report.get("detail") if isinstance(report.get("detail"), dict) else {}
    for key in (
        "ill_type",
        "kind",
        "pattern",
        "match_prefix",
        "repeat_sum_threshold",
        "window",
    ):
        if key in detail:
            out[key] = detail[key]
        if key in report:
            out[key] = report[key]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--incident-type", type=str, default=None)
    p.add_argument("--schema-only", action="store_true")
    args = p.parse_args(argv)

    golden = _load(args.golden)
    if args.report is not None:
        report_path = args.report
    elif args.report_dir is not None:
        report_path = _pick_report(args.report_dir, args.incident_type)
    else:
        raise SystemExit("pass --report or --report-dir")

    report = _load(report_path)
    got = _critical(report, args.schema_only)
    exp = {k: v for k, v in golden.items() if k not in IGNORE}
    # Only compare keys present in golden.
    mismatches = []
    for key, want in exp.items():
        if key not in got:
            mismatches.append(f"missing {key}: want {want!r}")
        elif got[key] != want:
            mismatches.append(f"{key}: got {got[key]!r} want {want!r}")

    print(f"report={report_path}")
    print(f"golden={args.golden}")
    print(f"got={got}")
    if mismatches:
        print("FAIL:")
        for m in mismatches:
            print(f"  - {m}")
        return 1
    print("PASS — critical fields match golden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
