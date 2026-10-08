#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Refresh a golden report summary from a live report_*.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from diff_report_golden import IGNORE, _critical, _load  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--schema-only", action="store_true")
    p.add_argument(
        "--notes",
        type=str,
        default="Refreshed from lab report; re-review before commit",
    )
    args = p.parse_args(argv)

    report = _load(args.report)
    payload: dict[str, Any] = _critical(report, args.schema_only)
    payload["notes"] = args.notes
    payload = {k: v for k, v in payload.items() if v is not None and k not in IGNORE}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
