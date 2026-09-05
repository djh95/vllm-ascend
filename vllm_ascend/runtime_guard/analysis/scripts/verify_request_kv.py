#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a runtime_guard report against its per-request KV dump folder (Step 5.5).

Unlike msprobe ``dump_tensor_data`` layouts, runtime_guard ``dump_kv`` writes::

    <report_dir>/kv_cache/<incident_type>/<req_id>/*.pt

each ``.pt`` is a dict with keys ``req_id``, ``block_ids``, ``layer``, ``tensor``, ...

Checks (token counts are informational only):
  1. report has req_id / incident_type
  2. block capacity vs token N (when block_size known)
  3. KV files exist and tensors are finite-ish
  4. file count / layers present
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return data


def _token_n(detail: dict[str, Any]) -> tuple[int, int, int]:
    prompt_ids = detail.get("prompt_token_ids") or []
    output_ids = detail.get("output_token_ids") or []
    n_prompt = len(prompt_ids) if isinstance(prompt_ids, list) else int(detail.get("prompt_token_count") or 0)
    n_output = len(output_ids) if isinstance(output_ids, list) else int(detail.get("output_token_count") or 0)
    return n_prompt, n_output, n_prompt + n_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="runtime_guard report_*.json")
    parser.add_argument(
        "--kv-dir",
        type=Path,
        default=None,
        help="KV dump dir (default: <report_dir>/kv_cache/<type>/<req_id>)",
    )
    parser.add_argument("--report-dir", type=Path, default=None, help="Override report root for default kv-dir")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--min-files", type=int, default=1, help="Minimum expected .pt files")
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch is required")
        return 2

    report = _load_report(args.report)
    detail = report.get("detail") if isinstance(report.get("detail"), dict) else {}
    req_id = str(report.get("req_id") or "")
    itype = str(report.get("incident_type") or args.report.parent.name)
    ok = True

    print("[1] report basics")
    print(f"    path={args.report}")
    print(f"    incident_type={itype} req_id={req_id or '-'} dump_armed={report.get('dump_armed')}")
    if not req_id:
        print("    [FAIL] missing req_id")
        ok = False

    n_prompt, n_output, n_total = _token_n(detail)
    print("[2] token 数（信息）")
    print(f"    prompt={n_prompt} output={n_output} N={n_total}")
    if detail.get("prompt_token_count") not in (None, n_prompt):
        print(f"    [note] prompt_token_count={detail.get('prompt_token_count')} vs ids len={n_prompt}")
    if detail.get("output_token_count") not in (None, n_output):
        print(f"    [note] output_token_count={detail.get('output_token_count')} vs ids len={n_output}")

    block_ids = list(detail.get("block_ids") or report.get("block_ids") or [])
    print("[3] block 容量")
    if block_ids:
        cap = len(block_ids) * args.block_size
        status = "OK" if (n_total == 0 or cap >= n_total) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"    block_ids={block_ids} cap={cap} N={n_total} → {status}")
    else:
        print("    [WARN] no block_ids in report (include_block_ids off?)")

    if args.kv_dir is not None:
        kv_dir = args.kv_dir
    else:
        report_dir = args.report_dir
        if report_dir is None:
            # .../report/<type>/file.json → report root is parent.parent
            report_dir = args.report.parent.parent if args.report.parent.name == itype else args.report.parent
        kv_dir = report_dir / "kv_cache" / itype / (req_id or "_missing_req_")

    print("[4] KV files")
    print(f"    kv_dir={kv_dir}")
    if not kv_dir.is_dir():
        print("    [FAIL] kv dir missing — dump_kv not in on_trigger, quota blocked, or path mismatch")
        ok = False
        pts: list[Path] = []
    else:
        pts = sorted(kv_dir.glob("*.pt"))
        print(f"    files={len(pts)}")
        if len(pts) < args.min_files:
            print(f"    [FAIL] expected >= {args.min_files} .pt files")
            ok = False

    print("[5] tensor health")
    layers: list[str] = []
    for path in pts:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"    [FAIL] load {path.name}: {exc}")
            ok = False
            continue
        if not isinstance(obj, dict) or "tensor" not in obj:
            print(f"    [FAIL] {path.name}: expected dict with 'tensor'")
            ok = False
            continue
        tensor = obj["tensor"]
        layer = str(obj.get("layer") or path.stem)
        layers.append(layer)
        if not hasattr(tensor, "reshape"):
            print(f"    [FAIL] {path.name}: bad tensor type {type(tensor)}")
            ok = False
            continue
        flat = tensor.detach().float().reshape(-1)
        n = int(flat.numel())
        n_nan = int(flat.isnan().sum().item()) if n else 0
        n_inf = int(flat.isinf().sum().item()) if n else 0
        flag = "OK" if n_nan == 0 and n_inf == 0 else "WARN"
        if n_nan or n_inf:
            # NaN/Inf is a finding, not necessarily a dump integrity failure.
            print(f"    [{flag}] {path.name} shape={tuple(tensor.shape)} nan={n_nan} inf={n_inf}")
        else:
            print(f"    [OK] {path.name} shape={tuple(tensor.shape)} finite={n}")
        dump_blocks = obj.get("block_ids")
        if block_ids and dump_blocks and list(dump_blocks) != list(block_ids):
            print(f"    [WARN] {path.name} block_ids != report block_ids")

    print("[6] layers")
    print(f"    unique_layers={len(set(layers))} files={len(pts)}")
    if pts and not layers:
        ok = False

    print()
    print("=" * 60)
    print("结论:", "PASS — report/KV 基本一致，可继续 ref 对比" if ok else "FAIL — 先修路径/配额/on_trigger")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
