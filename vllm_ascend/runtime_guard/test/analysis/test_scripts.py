# SPDX-License-Identifier: Apache-2.0
"""Offline analysis script UTs (synthetic report + wave/rank .pt layout)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm_ascend.runtime_guard.analysis.scripts._lib import (
    NativeLayerDump,
    compare_kv_dumps,
    gather_token_rows,
    report_dump_attempted,
    resolve_kv_dump_dir,
)
from vllm_ascend.runtime_guard.analysis.scripts import (
    correlate_incident,
    prepare_ref_inputs,
    summarize_reports,
    verify_request_kv,
)

RANK = "dp0_tp0_pp0_cp0"
BLOCK_SIZE = 4


def _write_report(path: Path, **overrides) -> dict:
    detail = {
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4, 5],
        "block_ids": [10, 11],
    }
    detail.update(overrides.pop("detail", {}))
    report = {
        "incident_type": "token_repeat",
        "req_id": "reqA",
        "dump_attempted": True,
        "dump_arm_wave": 2,
        "rank": RANK,
        "detail": detail,
    }
    report.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _write_layer_pt(path: Path, *, req_id: str = "reqA", layer: str = "L0") -> None:
    # [n_blocks=2, block_size=4, heads=1, dim=3] — enough for 5 tokens
    t = torch.arange(2 * BLOCK_SIZE * 1 * 3, dtype=torch.float32).reshape(2, BLOCK_SIZE, 1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "req_id": req_id,
            "block_ids": [10, 11],
            "layer": layer,
            "source": "test",
            "rank_tag": RANK,
            "num_kv_heads": 1,
            "tensor": t,
        },
        path,
    )


def _layout(report_root: Path) -> tuple[Path, Path, Path]:
    """report + kv under wave_2/rank; returns (report_path, kv_rank_dir, dump_dir)."""
    itype, req = "token_repeat", "reqA"
    dump_dir = report_root / "kv_cache" / itype / req / "wave_2"
    kv_rank = dump_dir / RANK
    report_path = report_root / itype / "report_test.json"
    report = _write_report(
        report_path,
        dump_dir=str(dump_dir),
        dump_arm_wave=2,
    )
    _write_layer_pt(kv_rank / "L0.pt")
    (dump_dir / "request_info.json").write_text(json.dumps({"req_id": req}), encoding="utf-8")
    return report_path, kv_rank, dump_dir


def test_report_dump_attempted_prefers_new_field():
    assert report_dump_attempted({"dump_attempted": True, "dump_armed": False}) is True
    assert report_dump_attempted({"dump_armed": True}) is True
    assert report_dump_attempted({}) is False


def test_resolve_kv_dump_dir_uses_dump_dir_and_rank(tmp_path: Path):
    rank = tmp_path / "wave_3" / RANK
    rank.mkdir(parents=True)
    (rank / "layer0.pt").write_bytes(b"x")
    report = {"dump_dir": str(tmp_path / "wave_3"), "rank": RANK}
    assert resolve_kv_dump_dir(report) == rank


def test_resolve_kv_dump_dir_scans_wave_rank(tmp_path: Path):
    report_dir = tmp_path / "report"
    rank = report_dir / "kv_cache" / "token_repeat" / "reqA" / "wave_2" / RANK
    rank.mkdir(parents=True)
    (rank / "L0.pt").write_bytes(b"x")
    report = {
        "incident_type": "token_repeat",
        "req_id": "reqA",
        "dump_arm_wave": 2,
        "rank": RANK,
    }
    assert resolve_kv_dump_dir(report, report_dir=report_dir) == rank


def test_gather_token_rows_block_layout():
    t = torch.arange(2 * BLOCK_SIZE * 1 * 3, dtype=torch.float32).reshape(2, BLOCK_SIZE, 1, 3)
    dump = NativeLayerDump(
        path=Path("x.pt"),
        layer="L0",
        req_id="r",
        block_ids=[10, 11],
        source="test",
        tensor=t,
    )
    rows = gather_token_rows(dump, n_tokens=5, block_size=BLOCK_SIZE, head=0)
    assert rows.shape == (5, 3)
    assert torch.allclose(rows[4], t[1, 0, 0].double())
    assert torch.allclose(rows[0], t[0, 0, 0].double())


def test_summarize_reports_dump_attempted_column(tmp_path: Path, capsys):
    report_root = tmp_path / "report"
    _write_report(report_root / "token_repeat" / "report_a.json", dump_attempted=True)
    assert summarize_reports.main(["--report-dir", str(report_root), "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "token_repeat" in out
    assert "Y" in out


def test_correlate_incident_finds_wave_rank_pt(tmp_path: Path, capsys):
    report_root = tmp_path / "report"
    _layout(report_root)
    rc = correlate_incident.main(
        ["--report-dir", str(report_root), "--req-id", "reqA", "--incident-type", "token_repeat"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dump_attempted=True" in out
    assert f"wave_2/{RANK}" in out.replace("\\", "/")
    assert "L0.pt" in out


def test_verify_request_kv_pass(tmp_path: Path):
    report_root = tmp_path / "report"
    report_path, _, _ = _layout(report_root)
    rc = verify_request_kv.main(
        ["--report", str(report_path), "--report-dir", str(report_root), "--block-size", str(BLOCK_SIZE)]
    )
    assert rc == 0


def test_verify_request_kv_fail_missing_kv(tmp_path: Path):
    report_root = tmp_path / "report"
    report_path = report_root / "token_repeat" / "report_x.json"
    _write_report(
        report_path,
        dump_dir=str(report_root / "kv_cache" / "token_repeat" / "reqA" / "wave_9"),
        dump_arm_wave=9,
    )
    rc = verify_request_kv.main(["--report", str(report_path), "--report-dir", str(report_root)])
    assert rc == 1


def test_compare_kv_dumps_identical_cos(tmp_path: Path):
    report_root = tmp_path / "report"
    report_path, kv_dir, _ = _layout(report_root)
    # second identical dump as "ref"
    ref_dir = tmp_path / "ref_rank"
    _write_layer_pt(ref_dir / "L0.pt")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = compare_kv_dumps(
        buggy_dir=kv_dir,
        ref_dir=ref_dir,
        detail=report["detail"],
        report=report,
        block_size=BLOCK_SIZE,
        head=0,
        cos_thresh=0.99,
    )
    assert analysis.first_bad_idx is None
    assert analysis.n_tokens == 5
    assert all(row.min_cos >= 0.99 for row in analysis.per_token)


def test_prepare_ref_inputs_writes_force_feed(tmp_path: Path):
    report_path = tmp_path / "report_test.json"
    _write_report(report_path)
    out = tmp_path / "ref_inputs.json"
    rc = prepare_ref_inputs.main(["--report", str(report_path), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["force_feed_token_ids"] == [1, 2, 3, 4, 5]
    assert payload["n_total"] == 5
