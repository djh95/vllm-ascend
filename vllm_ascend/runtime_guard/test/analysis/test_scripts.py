# SPDX-License-Identifier: Apache-2.0
"""S7: offline analysis helpers vs config dump layout."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm_ascend.runtime_guard.analysis.scripts._lib import (
    NativeLayerDump,
    gather_token_rows,
    report_dump_attempted,
    resolve_kv_dump_dir,
)


def test_report_dump_attempted_prefers_new_field():
    assert report_dump_attempted({"dump_attempted": True, "dump_armed": False}) is True
    assert report_dump_attempted({"dump_armed": True}) is True
    assert report_dump_attempted({}) is False


def test_resolve_kv_dump_dir_uses_dump_dir_and_rank(tmp_path: Path):
    rank = tmp_path / "wave_3" / "dp0_tp0_pp0_cp0"
    rank.mkdir(parents=True)
    (rank / "layer0.pt").write_bytes(b"x")
    report = {
        "dump_dir": str(tmp_path / "wave_3"),
        "rank": "dp0_tp0_pp0_cp0",
    }
    assert resolve_kv_dump_dir(report) == rank


def test_resolve_kv_dump_dir_scans_wave_rank(tmp_path: Path):
    report_dir = tmp_path / "report"
    rank = report_dir / "kv_cache" / "token_repeat" / "reqA" / "wave_2" / "dp0_tp0_pp0_cp0"
    rank.mkdir(parents=True)
    (rank / "L0.pt").write_bytes(b"x")
    report = {
        "incident_type": "token_repeat",
        "req_id": "reqA",
        "dump_arm_wave": 2,
        "rank": "dp0_tp0_pp0_cp0",
    }
    assert resolve_kv_dump_dir(report, report_dir=report_dir) == rank


def test_gather_token_rows_block_layout():
    # [n_blocks=2, block_size=4, heads=1, dim=3]
    t = torch.arange(2 * 4 * 1 * 3, dtype=torch.float32).reshape(2, 4, 1, 3)
    dump = NativeLayerDump(
        path=Path("x.pt"),
        layer="L0",
        req_id="r",
        block_ids=[10, 11],
        source="test",
        tensor=t,
    )
    rows = gather_token_rows(dump, n_tokens=5, block_size=4, head=0)
    assert rows.shape == (5, 3)
    # token 4 → block1 slot0
    assert torch.allclose(rows[4], t[1, 0, 0].double())
    assert torch.allclose(rows[0], t[0, 0, 0].double())
