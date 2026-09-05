#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UTs for runtime_guard post-incident analysis scripts (native dump_kv)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "runtime_guard"
    / "analysis"
    / "scripts"
)


def _load(name: str):
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    # Ensure shared _lib is importable for file-based loads.
    if "_lib" not in sys.modules:
        lib_spec = importlib.util.spec_from_file_location("_lib", _SCRIPTS / "_lib.py")
        assert lib_spec is not None and lib_spec.loader is not None
        lib_mod = importlib.util.module_from_spec(lib_spec)
        sys.modules["_lib"] = lib_mod
        lib_spec.loader.exec_module(lib_mod)
    path = _SCRIPTS / f"{name}.py"
    mod_name = f"_rg_analysis_{name}"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detail = {
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": [4, 5],
        "block_ids": [10, 11],
    }
    if isinstance(fields.get("detail"), dict):
        detail.update(fields.pop("detail"))
    base = {
        "incident_type": "token_repeat",
        "req_id": "req-1",
        "dump_armed": True,
        "detail": detail,
    }
    base.update(fields)
    path.write_text(json.dumps(base), encoding="utf-8")


def _write_layer_pt(dir_path: Path, layer: str, tensor: torch.Tensor, req_id: str = "req-1") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "req_id": req_id,
            "block_ids": [10, 11],
            "dump_all_blocks": False,
            "layer": layer,
            "source": "kv_caches",
            "tensor": tensor,
        },
        dir_path / f"{req_id}_{layer}_req.pt",
    )


def test_summarize_reports_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    summarize = _load("summarize_reports")
    report = tmp_path / "token_repeat" / "report_t1_req-1_pid1.json"
    _write_report(report)
    rc = summarize.main(["--report-dir", str(tmp_path), "--limit", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "token_repeat" in out
    assert "req-1" in out


def test_summarize_empty_dir(tmp_path: Path):
    summarize = _load("summarize_reports")
    assert summarize.main(["--report-dir", str(tmp_path)]) == 1


def test_correlate_incident_finds_kv(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    correlate = _load("correlate_incident")
    report = tmp_path / "token_repeat" / "report_t1_req-1_pid1.json"
    _write_report(report)
    kv_dir = tmp_path / "kv_cache" / "token_repeat" / "req-1"
    kv_dir.mkdir(parents=True)
    (kv_dir / "layer0_req.pt").write_bytes(b"")
    rc = correlate.main(["--report-dir", str(tmp_path), "--req-id", "req-1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reports=1" in out
    assert "verify_request_kv" in out


def test_correlate_miss(tmp_path: Path):
    correlate = _load("correlate_incident")
    assert correlate.main(["--report-dir", str(tmp_path), "--req-id", "missing"]) == 1


def test_verify_request_kv_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    verify = _load("verify_request_kv")
    report = tmp_path / "token_repeat" / "report_t1_req-1_pid1.json"
    _write_report(report)
    kv_dir = tmp_path / "kv_cache" / "token_repeat" / "req-1"
    _write_layer_pt(kv_dir, "layers.0", torch.zeros(256, 4))
    rc = verify.main(["--report", str(report), "--report-dir", str(tmp_path), "--block-size", "128"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


def test_verify_request_kv_missing_kv(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    verify = _load("verify_request_kv")
    report = tmp_path / "token_repeat" / "report_t1_req-1_pid1.json"
    _write_report(report)
    rc = verify.main(["--report", str(report), "--report-dir", str(tmp_path)])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out


def test_inspect_kv_dump(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    inspect = _load("inspect_kv_dump")
    path = tmp_path / "one.pt"
    torch.save(
        {
            "req_id": "req-1",
            "layer": "L0",
            "block_ids": [1],
            "dump_all_blocks": False,
            "source": "key_cache",
            "tensor": torch.tensor([1.0, 2.0, float("nan")]),
        },
        path,
    )
    rc = inspect.main(["--path", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nan=1" in out


def test_prepare_ref_inputs(tmp_path: Path):
    prep = _load("prepare_ref_inputs")
    report = tmp_path / "report.json"
    _write_report(report)
    out = tmp_path / "ref_inputs.json"
    assert prep.main(["--report", str(report), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["force_feed_token_ids"] == [1, 2, 3, 4, 5]


def test_request_from_report_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    req = _load("request_from_report")
    report = tmp_path / "report.json"
    _write_report(report)
    assert req.build_feed_ids(
        {"prompt_token_ids": [1, 2, 3], "output_token_ids": [4, 5]}, "full"
    ) == [1, 2, 3, 4, 5]
    assert req.build_feed_ids(
        {"prompt_token_ids": [1, 2, 3], "output_token_ids": [4, 5]}, "history"
    ) == [1, 2, 3, 4]
    rc = req.main(
        [
            "--report",
            str(report),
            "--dry-run",
            "--print-curl",
            "--model",
            "test-model",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # default --feed=history → prompt + output[:-1]
    assert "n_feed=4" in out
    assert "feed=history" in out
    assert "curl" in out


def test_argmin_layer_cos_prefers_nan():
    import importlib.util

    mod_name = "_lib_nan_ut"
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / "_lib.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    name, cos = mod.argmin_layer_cos({"L0": 1.0, "L1": 1.0, "L2": float("nan")})
    assert name == "L2"
    assert cos != cos  # NaN
    name2, cos2 = mod.argmin_layer_cos({"L0": 0.5, "L1": 0.9})
    assert name2 == "L0"
    assert cos2 == 0.5


def test_compare_kv_similarity_two_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    compare = _load("compare_kv_similarity")
    detail = {"prompt_token_ids": [1, 2], "output_token_ids": [3, 4], "block_ids": [0]}
    buggy_report = tmp_path / "buggy_report.json"
    ref_report = tmp_path / "ref_report.json"
    _write_report(buggy_report, req_id="bad-req", detail=detail)
    _write_report(ref_report, req_id="ref-req", incident_type="manual_trigger", detail=detail)
    bad = tmp_path / "bad"
    ref = tmp_path / "ref"
    base = torch.ones(8, 4)
    _write_layer_pt(bad, "L0", base.clone(), req_id="bad-req")
    _write_layer_pt(bad, "L1", base.clone(), req_id="bad-req")
    ref0 = base.clone()
    ref0[2] = 0.0
    _write_layer_pt(ref, "L0", ref0, req_id="ref-req")
    _write_layer_pt(ref, "L1", base.clone(), req_id="ref-req")

    json_out = tmp_path / "result.json"
    rc = compare.main(
        [
            "--buggy-dir",
            str(bad),
            "--ref-dir",
            str(ref),
            "--buggy-report",
            str(buggy_report),
            "--ref-report",
            str(ref_report),
            "--cos-thresh",
            "0.99",
            "--json-out",
            str(json_out),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "TABLE 1" in out
    assert "TABLE 2" in out
    payload = json.loads(json_out.read_text())
    assert payload["first_bad"]["token_idx"] == 2


def test_locate_first_divergence_two_tables(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    locate = _load("locate_first_divergence")
    report = tmp_path / "report.json"
    # N=4 tokens
    _write_report(
        report,
        detail={
            "prompt_token_ids": [1, 2],
            "output_token_ids": [3, 4],
            "block_ids": [0],
        },
    )
    bad = tmp_path / "bad"
    ref = tmp_path / "ref"
    # [N_slots, D] with block_size unused for concat dump; length >= 4
    base = torch.ones(8, 4)
    _write_layer_pt(bad, "L0", base.clone())
    _write_layer_pt(bad, "L1", base.clone())
    ref0 = base.clone()
    ref0[2] = 0.0  # diverge at token 2
    _write_layer_pt(ref, "L0", ref0)
    _write_layer_pt(ref, "L1", base.clone())

    rc = locate.main(
        [
            "--buggy-dir",
            str(bad),
            "--ref-dir",
            str(ref),
            "--report",
            str(report),
            "--block-size",
            "128",
            "--cos-thresh",
            "0.99",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "TABLE 1" in out
    assert "TABLE 2" in out
    assert "第一坏点" in out or "token 2" in out


def test_compare_per_layer(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    compare = _load("compare_per_layer")
    report = tmp_path / "report.json"
    _write_report(
        report,
        detail={"prompt_token_ids": [1, 2], "output_token_ids": [3], "block_ids": [0]},
    )
    bad = tmp_path / "bad"
    ref = tmp_path / "ref"
    t = torch.ones(8, 2)
    _write_layer_pt(bad, "L0", t.clone())
    _write_layer_pt(ref, "L0", t.clone())
    rc = compare.main(
        ["--buggy-dir", str(bad), "--ref-dir", str(ref), "--report", str(report), "--num-tokens", "3"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "L0" in out
    assert "min_cos" in out


def test_locate_missing_dir_exits_nonzero(tmp_path: Path):
    locate = _load("locate_first_divergence")
    rc = locate.main(
        ["--buggy-dir", str(tmp_path / "x"), "--ref-dir", str(tmp_path / "y"), "--num-tokens", "2"]
    )
    assert rc == 1


@pytest.mark.parametrize(
    "name",
    [
        "summarize_reports",
        "correlate_incident",
        "verify_request_kv",
        "inspect_kv_dump",
        "prepare_ref_inputs",
        "request_from_report",
        "compare_kv_similarity",
        "locate_first_divergence",
        "compare_per_layer",
    ],
)
def test_script_help_exits_zero(name: str):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / f"{name}.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout
