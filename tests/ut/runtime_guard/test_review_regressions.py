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

"""Regression UTs from the 2026-09-01 white-box review (task_spec/review_findings_20260901.md).

IDs map to review findings:
  V1  P0-4  matched_layers natural sort (first-divergence layer ordering)
  V2  P0-5  dumps_report_json tolerates non-JSON scalars (report never lost)
  V3  P0-1  soft-fail: hook exceptions must never propagate into the engine
  V4  P0-2  shipped example template loads + validates as-is
  V5  P0-3  bootstrap invalid content falls back to defaults (no crash)
  V6  P1-C1 sync_mode frozen across hot-reload (DP collective safety)
  V7  P1-A4 position_alignment violation labels
  V8  P1-B2 wave stamps discarded when requests are reaped (no leak)
  V9  P1-B3 ActionQueue: heavy job dropped (never inline on hot path); stop works with full queue
  V10 P1-C5 unknown detector sub-key rejected on reload (typo protection)
  V11 P1-A3 token_logprob window hot-resize rebuilds buffers
  V12 P1-A1 logits_finite: unattributable row alerts loudly, never misattributes
  V13 P0-2  JSONC comments + trailing commas parse
"""

from __future__ import annotations

import json
import os
import shutil
from collections import deque
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import vllm_ascend.runtime_config.config as cfg_mod
from vllm_ascend.runtime_config.config import RuntimeConfig
from vllm_ascend.runtime_guard.action.queue import ActionQueue
from vllm_ascend.runtime_guard.analysis.scripts._lib import NativeLayerDump, matched_layers
from vllm_ascend.runtime_guard.async_output import AscendAsyncOutput
from vllm_ascend.runtime_guard.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.runtime_guard.detector.manager import DetectorManager
from vllm_ascend.runtime_guard.detector.placement import (
    DetectorSpec,
    ExecScope,
    PlacementPlan,
    plan_placement,
)
from vllm_ascend.runtime_guard.detector.token_logprob import TokenLogprobDetector
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor
from vllm_ascend.runtime_guard.report import dumps_report_json
from vllm_ascend.runtime_guard.wave_tracker import WaveTracker


def _fake_dump(name: str) -> NativeLayerDump:
    return NativeLayerDump(
        path=Path("/dev/null"),
        layer=name,
        req_id="r",
        block_ids=[0],
        dump_all_blocks=False,
        source="ut",
        tensor=None,
    )


# ---------------------------------------------------------------- V1 (P0-4)


def test_v1_matched_layers_natural_sort():
    buggy = {f"layer_{i}": _fake_dump(f"layer_{i}") for i in (10, 2, 1, 33)}
    ref = dict(buggy)
    names = matched_layers(buggy, ref)
    assert names == ["layer_1", "layer_2", "layer_10", "layer_33"]


# ---------------------------------------------------------------- V2 (P0-5)


def test_v2_report_json_tolerates_non_json_scalars():
    out = dumps_report_json(
        {
            "incident_type": "token_repeat",
            "detail": {
                "np_int": np.int64(5),
                "torch_scalar": torch.tensor(7),
                "nan_float": float("nan"),
            },
        }
    )
    parsed = json.loads(out)
    assert parsed["incident_type"] == "token_repeat"
    assert "np_int" in parsed["detail"]
    assert "torch_scalar" in parsed["detail"]


# ---------------------------------------------------------------- V3 (P0-1)


def _bare_processor() -> RuntimeGuardProcessor:
    p = object.__new__(RuntimeGuardProcessor)
    p.detectors = MagicMock()
    p.detectors.check_after_sample = MagicMock(side_effect=RuntimeError("boom"))
    p.detectors.check_before_sample = MagicMock(side_effect=RuntimeError("boom"))
    p.wave_tracker = None
    p.runner = None
    p._handle_alert = MagicMock()
    p._reap_finished_requests = MagicMock()
    p._last_input_batch = None
    return p


def test_v3a_check_after_sample_soft_fail():
    p = _bare_processor()
    p.check_after_sample(sampled_token_ids=[1], logprobs_lists=None, req_ids=["r1"])


def test_v3b_check_before_sample_soft_fail():
    p = _bare_processor()
    p.check_before_sample(scheduler_output=None, logits=torch.randn(2, 8), positions=None)


def test_v3c_async_output_boundary_soft_fail():
    out = SimpleNamespace(sampled_token_ids=[1], logprobs=None, req_ids=["r1"])
    inner = MagicMock()
    inner.get_output.return_value = out
    guard = MagicMock()
    guard.check_after_sample.side_effect = RuntimeError("boom")
    runner = SimpleNamespace(runtime_guard=guard)
    assert AscendAsyncOutput(inner, runner).get_output() is out


def test_v3d_run_sample_phase_hook_failure_does_not_block_sampling():
    p = _bare_processor()

    sampled: list[int] = []

    def sample_fn():
        sampled.append(1)
        return SimpleNamespace(
            scheduler_output=None,
            input_batch=None,
            finished_req_ids=None,
            req_ids_output_copy=["r1"],
            valid_sampled_token_ids=[1],
            logprobs_lists=None,
            sampler_output=SimpleNamespace(sampled_token_ids=[1]),
        )

    def boom(*args, **kwargs):
        raise RuntimeError("hook boom")

    p.ensure_logprobs_for_detection = boom
    p.note_kv_block_writes = boom
    p.mark_finished = boom
    p.record_sample_waves = boom
    p.check_after_spec = boom
    p.check_after_sample = boom
    p.should_check_after_spec = lambda: False

    result, routed = p.run_sample_phase(
        sample_fn=sample_fn,
        speculative_config=None,
        need_accepted_tokens=False,
        use_async=False,
    )
    assert sampled == [1]
    assert result.req_ids_output_copy == ["r1"]


# ---------------------------------------------------------------- V4 (P0-2)


def _template_path() -> Path:
    return Path(cfg_mod.__file__).parent / "templates" / "runtime_config.example.jsonc"


def test_v4_example_template_loads_and_validates(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    shutil.copy(_template_path(), cfg_path)
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.detectors_enabled_in(cfg._data) is False
    assert cfg.dump_enabled() is False
    # The template must genuinely parse (JSONC) and validate: reload succeeds.
    assert cfg.reload(force=True) is True


# ---------------------------------------------------------------- V13 (P0-2)


def test_v13_jsonc_comments_and_trailing_commas(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(
        """{
  // enable repeat detection
  "detector": {
    "token_repeat": { "enabled": true, "window": 8, }, /* trailing comma above */
  },
}""",
        encoding="utf-8",
    )
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.detector_get("token_repeat", "enabled") is True
    assert cfg.detector_get("token_repeat", "window") == 8


# ---------------------------------------------------------------- V5 (P0-3)


def test_v5_bootstrap_invalid_content_falls_back_to_defaults(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(
        json.dumps({"detector": {"fatal_error": {"enabled": True}}}),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.detectors_enabled_in(cfg._data) is False


# ---------------------------------------------------------------- V6 (P1-C1)


def test_v6_sync_mode_frozen_across_reload(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(json.dumps({"sync_mode": "broadcast"}), encoding="utf-8")
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
        sync_mode="broadcast",
    )
    cfg_path.write_text(json.dumps({"sync_mode": "file"}), encoding="utf-8")
    os.utime(cfg_path, (time.time() + 10, time.time() + 10))
    assert cfg.reload(force=True) is True
    assert cfg.sync_mode == "broadcast"


# ---------------------------------------------------------------- V7 (P1-A4)


def test_v7_position_violation_labels():
    from vllm_ascend.runtime_guard.detector.position_alignment import classify_violation

    assert classify_violation(np.array([5, 6, 7]), 5) == ""
    assert classify_violation(np.array([6, 7, 8]), 5) == "wrong_start"
    assert classify_violation(np.array([6]), 5) == "wrong_start"
    assert classify_violation(np.array([5, 7, 9]), 5) == "non_consecutive"
    # gaps dominate a wrong start: [6, 8, 10] has holes AND offset
    assert classify_violation(np.array([6, 8, 10]), 5) == "non_consecutive"


# ---------------------------------------------------------------- V8 (P1-B2)


def test_v8a_wave_tracker_discard():
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    wt.record_sample_waves(["r1"])
    assert wt.take_sample_wave("r1") == 1
    wt.record_sample_waves(["r1"])
    wt.discard("r1")
    wt.discard("never-seen")  # no raise
    assert wt._sample_waves == {}


def test_v8b_reap_discards_wave_stamps():
    p = _bare_processor()
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    wt.record_sample_waves(["r1", "r2"])
    p.wave_tracker = wt
    p.runtime_config = MagicMock()
    p.runtime_config.log_print_output_on_finish.return_value = False
    store = MagicMock()
    store.list_reapable.return_value = ["r1", "r2"]
    with patch("vllm_ascend.runtime_guard.processor.RequestGuardStore") as store_cls:
        store_cls.get.return_value = store
        # Call the real method (the bare processor mocks the instance attr).
        RuntimeGuardProcessor._reap_finished_requests(p)
    assert wt._sample_waves == {}
    assert store.clear_many.called


# ---------------------------------------------------------------- V9 (P1-B3)


def test_v9a_full_queue_drops_heavy_job():
    q = ActionQueue(maxsize=1, name="ut-heavy-drop")
    q.start()
    gate = threading.Event()
    try:
        q.submit(lambda: gate.wait(2.0))
        time.sleep(0.05)
        ran: list[int] = []
        q.submit(lambda: ran.append(1), heavy=True)
        assert ran == []
    finally:
        gate.set()
        q.stop()


def test_v9b_stop_with_full_queue_still_stops_worker():
    q = ActionQueue(maxsize=2, name="ut-stop-full")
    q.start()
    gate = threading.Event()
    q.submit(lambda: gate.wait(3.0))
    q.submit(lambda: None)
    q.submit(lambda: None)  # pending queue now full
    t = q._thread
    assert t is not None
    q.stop()  # queue full: sentinel must still be delivered
    gate.set()  # let the in-flight job finish
    t.join(timeout=3.0)
    assert not t.is_alive()


# ---------------------------------------------------------------- V10 (P1-C5)


def test_v10_unknown_detector_key_rejected_on_reload(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(json.dumps({"detector": {"token_repeat": {"enabled": False}}}), encoding="utf-8")
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    # typo key "windw" must fail the reload loudly instead of silently defaulting
    cfg_path.write_text(
        json.dumps({"detector": {"token_repeat": {"enabled": True, "windw": 8}}}),
        encoding="utf-8",
    )
    os.utime(cfg_path, (time.time() + 10, time.time() + 10))
    assert cfg.reload(force=True) is False
    assert cfg.detector_get("token_repeat", "enabled") is False


# ---------------------------------------------------------------- V11 (P1-A3)


def test_v11_token_logprob_window_hot_resize_rebuilds_buffers():
    section = {"enabled": True, "window": 8, "stride": 4, "topk": 5}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
        disable_detector_unavailable=lambda *a, **k: None,
    )
    det = TokenLogprobDetector(runtime_config=rc, runner=None)
    assert det._window == 8
    det._buf["r1"] = deque([1, 2, 3, 4, 5], maxlen=8)
    det._since_check["r1"] = 0
    assert det._buf["r1"].maxlen == 8

    section["window"] = 4  # hot-reload
    det.refresh_from_config()
    assert det._window == 4
    assert det._buf["r1"].maxlen == 4


# ---------------------------------------------------------------- V12 (P1-A1)


def test_v12_logits_finite_unattributable_row_alerts_not_misattributes():
    section = {"enabled": True}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    input_batch = SimpleNamespace(req_ids=["a", "b"])
    runner = SimpleNamespace(input_batch=input_batch)  # no query_start_loc
    det = LogitsFiniteDetector(runtime_config=rc, runner=runner)

    logits = torch.randn(16, 8)
    logits[5, :] = float("nan")  # row 5 is a token of req "a" (2 reqs × 8 tokens)
    idx = torch.arange(16)  # chunked-prefill logits_indices, no qsl to map spans

    alerts = det.check_all(logits=logits, logits_indices=idx, input_batch=input_batch)
    assert len(alerts) == 1
    assert alerts[0].req_id is None  # never guessed
    assert alerts[0].detail.get("attribution") == "unresolved_row_to_request"


def test_v12b_logits_finite_decode_rows_still_attributed():
    section = {"enabled": True}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    input_batch = SimpleNamespace(req_ids=["a", "b"])
    runner = SimpleNamespace(input_batch=input_batch)
    det = LogitsFiniteDetector(runtime_config=rc, runner=runner)

    logits = torch.randn(2, 8)
    logits[1, :] = float("nan")
    alerts = det.check_all(logits=logits, logits_indices=None, input_batch=input_batch)
    assert len(alerts) == 1
    assert alerts[0].req_id == "b"


# ---------------------------------------------------------------- V14 (P1-B'4)


def _quota_rc(max_times: int, cooldown: float) -> SimpleNamespace:
    return SimpleNamespace(
        dump_max_times=lambda: max_times,
        dump_cooldown_seconds=lambda: cooldown,
    )


def test_v14_quota_try_consume_atomic_and_refund():
    from vllm_ascend.runtime_guard.quota import DumpQuota

    q = DumpQuota(_quota_rc(max_times=2, cooldown=0.0))
    assert q.try_consume() is True
    assert q.try_consume() is True
    assert q.try_consume() is False  # cap reached, atomically
    q.refund()
    assert q.try_consume() is True  # refund restored one unit


def test_v14b_quota_cooldown_block_must_not_burn():
    from vllm_ascend.runtime_guard.quota import DumpQuota

    q = DumpQuota(_quota_rc(max_times=5, cooldown=3600.0))
    assert q.try_consume() is True
    assert q.try_consume() is False  # inside cooldown
    assert q.total_count == 1  # blocked consume must not count
    q.refund()  # captured-nothing scenario
    assert q.total_count == 0


# ---------------------------------------------------------------- V15 (P1-B'2)


def test_v15_report_writer_dedupes_same_pair(tmp_path: Path):
    from vllm_ascend.runtime_guard.report import ReportWriter

    w = ReportWriter(tmp_path / "report")
    kw = dict(incident_type="token_repeat", req_id="r1", detail={"x": 1})
    assert w.write(**kw) is not None
    assert w.write(**kw) is None  # same (type, req) inside cooldown → skipped
    assert w.write(incident_type="token_repeat", req_id="r2", detail={"x": 1}) is not None


def test_v15b_report_writer_cooldown_expires(tmp_path: Path, monkeypatch):
    from vllm_ascend.runtime_guard.report import _SAME_PAIR_COOLDOWN_S, ReportWriter

    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    w = ReportWriter(tmp_path / "report")
    kw = dict(incident_type="token_repeat", req_id="r1", detail={"x": 1})
    assert w.write(**kw) is not None
    assert w.write(**kw) is None
    clock["t"] += _SAME_PAIR_COOLDOWN_S + 0.1
    assert w.write(**kw) is not None


# ------------------------------------------------- V16 (detector clear_finished)


def test_v16_token_logprob_clear_finished_no_cross_request_leak():
    section = {"enabled": True, "window": 8, "stride": 4, "topk": 5}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
        disable_detector_unavailable=lambda *a, **k: None,
    )
    det = TokenLogprobDetector(runtime_config=rc, runner=None)
    det._buf["r1"] = deque([1], maxlen=8)
    det._since_check["r1"] = 3
    det._checked.add("r1")
    det.clear_finished("r1")
    assert "r1" not in det._buf
    assert "r1" not in det._since_check
    assert "r1" not in det._checked


def test_v16b_spec_acceptance_clear_finished_history():
    from vllm_ascend.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector

    section = {"enabled": True, "window": 2}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    det = SpecAcceptanceDetector(runtime_config=rc, runner=None)
    det._history["r1"].append((1, 2, [1, 2], [1]))
    det.clear_finished("r1")
    assert "r1" not in det._history


# ------------------------------------------------- V17 (spec short batch)


def test_v17_spec_acceptance_short_batch_no_index_error():
    from vllm_ascend.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector

    section = {
        "enabled": True,
        "window": 2,
        "low_threshold": 0.3,
        "len_low_threshold": 1.4,
        "high_threshold": 0.96,
        "len_high_threshold": 2.8,
    }
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    runner = SimpleNamespace(
        tp_rank=1,
        speculative_config=SimpleNamespace(),
        input_batch=SimpleNamespace(req_ids=["a", "b"], num_draft_tokens_per_req=None),
        requests=None,
    )
    det = SpecAcceptanceDetector(runtime_config=rc, runner=runner)
    sampled = torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]])
    with patch(
        "vllm_ascend.runtime_guard.detector.spec_acceptance.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        alerts = det.check_all(sampled, [1])  # accepted shorter than req_ids
    assert isinstance(alerts, list)


# ---------------------------------------------------------------- payload (B'6)


def test_v18_dump_payload_carries_tp_rank_and_heads(tmp_path: Path):
    from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader

    cache = torch.randn(4, 8, 2, 16)  # [blocks, block_size, kv_heads, head_dim]
    runner = SimpleNamespace(kv_caches={"L0": cache}, tp_rank=3)
    reader = KvCacheReader(runner)
    snaps = reader.snapshot_request_blocks(
        req_id="r1",
        block_ids=[0, 2],
        out_dir=tmp_path / "kv",
        dump_all_blocks=False,
    )
    assert snaps[0].payload["tp_rank"] == 3
    assert snaps[0].payload["num_kv_heads"] == 2


# ------------------------------------------------- manual_dump drain (#5 / V19)


class _ConsumeRecorder:
    """runtime_config stand-in recording consume_manual_trigger calls."""

    def __init__(self, remaining: int = 2) -> None:
        self.remaining = remaining
        self.consume_calls = 0

    def consume_manual_trigger(self) -> bool:
        self.consume_calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            return True
        return False

    def manual_trigger_count(self) -> int:
        return self.remaining

    def dump_get(self, key, default=None):
        return default

    def dump_root(self):
        return "/tmp/ut-rg-report/kv_cache"

    @property
    def report_dir(self):
        return "/tmp/ut-rg-report"


def _dump_ctx(incident, runtime_config, kv_reader, quota) -> SimpleNamespace:
    return SimpleNamespace(
        incident=incident,
        runner=SimpleNamespace(tp_rank=0),
        runtime_config=runtime_config,
        report_writer=MagicMock(),
        kv_reader=kv_reader,
        quota=quota,
        rank_tag="TP0",
        tokenizer=None,
        detail={},
        action_overrides={},
        batch_rows=None,
        submit_async=lambda job: None,
    )


def _quota_stub(ok: bool = True):
    q = MagicMock()
    q.try_consume.return_value = ok
    return q


def test_v19a_manual_trigger_success_consumes_one_count():
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident
    from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE

    rc = _ConsumeRecorder(remaining=2)
    snap = SimpleNamespace(payload={})
    kv_reader = MagicMock()
    kv_reader.iter_request_snapshots.return_value = iter([snap])
    ctx = _dump_ctx(
        Incident(incident_type=MANUAL_TRIGGER_TYPE, req_id="r1", consume_quota=False),
        rc,
        kv_reader,
        _quota_stub(),
    )
    with patch(
        "vllm_ascend.runtime_guard.action.actions.is_action_leader_rank",
        return_value=True,
    ):
        prepared = DumpKvAction().prepare(ctx)
    assert prepared is None  # submit_async path: nothing handed to commit
    assert rc.consume_calls == 1
    assert rc.remaining == 1


def test_v19b_non_manual_incident_does_not_consume():
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident

    rc = _ConsumeRecorder(remaining=2)
    kv_reader = MagicMock()
    kv_reader.iter_request_snapshots.return_value = iter([SimpleNamespace(payload={})])
    ctx = _dump_ctx(
        Incident(incident_type="token_repeat", req_id="r1", consume_quota=True),
        rc,
        kv_reader,
        _quota_stub(),
    )
    with patch(
        "vllm_ascend.runtime_guard.action.actions.is_action_leader_rank",
        return_value=True,
    ):
        DumpKvAction().prepare(ctx)
    assert rc.consume_calls == 0


def test_v19c_no_snapshots_refunds_and_skips_consume():
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident
    from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE

    rc = _ConsumeRecorder(remaining=2)
    kv_reader = MagicMock()
    kv_reader.iter_request_snapshots.return_value = iter([])
    quota = _quota_stub()
    ctx = _dump_ctx(
        Incident(incident_type=MANUAL_TRIGGER_TYPE, req_id="r1", consume_quota=False),
        rc,
        kv_reader,
        quota,
    )
    with patch(
        "vllm_ascend.runtime_guard.action.actions.is_action_leader_rank",
        return_value=True,
    ):
        assert DumpKvAction().prepare(ctx) is None
    assert rc.consume_calls == 0
    quota.refund.assert_called_once_with(consume_quota=False)


# ------------------------------------------------ block write meta (V20)


def test_v20_block_meta_creation_and_last_write():
    from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker

    KvBlockMetaTracker.reset_for_tests()
    t = KvBlockMetaTracker.get()
    t.record_writes("req-A", [7, 8], wave=3)
    t.record_writes("req-B", [8], wave=5)

    assert t.last_writer_req_id(7) == "req-A"
    assert t.last_writer_req_id(8) == "req-B"
    detail = {e["block_id"]: e for e in t.blocks_detail([7, 8], include_wave=True, include_writer=True, include_creation=True)}
    # Creation stays with the first writer; last write tracks the latest.
    assert detail[8]["created_by_req_id"] == "req-A"
    assert detail[8]["last_writer_req_id"] == "req-B"
    # Wave counters stay internal (detector use); reports carry timestamps only.
    assert "last_write_wave" not in detail[8]
    assert "created_wave" not in detail[8]
    # Wall-clock stamps present and ordered (created <= last).
    assert detail[8]["created_at"] <= detail[8]["last_write_at"]
    assert detail[7]["last_write_at"] is not None
    # Violation preview unchanged by creation tracking.
    v = t.preview_write_checks("req-C", [8], 4)
    assert [x.violation for x in v] == ["wave_regression"]
    KvBlockMetaTracker.reset_for_tests()


# ------------------------------------------- default-path merge (V21)


def test_v21_default_path_file_keys_win_rest_default(tmp_path: Path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    cfg_file = tmp_path / "runtime" / "config" / "runtime_config.json"
    cfg_file.parent.mkdir(parents=True)
    # Only one key configured; everything else must fall back to defaults.
    cfg_file.write_text('{"detector": {"token_repeat": {"enabled": true}}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = cfg.RuntimeConfig(config_path=None)
    assert rc.detector_get("token_repeat", "enabled", False) is True
    # Unconfigured keys → defaults (token_repeat window default, substring off).
    assert rc.detector_get("token_repeat", "window", 0) == int(
        cfg._DEFAULTS["detector"]["token_repeat"]["window"]
    )
    assert rc.detector_get("output_substring", "enabled", True) is False


def test_v21b_default_path_missing_file_pure_defaults(tmp_path: Path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    rc = cfg.RuntimeConfig(config_path=None)
    assert rc.detector_get("token_repeat", "enabled", True) is False
    assert rc.detector_get("output_substring", "enabled", True) is False


# ------------------------------------------- slot last-write meta (V22)


def test_v22_slot_meta_last_write_token_and_writer():
    from vllm_ascend.runtime_guard import kv_block_meta as kbm
    from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker

    KvBlockMetaTracker.reset_for_tests()
    t = KvBlockMetaTracker.get()
    # req-A writes prompt positions 0..3 of block 9 (block_size 4).
    t.record_slot_writes("req-A", [(9 * 4 + i, 100 + i) for i in range(4)])
    # req-B later rewrites slot 37 (= block 9 offset 1) with a new token.
    t.record_slot_writes("req-B", [(37, 999)])

    detail = {e["slot"]: e for e in t.slots_detail([36, 37, 38, 40])}
    assert set(detail) == {36, 37, 38}  # slot 40 untouched → absent
    assert detail[36]["token_id"] == 100
    assert detail[36]["last_writer_req_id"] == "req-A"
    assert detail[36]["last_write_at"] is not None
    assert "last_write_wave" not in detail[36]
    assert detail[37]["token_id"] == 999
    assert detail[37]["last_writer_req_id"] == "req-B"
    assert detail[37]["last_write_at"] >= detail[36]["last_write_at"]
    # Unknown token (None) is preserved, not dropped.
    t.record_slot_writes("req-C", [(38, None)])
    detail2 = {e["slot"]: e for e in t.slots_detail([38])}
    assert detail2[38]["token_id"] is None
    assert detail2[38]["last_writer_req_id"] == "req-C"
    # Slots covered by blocks helper.
    assert kbm.slots_for_block_ids([9], block_size=4) == [36, 37, 38, 39]
    # Block size fallback.
    assert kbm.resolve_block_size(None) == 16
    KvBlockMetaTracker.reset_for_tests()


def test_v22b_slot_meta_cap_evicts_oldest_half(monkeypatch):
    from vllm_ascend.runtime_guard import kv_block_meta as kbm
    from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker

    monkeypatch.setattr(kbm, "_SLOT_META_CAP", 8)
    KvBlockMetaTracker.reset_for_tests()
    t = KvBlockMetaTracker.get()
    t.record_slot_writes("req-A", [(i, i) for i in range(12)])
    assert len(t._slot_meta) == 4  # 12 - 8//2 evicted, oldest first
    assert t.slots_detail([0, 1, 2, 3]) == []  # oldest gone
    assert {e["slot"] for e in t.slots_detail([8, 9, 10, 11])} == {8, 9, 10, 11}
    KvBlockMetaTracker.reset_for_tests()


def test_v22c_config_slot_last_write_flag(tmp_path: Path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    rc = cfg.RuntimeConfig(config_path=None)
    assert rc.report_slot_last_write() is False
    cfg_file = tmp_path / "runtime" / "config" / "runtime_config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text('{"report": {"slot_last_write": true}}', encoding="utf-8")
    rc2 = cfg.RuntimeConfig(config_path=None)
    assert rc2.report_slot_last_write() is True


# ------------------------------------------- configurable dump root (V23)


def test_v23_dump_root_default_json_and_startup_seed(tmp_path: Path, monkeypatch):
    import json

    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "runtime" / "config" / "runtime_config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text("{}", encoding="utf-8")

    # Default: derived <report_dir>/kv_cache.
    rc = cfg.RuntimeConfig(config_path=str(cfg_file))
    assert rc.dump_root() == rc.report_dir / "kv_cache"

    # JSON key wins and is hot-reload visible.
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    data["dump"] = {"dump_dir": str(tmp_path / "custom_dumps")}
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    assert rc.reload()
    assert rc.dump_root() == (tmp_path / "custom_dumps").resolve()

    # Startup arg seeds JSON dump_dir when the key is omitted.
    cfg_file.write_text("{}", encoding="utf-8")
    seeded = cfg.RuntimeConfig(config_path=str(cfg_file), dump_dir=str(tmp_path / "from_startup"))
    assert seeded.dump_root() == (tmp_path / "from_startup").resolve()
    # Explicit null in JSON clears the seed (user intent → derived default).
    cfg_file.write_text('{"dump": {"dump_dir": null}}', encoding="utf-8")
    cleared = cfg.RuntimeConfig(config_path=str(cfg_file), dump_dir=str(tmp_path / "from_startup"))
    assert cleared.dump_root() == cleared.report_dir / "kv_cache"


def test_v23b_report_writer_dump_dir_follows_provider(tmp_path: Path):
    from vllm_ascend.runtime_guard.report import ReportWriter

    root = tmp_path / "kv_root"

    def provider() -> str:
        return str(root)

    w = ReportWriter(tmp_path / "reports", dump_root_provider=provider)
    path = w.write(incident_type="block_kv", req_id="r-1", detail={"x": 1})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["dump_dir"] == str(root / "block_kv" / "r-1")
    w2 = ReportWriter(tmp_path / "reports")
    path2 = w2.write(incident_type="block_kv", req_id="r-1", detail={"x": 1})
    assert json.loads(path2.read_text(encoding="utf-8"))["dump_dir"] == str(
        tmp_path / "reports" / "kv_cache" / "block_kv" / "r-1"
    )


# ------------------------------------------- slot-token consistency (V24)


def test_v24_slot_consistency_import_ok_mismatch_fires_first_mode_dedupe():
    from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker
    from vllm_ascend.runtime_guard.detector.slot_consistency import SlotConsistencyDetector

    KvBlockMetaTracker.reset_for_tests()
    t = KvBlockMetaTracker.get()
    # req-A wrote block 9 positions 0..3 (block_size 4) with tokens 100..103.
    t.record_slot_writes("req-A", [(36 + i, 100 + i) for i in range(4)])

    det = SlotConsistencyDetector()
    det._enabled = True

    kw = dict(req_idx=0, block_ids=[9], block_size=4, computed_before=4, scheduled=0)

    # Prefix-cache import: same tokens, different writer → consistent, no alert.
    assert det.check_slots(req_id="req-B", seq=[100, 101, 102, 103], **kw) == []

    # Same tokens + fresh own write appended (decode position) → consistent.
    t.record_slot_writes("req-B", [(39, 103)])
    assert (
        det.check_slots(req_id="req-B2", seq=[100, 101, 102, 103], **kw) == []
    )

    # req-C's sequence differs at position 1 → mismatch with req-A's residue.
    alerts = det.check_slots(req_id="req-C", seq=[100, 999, 102, 103], **kw)
    assert len(alerts) == 1
    d = alerts[0].detail
    assert d["num_mismatches"] == 1
    m = d["mismatches"][0]
    assert m["pos"] == 1
    assert m["slot"] == 37
    assert m["expected_token"] == 999
    assert m["actual_token"] == 101
    assert m["last_writer_req_id"] == "req-A"
    assert m["last_write_at"] is not None
    assert d["checked_positions"] == 4
    assert d["unverified_slots"] == 0

    # first mode: same req is not rechecked on later steps.
    assert det.check_slots(req_id="req-C", seq=[100, 777, 102, 103], **kw) == []
    assert det.check_slots(req_id="req-C2", seq=[100, 777, 102, 103], **kw) != []

    # Untouched positions (no meta) count as unverified, never alert.
    det2 = SlotConsistencyDetector()
    det2._enabled = True
    alerts2 = det2.check_slots(req_id="req-D", seq=[100, 101, 102, 103], **kw)
    assert alerts2 == []  # tracker holds A's meta → consistent; use fresh block instead
    alerts3 = det2.check_slots(
        req_id="req-E", seq=[1, 2, 3, 4], block_ids=[77], block_size=4, computed_before=4, scheduled=0, req_idx=0
    )
    assert alerts3 == []  # block 77 has no meta → all unverified, no false positive
    KvBlockMetaTracker.reset_for_tests()


def test_v24b_slot_consistency_step_mode_and_clear_finished():
    from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker
    from vllm_ascend.runtime_guard.detector.slot_consistency import SlotConsistencyDetector

    KvBlockMetaTracker.reset_for_tests()
    t = KvBlockMetaTracker.get()
    t.record_slot_writes("req-A", [(0, 10), (1, 11)])

    det = SlotConsistencyDetector()
    det._enabled = True
    det._mode = "step"
    kw = dict(req_idx=0, block_ids=[0], block_size=4, computed_before=2, scheduled=0)

    # Step 1: consistent.
    assert det.check_slots(req_id="req-B", seq=[10, 11], **kw) == []
    # Another request overwrites slot 1 with its own token (contamination).
    t.record_slot_writes("req-X", [(1, 99)])
    # Step 2: step mode rechecks the whole prefix and catches it.
    alerts = det.check_slots(req_id="req-B", seq=[10, 11], **kw)
    assert len(alerts) == 1
    assert alerts[0].detail["mismatches"][0]["last_writer_req_id"] == "req-X"
    # clear_finished forgets the req (re-detect after finish/retry).
    det.clear_finished("req-B")
    assert det.check_slots(req_id="req-B", seq=[10, 11], **kw) != []
    KvBlockMetaTracker.reset_for_tests()


def test_v24c_slot_consistency_config_section_defaults():
    import vllm_ascend.runtime_config.config as cfg

    assert cfg._DEFAULTS["detector"]["slot_consistency"] == {
        "enabled": False,
        "exec_scope": "auto",
        "mode": "first",
    }
    # Unknown mode value keeps the previous setting (validated in _apply_detector_values).
    from vllm_ascend.runtime_guard.detector.slot_consistency import SlotConsistencyDetector

    det = SlotConsistencyDetector()
    det._apply_detector_values(lambda key, default=None: "bogus" if key == "mode" else default)
    assert det._mode == "first"
    det._apply_detector_values(lambda key, default=None: "step" if key == "mode" else default)
    assert det._mode == "step"


# ------------------------------------------- detector placement / ExecScope (V25)


def _spec(name, *, scope=ExecScope.ANY, rank_local=False, cost=1.0, enabled=True):
    return DetectorSpec(
        incident_type=name,
        exec_scope=scope,
        rank_local_data=rank_local,
        cost=cost,
        enabled=enabled,
    )


def test_v25_placement_lpt_balance_deterministic_all_scope():
    # Enabled ANY detectors LPT-balance across ranks; heaviest lands first.
    specs = [
        _spec("logits_finite", cost=2.0),
        _spec("position_alignment", cost=1.0),
        _spec("token_repeat", cost=0.5),
        _spec("token_logprob", cost=1.5),
    ]
    plan = plan_placement(specs, tp_size=2, rank_local_world=False)
    # Total 5.0 → best split 2.5/2.5 is impossible with these weights; LPT
    # gives rank0={logits_finite(2)+token_repeat(0.5)} rank1={token_logprob(1.5)+position(1)}.
    assert plan.rank_of("logits_finite") != plan.rank_of("token_logprob")
    loads = [0.0, 0.0]
    for s in specs:
        loads[plan.rank_of(s.incident_type)] += s.cost
    assert abs(loads[0] - loads[1]) <= 2.0  # balanced within one heavy item
    # Determinism: same inputs → same plan.
    assert plan_placement(specs, tp_size=2, rank_local_world=False) == plan
    # Disabled detectors are never placed; LEADER goes to rank 0.
    p2 = plan_placement(
        [_spec("x", scope=ExecScope.LEADER, enabled=False), _spec("y", scope=ExecScope.LEADER)],
        tp_size=4,
        rank_local_world=False,
    )
    assert p2.rank_of("x") is None and p2.rank_of("y") == 0
    # Rank-local data under CP/DP → ALL (every rank); pure TP → single rank.
    p3 = plan_placement([_spec("slot_consistency", rank_local=True)], tp_size=2, rank_local_world=True)
    assert p3.runs_here("slot_consistency", 0) and p3.runs_here("slot_consistency", 1)
    p4 = plan_placement([_spec("slot_consistency", rank_local=True)], tp_size=2, rank_local_world=False)
    assert p4.all_ranks == frozenset()
    # EXTERNAL is never scheduled.
    p5 = plan_placement([_spec("future_offload", scope=ExecScope.EXTERNAL)], tp_size=2, rank_local_world=False)
    assert p5.rank_of("future_offload") is None and not p5.runs_here("future_offload", 0)


def test_v25_placement_manual_and_pin():
    specs = [_spec("logits_finite", cost=2.0), _spec("token_repeat", cost=0.5)]
    # Manual mode: map wins for ANY detectors; unknown detectors in the map
    # are ignored; out-of-range ranks fall back to auto.
    plan = plan_placement(
        specs,
        tp_size=2,
        rank_local_world=False,
        mode="manual",
        manual={"logits_finite": 1, "no_such": 3, "token_repeat": 9},
    )
    assert plan.rank_of("logits_finite") == 1
    assert plan.rank_of("token_repeat") in (0, 1)  # invalid 9 → auto
    # pin keeps previous assignment for still-enabled detectors.
    prev = PlacementPlan(assignment={"logits_finite": 1}, all_ranks=frozenset())
    pinned = plan_placement(specs, tp_size=2, rank_local_world=False, pin=True, previous=prev)
    assert pinned.rank_of("logits_finite") == 1


def test_v25b_manager_gates_stages_by_placement(tmp_path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    rc = cfg.RuntimeConfig(config_path=None)
    rc._data["detector"]["logits_finite"]["enabled"] = True
    rc._data["detector_placement"]["mode"] = "manual"
    rc._data["detector_placement"]["manual"] = {"logits_finite": 1}

    def make_manager(tp_rank: int) -> DetectorManager:
        runner = SimpleNamespace(tp_rank=tp_rank, tp_size=2, input_batch=SimpleNamespace(req_ids=[]))
        return DetectorManager(runtime_config=rc, runner=runner)

    rank0 = make_manager(0)
    rank1 = make_manager(1)
    assert rank0._here(rank0._logits_finite_det) is False
    assert rank1._here(rank1._logits_finite_det) is True
    # rank1 also runs it (topology fell back to runner.tp_size=2).

    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        return []

    rank0._logits_finite_det.check_all = _boom
    rank1._logits_finite_det.check_all = _boom
    out0 = rank0.check_before_sample(
        scheduler_output=None, logits=None, positions=None, input_batch=None
    )
    out1 = rank1.check_before_sample(
        scheduler_output=None, logits=None, positions=None, input_batch=None
    )
    assert out0 == [] and calls["n"] == 1  # only rank1 invoked the detector


def test_v25c_config_placement_schema(tmp_path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "runtime" / "config" / "runtime_config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text("{}", encoding="utf-8")
    rc = cfg.RuntimeConfig(config_path=str(cfg_file))
    assert rc.detector_placement_mode() == "auto"
    assert rc.detector_placement_manual() == {}
    assert rc.detector_placement_pin() is False

    # Bad exec_scope rejected on reload (old config kept).
    good = json.loads(cfg_file.read_text())
    bad = dict(good, detector=dict(logits_finite={"enabled": True, "exec_scope": "everywhere"}))
    cfg_file.write_text(json.dumps(bad), encoding="utf-8")
    assert rc.reload() is False
    assert rc.detector_get("logits_finite", "exec_scope", "auto") == "auto"

    # Valid scope accepted.
    ok = dict(good, detector=dict(logits_finite={"enabled": True, "exec_scope": "any"}))
    cfg_file.write_text(json.dumps(ok), encoding="utf-8")
    assert rc.reload() is True
    assert rc.detector_get("logits_finite", "exec_scope") == "any"

    # placement schema: bad mode / unknown detector / bad rank / bad pin.
    for payload in (
        {"detector_placement": {"mode": "roundrobin"}},
        {"detector_placement": {"manual": {"no_such_detector": 0}}},
        {"detector_placement": {"manual": {"token_repeat": True}}},
        {"detector_placement": {"pin": "yes"}},
    ):
        cfg_file.write_text(json.dumps(dict(good, **payload)), encoding="utf-8")
        assert rc.reload() is False, payload
    fine = dict(good, detector_placement={"mode": "manual", "manual": {"token_repeat": 1}, "pin": True})
    cfg_file.write_text(json.dumps(fine), encoding="utf-8")
    assert rc.reload() is True
    assert rc.detector_placement_mode() == "manual"
    assert rc.detector_placement_manual() == {"token_repeat": 1}
    assert rc.detector_placement_pin() is True


def test_v25d_manager_replan_on_config_change(tmp_path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    rc = cfg.RuntimeConfig(config_path=None)
    runner = SimpleNamespace(tp_rank=0, tp_size=2, input_batch=SimpleNamespace(req_ids=[]))
    mgr = DetectorManager(runtime_config=rc, runner=runner)
    assert mgr._plan.rank_of("output_substring") is None  # nothing enabled

    rc._data["detector"]["output_substring"]["enabled"] = True
    mgr.apply_runtime_config()
    assert mgr._plan.rank_of("output_substring") == 0  # auto LPT → rank0

    rc._data["detector_placement"]["mode"] = "manual"
    rc._data["detector_placement"]["manual"] = {"output_substring": 1}
    mgr.apply_runtime_config()
    assert mgr._plan.rank_of("output_substring") == 1


def test_v25e_rank_gate_allows_non_tp0_detection(monkeypatch):
    import vllm_ascend.runtime_guard.rank_gate as rank_gate

    monkeypatch.setattr(
        rank_gate, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    # Async scheduling no longer pins detection to TP0 (placement decides).
    runner = SimpleNamespace(use_async_scheduling=True, tp_rank=1)
    assert rank_gate.anomaly_check_rank_skip_reason(runner) is None
    # Non-last PP rank still gated.
    monkeypatch.setattr(
        rank_gate, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    assert rank_gate.anomaly_check_rank_skip_reason(runner) == "not last PP rank"
