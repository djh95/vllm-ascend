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

"""P0 microbench: hot-path overhead when feature is off / hot-reload-only.

These are CPU-side isolation checks (not NPU e2e). They encode the claim that
``refresh_config`` with hot-reload disabled is near-free, and that enabling
reload without detectors does not explode per-call cost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from vllm_ascend.runtime_config.config import RuntimeConfig
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor, SamplePhaseResult


def _cfg(tmp_path: Path, *, reload: float) -> RuntimeConfig:
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    return RuntimeConfig(
        config_path=path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=reload,
        sync_mode="file",
    )


def _bench(fn, n: int = 2000) -> float:
    # Warmup
    for _ in range(50):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6  # µs / call


def test_refresh_config_hot_reload_off_is_cheap(tmp_path: Path):
    """I2: reload=0 → refresh_config must stay microsecond-scale on CPU."""
    RuntimeGuardProcessor.reset_for_tests()
    cfg = _cfg(tmp_path, reload=0.0)
    runner = MagicMock()
    runner.tp_rank = 0

    def _init(self, r):
        self.runner = r
        self.runtime_config = cfg
        self.manual_triggers = MagicMock()
        self.manual_triggers.consume_once.return_value = None
        self.action_executor = MagicMock()
        self.detectors = MagicMock()
        self.report_writer = MagicMock()
        self.kv_reader = MagicMock()
        self.quota = MagicMock()

    with patch.object(RuntimeGuardProcessor, "_init_from_runner", _init):
        proc = RuntimeGuardProcessor.bind(runner)

    us = _bench(lambda: proc.refresh_config(allow_arm=True))
    # Generous bound: even CI noise should stay << 1ms when reload is off.
    assert us < 500.0, f"refresh_config (reload=0) too slow: {us:.1f} µs"

    RuntimeGuardProcessor.reset_for_tests()


def test_refresh_config_reload_on_detectors_off_bounded(tmp_path: Path):
    """I3: reload>0 but detectors/dump off — still far below sampling latency."""
    RuntimeGuardProcessor.reset_for_tests()
    cfg = _cfg(tmp_path, reload=0.05)
    runner = MagicMock()
    runner.tp_rank = 0

    def _init(self, r):
        self.runner = r
        self.runtime_config = cfg
        self.manual_triggers = MagicMock()
        self.manual_triggers.consume_once.return_value = None
        self.action_executor = MagicMock()
        self.action_executor.apply_runtime_config = MagicMock()
        self.detectors = MagicMock()
        self.detectors.apply_runtime_config = MagicMock()
        self.report_writer = MagicMock()
        self.kv_reader = MagicMock()
        self.quota = MagicMock()

    with (
        patch.object(RuntimeGuardProcessor, "_init_from_runner", _init),
        patch(
            "vllm_ascend.runtime_guard.processor.InputFilterManager"
        ) as ifm,
    ):
        ifm.get.return_value = MagicMock()
        proc = RuntimeGuardProcessor.bind(runner)
        us = _bench(lambda: proc.refresh_config(allow_arm=True), n=500)

    # File stat + optional JSON read: allow up to a few ms worst case on CI.
    assert us < 5000.0, f"refresh_config (reload on, idle) too slow: {us:.1f} µs"
    RuntimeGuardProcessor.reset_for_tests()


def test_isolation_refresh_vs_pure_noop(tmp_path: Path):
    """A0': bound cost of refresh_config(reload=0) vs empty Python call.

    This is the strongest *in-process* proxy for 'feature installed but off'.
    It does **not** replace live NPU E1 (build without bind vs with bind).
    """
    RuntimeGuardProcessor.reset_for_tests()
    cfg = _cfg(tmp_path, reload=0.0)
    runner = MagicMock()
    runner.tp_rank = 0

    def _init(self, r):
        self.runner = r
        self.runtime_config = cfg
        self.manual_triggers = MagicMock()
        self.manual_triggers.consume_once.return_value = None
        self.action_executor = MagicMock()
        self.detectors = MagicMock()
        self.report_writer = MagicMock()
        self.kv_reader = MagicMock()
        self.quota = MagicMock()

    with patch.object(RuntimeGuardProcessor, "_init_from_runner", _init):
        proc = RuntimeGuardProcessor.bind(runner)

    def noop():
        return None

    us_noop = _bench(noop, n=5000)
    us_rg = _bench(lambda: proc.refresh_config(allow_arm=True), n=2000)
    # Off-path should stay within a small multiple of a pure Python call on CI.
    # Absolute cap still applies (see sibling test).
    assert us_rg < 500.0, f"refresh_config too slow: {us_rg:.1f} µs"
    assert us_rg < us_noop + 400.0, (
        f"reload=0 path not near-noop: rg={us_rg:.1f}µs noop={us_noop:.1f}µs"
    )
    RuntimeGuardProcessor.reset_for_tests()


def test_runtime_config_sync_noop_when_reload_disabled(tmp_path: Path):
    cfg = _cfg(tmp_path, reload=0.0)
    assert cfg.hot_reload_enabled is False
    # Must not raise; returns False without polling forever.
    assert cfg.sync_runtime_config() is False


def test_needs_sample_phase_hooks_and_cumulative_io_flags(tmp_path: Path):
    cfg = _cfg(tmp_path, reload=0.0)
    assert cfg.needs_sample_phase_hooks() is False
    assert cfg.needs_cumulative_io() is False

    cfg._data["log"]["print_output_on_finish"] = True
    assert cfg.needs_sample_phase_hooks() is True
    assert cfg.needs_cumulative_io() is True

    cfg._data["log"]["print_output_on_finish"] = False
    cfg._data["detector"]["token_logprob"]["enabled"] = True
    assert cfg.needs_sample_phase_hooks() is True
    # token_logprob alone does not need cumulative IO without sensitive reports
    assert cfg.needs_cumulative_io() is False

    cfg._data["report"]["save_sensitive_info"] = True
    assert cfg.needs_cumulative_io() is True


def test_run_sample_phase_idle_skips_hooks(tmp_path: Path):
    """A-path: detectors/print/block-meta off → sample_fn only."""
    RuntimeGuardProcessor.reset_for_tests()
    cfg = _cfg(tmp_path, reload=0.0)
    runner = MagicMock()
    runner.tp_rank = 0

    def _init(self, r):
        self.runner = r
        self.runtime_config = cfg
        self.manual_triggers = MagicMock()
        self.action_executor = MagicMock()
        self.detectors = MagicMock()
        self.report_writer = MagicMock()
        self.kv_reader = MagicMock()
        self.quota = MagicMock()
        self.wave_tracker = MagicMock()

    with patch.object(RuntimeGuardProcessor, "_init_from_runner", _init):
        proc = RuntimeGuardProcessor.bind(runner)

    calls: list[str] = []

    def sample_fn():
        calls.append("sample")
        return SamplePhaseResult(
            scheduler_output=None,
            input_batch=None,
            model_runner_output=None,
            sampler_output=MagicMock(),
            valid_sampled_token_ids=[1],
            logprobs_lists=None,
            req_ids_output_copy=["r1"],
            req_id_to_index_output_copy=None,
            invalid_req_indices=None,
            finished_req_ids=None,
        )

    proc.ensure_logprobs_for_detection = lambda: calls.append("ensure")  # type: ignore[method-assign]
    proc.note_kv_block_writes = lambda *a, **k: calls.append("note")  # type: ignore[method-assign]
    proc.mark_finished = lambda *a, **k: calls.append("mark")  # type: ignore[method-assign]
    proc.record_sample_waves = lambda *a, **k: calls.append("waves")  # type: ignore[method-assign]
    proc.check_after_sample = lambda *a, **k: calls.append("after")  # type: ignore[method-assign]

    proc.run_sample_phase(
        sample_fn=sample_fn,
        speculative_config=None,
        need_accepted_tokens=False,
        use_async=False,
    )
    assert calls == ["sample"]
    RuntimeGuardProcessor.reset_for_tests()


def test_broadcast_skips_all_reduce_when_far_from_interval(tmp_path: Path):
    """B-path: clearly before interval → no collective."""
    cfg = _cfg(tmp_path, reload=5.0)
    cfg._initial_broadcast_done = True
    cfg._last_reload_ts = time.time()  # just synced

    group = MagicMock()
    group.world_size = 2
    group.cpu_group = object()
    with patch("torch.distributed.all_reduce") as ar:
        assert cfg._maybe_reload_broadcast(group) is False
        ar.assert_not_called()


def test_refresh_config_skips_filter_apply_when_unchanged(tmp_path: Path):
    RuntimeGuardProcessor.reset_for_tests()
    cfg = _cfg(tmp_path, reload=0.05)
    runner = MagicMock()
    runner.tp_rank = 0
    ifm = MagicMock()

    def _init(self, r):
        self.runner = r
        self.runtime_config = cfg
        self.manual_triggers = MagicMock()
        self.manual_triggers.consume_once.return_value = None
        self.action_executor = MagicMock()
        self.detectors = MagicMock()
        self.report_writer = MagicMock()
        self.kv_reader = MagicMock()
        self.quota = MagicMock()

    with (
        patch.object(RuntimeGuardProcessor, "_init_from_runner", _init),
        patch(
            "vllm_ascend.runtime_guard.processor.InputFilterManager"
        ) as ifm_cls,
        patch.object(cfg, "sync_runtime_config", return_value=False),
    ):
        ifm_cls.get.return_value = ifm
        proc = RuntimeGuardProcessor.bind(runner)
        proc.refresh_config(allow_arm=True)

    ifm.apply_from_config.assert_not_called()
    RuntimeGuardProcessor.reset_for_tests()
