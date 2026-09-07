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

"""P0 UT: RuntimeConfig load / soft-fail / hot-reload / dump flags."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vllm_ascend.runtime_config.config import RuntimeConfig


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_defaults_dump_and_detectors_off(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.hot_reload_enabled is False
    assert cfg.dump_enabled() is False
    assert cfg.dump_max_times() == 0
    assert cfg.detectors_enabled_in(cfg._data) is False


def test_startup_overlay_enables_detector(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
        startup_overlay={
            "detector": {
                "token_repeat": {"enabled": True, "window": 8, "repeat_sum_threshold": 4},
            }
        },
    )
    assert cfg.detector_get("token_repeat", "enabled") is True
    assert cfg.detector_get("token_repeat", "window") == 8


def test_malformed_json_keeps_previous(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(
        cfg_path,
        {"detector": {"token_repeat": {"enabled": True, "window": 16}}},
    )
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0.01,
        sync_mode="file",
    )
    assert cfg.detector_get("token_repeat", "enabled") is True
    # Corrupt file; reload must soft-fail and keep in-memory config.
    cfg_path.write_text("{not-json", encoding="utf-8")
    time.sleep(0.02)
    changed = cfg.reload(force=True)
    assert changed is False or cfg.detector_get("token_repeat", "enabled") is True
    assert cfg.detector_get("token_repeat", "window") == 16


def test_hot_reload_picks_up_detector_enable(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {"detector": {"token_repeat": {"enabled": False}}})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0.01,
        sync_mode="file",
    )
    assert cfg.detector_get("token_repeat", "enabled") is False
    _write(
        cfg_path,
        {
            "detector": {
                "token_repeat": {
                    "enabled": True,
                    "window": 8,
                    "repeat_sum_threshold": 4,
                    "min_tokens": 4,
                }
            }
        },
    )
    time.sleep(0.02)
    assert cfg.reload(force=True) is True
    assert cfg.detector_get("token_repeat", "enabled") is True


def test_dump_auto_and_manual_exclusive_or_derived(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {"dump": {"auto_max_times": 0, "manual_dump": False}})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=1,
    )
    assert cfg.dump_enabled() is False
    _write(cfg_path, {"dump": {"auto_max_times": 3, "manual_dump": False}})
    time.sleep(0.01)
    cfg.reload(force=True)
    assert cfg.dump_max_times() == 3
    assert cfg.dump_enabled() is True


def test_actions_default_on_trigger_includes_report(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
    )
    actions = cfg.actions_default_on_trigger()
    assert "report" in actions
