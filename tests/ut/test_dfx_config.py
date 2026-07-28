from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from vllm_ascend.dfx.report import DfxReportWriter
from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig, _leaf_changes


def test_leaf_changes_reports_only_diffs():
    old = {"dump": {"max_times": 0, "enabled": True}, "log": {"level": "INFO"}}
    new = {"dump": {"max_times": 3, "enabled": True}, "log": {"level": "INFO"}}
    assert _leaf_changes(old, new) == ["dump.max_times: 0 -> 3"]


def test_dfx_config_hot_reload_and_defaults(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    report_dir = tmp_path / "report"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=report_dir,
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
    )

    assert cfg_path.exists()
    assert cfg.hot_reload_enabled is True
    assert cfg.dump_max_times() == 0
    assert cfg.log_level() == "INFO"
    assert cfg.detector_get("enable_spec_acceptance_check") is True

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["max_times"] = 3
    payload["log"]["level"] = "DEBUG"
    payload["detector"]["enable_token_logprob_check"] = True
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    # Bypass interval gate.
    cfg._last_reload_ts = 0.0
    assert cfg.maybe_reload() is True
    assert cfg.dump_max_times() == 3
    assert cfg.log_level() == "DEBUG"
    assert cfg.detector_get("enable_token_logprob_check") is True


def test_dfx_hot_reload_disabled_by_default(tmp_path: Path):
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.hot_reload_enabled is False
    assert cfg.maybe_reload() is False
    # File change must not apply while hot-reload is off.
    payload = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    payload["dump"]["max_times"] = 9
    cfg.config_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.maybe_reload() is False
    assert cfg.dump_max_times() == 0


def test_dfx_report_writer_appends_anomaly_line(tmp_path: Path):
    writer = DfxReportWriter(tmp_path / "report")
    path = writer.write(
        anomaly_type="spec_acceptance",
        req_id="req-1",
        detail={"acceptance_rate": 0.1},
        rank_tag="tp0",
    )
    assert path is not None
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["anomaly_type"] == "spec_acceptance"
    assert record["req_id"] == "req-1"
    assert record["rank"] == "tp0"
    assert record["detail"]["acceptance_rate"] == 0.1


def test_legacy_dynamic_dump_overlay(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        legacy_dynamic_dump={
            "dynamic_dump_max_times": 2,
            "enable_token_logprob_check": True,
            "spec_acceptance_window": 20,
        },
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 2
    assert cfg.detector_get("enable_token_logprob_check") is True
    assert cfg.detector_get("spec_acceptance_window") == 20
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["max_times"] == 2
    assert saved["detector"]["spec_acceptance_window"] == 20
    # Defaults filled in for missing keys.
    assert "dump_once" in saved["dump"]
    assert "enable_spec_acceptance_check" in saved["detector"]


def test_empty_legacy_does_not_override_json(tmp_path: Path):
    """Empty startup overlay must not clobber JSON with DynamicDumpConfig defaults."""
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"enabled": True, "max_times": 9, "cooldown_seconds": 10},
                "detector": {
                    "enable_token_logprob_check": True,
                    "spec_acceptance_window": 33,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        legacy_dynamic_dump={},  # same as DynamicDumpConfig({}).user_overrides
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 9
    assert cfg.detector_get("enable_token_logprob_check") is True
    assert cfg.detector_get("spec_acceptance_window") == 33
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["max_times"] == 9
    assert saved["detector"]["spec_acceptance_window"] == 33


def test_explicit_startup_overrides_json(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"enabled": True, "max_times": 9, "cooldown_seconds": 10},
                "detector": {
                    "enable_token_logprob_check": False,
                    "spec_acceptance_window": 33,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        legacy_dynamic_dump={
            "dynamic_dump_max_times": 2,
            "spec_acceptance_window": 20,
        },
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 2
    assert cfg.detector_get("enable_token_logprob_check") is False  # JSON wins (not in startup)
    assert cfg.detector_get("spec_acceptance_window") == 20  # startup wins
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["max_times"] == 2
    assert saved["detector"]["enable_token_logprob_check"] is False
    assert saved["detector"]["spec_acceptance_window"] == 20


def test_hot_reload_follows_json_not_startup_overlay(tmp_path: Path):
    """After bootstrap, editing JSON must win even for keys that were in startup overlay."""
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        legacy_dynamic_dump={
            "dynamic_dump_max_times": 2,
            "spec_acceptance_window": 20,
        },
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert cfg.dump_max_times() == 2
    assert cfg.detector_get("spec_acceptance_window") == 20

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["max_times"] = 9
    payload["detector"]["spec_acceptance_window"] = 33
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg._last_reload_ts = 0.0
    assert cfg.maybe_reload() is True
    assert cfg.dump_max_times() == 9
    assert cfg.detector_get("spec_acceptance_window") == 33


def test_no_explicit_path_startup_overwrites_default_json(tmp_path: Path, monkeypatch):
    """Without dfx_config_path, startup overlay ignores prior default-path content."""
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        json.dumps({"dump": {"max_times": 9}, "detector": {"spec_acceptance_window": 33}}),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        None,  # no explicit path → default under cwd
        legacy_dynamic_dump={"dynamic_dump_max_times": 2, "spec_acceptance_window": 20},
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.config_path == cfg_path.resolve()
    assert cfg.dump_max_times() == 2
    assert cfg.detector_get("spec_acceptance_window") == 20
    # Prior JSON max_times=9 discarded (覆盖).
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["max_times"] == 2
    assert saved["detector"]["spec_acceptance_window"] == 20


def test_bootstrap_and_save_skip_persist_on_non_leader(tmp_path: Path, monkeypatch):
    """Non-leader ranks keep in-memory merge but must not write JSON."""
    monkeypatch.setenv("RANK", "1")
    cfg_path = tmp_path / "dfx_config.json"
    prior = {
        "dump": {"enabled": True, "max_times": 9, "cooldown_seconds": 10, "dump_once": False},
        "log": {"enabled": True, "level": "INFO"},
        "metrics": {"enabled": True, "level": "INFO"},
        "trace": {"enabled": False, "level": "INFO", "otlp_endpoint": None},
        "detector": {"spec_acceptance_window": 33},
    }
    cfg_path.write_text(json.dumps(prior), encoding="utf-8")
    before = cfg_path.read_text(encoding="utf-8")

    cfg = DfxRuntimeConfig(
        cfg_path,
        legacy_dynamic_dump={"dynamic_dump_max_times": 2},
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 2  # in-memory startup overlay applied
    assert cfg_path.read_text(encoding="utf-8") == before  # disk unchanged
    assert cfg.save({"dump": {"max_times": 1}}) is False
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 9


def test_bootstrap_overwrite_default_when_leader(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({"dump": {"max_times": 9}}), encoding="utf-8")
    cfg = DfxRuntimeConfig(
        None,
        legacy_dynamic_dump={"dynamic_dump_max_times": 2},
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 2
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 2


def test_ensure_persisted_deferred_to_worker_leader(tmp_path: Path, monkeypatch):
    """AscendConfig-style: no write at ctor; leader ensure_persisted writes once."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        legacy_dynamic_dump={"dynamic_dump_max_times": 2},
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 2
    assert not cfg_path.exists()
    assert cfg.ensure_persisted() is True
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 2
    # Idempotent.
    mtime = cfg_path.stat().st_mtime
    assert cfg.ensure_persisted() is True
    assert cfg_path.stat().st_mtime == mtime


def test_ensure_persisted_skips_rewrite_when_file_exists(tmp_path: Path, monkeypatch):
    """Existing JSON must not be rewritten on restart (mtime churn / clobber)."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"max_times": 7}, "log": {"level": "WARNING"}}),
        encoding="utf-8",
    )
    mtime_before = cfg_path.stat().st_mtime
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 7
    assert cfg.ensure_persisted() is True
    assert cfg_path.stat().st_mtime == mtime_before
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 7


def test_save_prefers_disk_over_stale_memory(tmp_path: Path, monkeypatch):
    """save() must not wipe hand-edits that landed on disk after bootstrap."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(json.dumps({"dump": {"max_times": 0, "dump_once": True}}), encoding="utf-8")
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 0
    # Concurrent hand-edit on disk (stale memory still has max_times=0).
    cfg_path.write_text(
        json.dumps({"dump": {"max_times": 5, "dump_once": True}}),
        encoding="utf-8",
    )
    assert cfg.save({"dump": {"dump_once": False}}) is True
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["max_times"] == 5
    assert saved["dump"]["dump_once"] is False
    assert cfg.dump_max_times() == 5


def test_overwrite_deferred_removes_stale_default_json(tmp_path: Path, monkeypatch):
    """API-style bootstrap must not leave stale default JSON for the reloader."""
    monkeypatch.delenv("RANK", raising=False)
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({"dump": {"max_times": 9}, "log": {"level": "DEBUG"}}), encoding="utf-8")

    cfg = DfxRuntimeConfig(
        None,
        legacy_dynamic_dump={"dynamic_dump_max_times": 2},
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert cfg.dump_max_times() == 2
    assert not cfg_path.exists()
    # Worker leader later materializes.
    monkeypatch.setenv("RANK", "0")
    assert cfg.ensure_persisted() is True
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 2


def test_ensure_persisted_skip_non_leader(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.ensure_persisted() is False
    assert not cfg_path.exists()


def test_non_worker_background_reload_skips_when_rank_set(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert cfg.start_non_worker_background_reload() is False


def test_non_worker_background_reload_starts_without_rank(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert cfg.start_non_worker_background_reload() is True
    assert cfg._bg_thread is not None and cfg._bg_thread.is_alive()
    # Idempotent.
    assert cfg.start_non_worker_background_reload() is False


def test_reload_noop_when_file_missing(tmp_path: Path):
    cfg_path = tmp_path / "missing.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert not cfg_path.exists()
    cfg._last_reload_ts = 0.0
    assert cfg.reload(force=False) is False
    assert cfg.dump_max_times() == 0


def test_apply_log_switches_sets_dfx_logger_level(tmp_path: Path):
    import logging

    from vllm_ascend.dfx.runtime_config import DFX_LOGGER_NAMES

    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.save({"log": {"enabled": True, "level": "WARNING"}})
    cfg.apply_log_switches()
    for name in DFX_LOGGER_NAMES:
        assert logging.getLogger(name).level == logging.WARNING


def test_broadcast_sync_applies_leader_payload_to_follower(tmp_path: Path):
    leader = DfxRuntimeConfig(
        tmp_path / "leader.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="broadcast",
        reload_interval_seconds=5,
    )
    follower = DfxRuntimeConfig(
        tmp_path / "follower.json",
        report_dir=tmp_path / "report2",
        ensure_file=True,
        sync_mode="broadcast",
        reload_interval_seconds=5,
    )
    # Simulate leader edited dump.max_times.
    assert leader.save({"dump": {"max_times": 7}})

    world = MagicMock()
    world.world_size = 2
    world.cpu_group = object()
    world.is_first_rank = True

    # Leader path: build payload via real reload, then hand object to follower.
    with (
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("torch.distributed.all_reduce") as ar,
        patch.object(world, "broadcast_object", side_effect=lambda obj, src=0: obj),
    ):
        ar.side_effect = lambda t, op=None, group=None: t.fill_(1.0)
        leader._last_reload_ts = 0.0
        leader._initial_broadcast_done = False
        assert leader.maybe_reload() is True

    world.is_first_rank = False
    payload = {"version": float(leader._version), "data": leader._data}
    with (
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("torch.distributed.all_reduce") as ar,
        patch.object(world, "broadcast_object", side_effect=lambda obj, src=0: payload),
    ):
        ar.side_effect = lambda t, op=None, group=None: t.fill_(1.0)
        follower._last_reload_ts = 0.0
        follower._initial_broadcast_done = False
        assert follower.maybe_reload() is True
    assert follower.dump_max_times() == 7
