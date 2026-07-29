import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.dfx.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.dfx.dumper import Dumper


def _make_dumper() -> Dumper:
    return Dumper.__new__(Dumper)


def test_finalize_dump_data_uses_debugger_specific_step_signature():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = True
    dumper._msprobe_dump_active = False
    dumper._dump_needs_forward = False
    dumper._dump_forward_seen = False
    dumper.disable_msprobe_dump_if_needed = MagicMock()

    dumper.finalize_dump_data()

    dumper._debugger.stop.assert_called_once_with()
    dumper._debugger.step.assert_called_once_with()
    dumper.disable_msprobe_dump_if_needed.assert_called_once_with()


def test_finalize_dump_data_does_not_consume_dummy_forward():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = True
    dumper._msprobe_dump_active = True
    dumper._dump_needs_forward = True
    dumper._dump_forward_seen = True
    dumper.disable_msprobe_dump_if_needed = MagicMock()

    dumper.finalize_dump_data(dump=False)

    dumper._debugger.step.assert_called_once_with(dump=False)
    assert not dumper._dump_forward_seen
    dumper.disable_msprobe_dump_if_needed.assert_not_called()


def test_spec_check_preserves_existing_full_log_marker_when_not_triggered():
    req_id = "req-1"
    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(
        tp_rank=0,
        input_batch=SimpleNamespace(req_output_token_ids=[[10]]),
    )
    dumper.full_log_requests_this_step = {req_id: True}
    dumper.is_related_local_request = MagicMock(return_value=True)
    dumper.enable_msprobe_dump_if_needed = MagicMock()
    dumper.handle_anomaly_alert = MagicMock()

    detector = SpecAcceptanceDetector.__new__(SpecAcceptanceDetector)
    detector._runner = dumper.runner
    detector._dfx_config = None
    detector._is_related_request = dumper.is_related_local_request
    detector._enabled = True
    detector._history = defaultdict(list)
    detector._window = 1
    detector._low_threshold = 0.1
    detector._len_low_threshold = 0.1
    detector._high_threshold = 2.0
    detector._len_high_threshold = 2.0

    with patch("vllm_ascend.dfx.detector.spec_acceptance.get_pp_group") as get_pp_group:
        get_pp_group.return_value.is_last_rank = True
        alert = detector.check_one(
            req_idx=0,
            req_id=req_id,
            req_state=SimpleNamespace(
                prev_num_draft_len=1,
                prompt_token_ids=[],
                output_token_ids=[],
            ),
            accepted_token_num=2,
            sampled_ids=[10, 11],
        )

    assert alert is None
    assert dumper.full_log_requests_this_step == {req_id: True}
    dumper.handle_anomaly_alert.assert_not_called()
    dumper.enable_msprobe_dump_if_needed.assert_not_called()


def test_handle_anomaly_alert_marks_full_log():
    from vllm_ascend.dfx.detector.alert import AnomalyAlert

    dumper = _make_dumper()
    dumper.full_log_requests_this_step = {}
    dumper._debug_log_full_carry = {}
    dumper.enable_msprobe_dump_if_needed = MagicMock(return_value=True)
    alert = AnomalyAlert(
        anomaly_type="token_logprob",
        req_id="req-1",
        req_idx=0,
        is_ill=True,
        ill_type=1,
        detail={"hits": 1},
        skip_related_check=True,
        mark_full_log=True,
    )
    assert dumper.handle_anomaly_alert(alert) is True
    dumper.enable_msprobe_dump_if_needed.assert_called_once()
    assert dumper.full_log_requests_this_step["req-1"] is True
    assert dumper.take_debug_log_full() == {"req-1": True}
    assert dumper._debug_log_full_carry == {}


def test_dump_forward_arms_debug_log_full_survives_next_start_clear():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = False
    dumper._msprobe_dump_active = True
    dumper._dump_needs_forward = True
    dumper._dump_forward_seen = False
    dumper._dump_full_log_req_id = "req-dump"
    dumper.full_log_requests_this_step = {}
    dumper._debug_log_full_carry = {}
    dumper.runner = SimpleNamespace(model=MagicMock())
    dumper._dump_rank_tag = MagicMock(return_value="tp0")
    dumper._dump_state_tag = MagicMock(return_value="active")

    dumper.start_dump_data()
    assert dumper.full_log_requests_this_step == {"req-dump": True}
    assert dumper._debug_log_full_carry == {"req-dump": True}

    # Next step's start clears per-step map; carry must remain until take().
    dumper._msprobe_dump_active = False
    dumper._dump_needs_forward = False
    dumper.start_dump_data()
    assert dumper.full_log_requests_this_step == {}
    assert dumper.take_debug_log_full() == {"req-dump": True}


def test_sync_dump_pending_or_does_not_touch_config():
    dumper = _make_dumper()
    dumper.dfx_config = MagicMock()
    dumper._pending_dump = False
    dumper._anomaly_dump_feature_enabled = MagicMock(return_value=False)
    dumper._use_pending_dump_sync = MagicMock(return_value=False)
    dumper.apply_dfx_config = MagicMock()

    assert dumper.sync_dump_pending_or() is False
    dumper.dfx_config.maybe_reload.assert_not_called()
    dumper.dfx_config.sync_dfx_config.assert_not_called()
    dumper.apply_dfx_config.assert_not_called()


def test_async_pending_does_not_consume_quota_before_activation():
    req_id = "req-1"
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper.dfx_config = MagicMock()
    dumper.dfx_config.dump_enabled.return_value = True
    dumper._pending_dump = False
    dumper._pending_dump_req_id = None
    dumper._pending_dump_skip_quota = False
    dumper._msprobe_dump_active = False
    dumper._msprobe_dumped_req_ids = set()
    dumper._msprobe_dump_total_count = 0
    dumper._dump_max_times = 2
    dumper._msprobe_last_dump_ts = None
    dumper._dump_cooldown_seconds = 0
    dumper._use_pending_dump_sync = MagicMock(return_value=True)

    with patch("vllm_ascend.dfx.dumper.get_pp_group") as get_pp_group:
        get_pp_group.return_value.is_last_rank = True
        armed = dumper.enable_msprobe_dump_if_needed(
            req_id,
            skip_related_check=True,
        )

    assert armed
    assert dumper._pending_dump
    assert dumper._pending_dump_req_id == req_id
    assert dumper._msprobe_dump_total_count == 0


def test_dump_once_via_manual_detector_skips_quota():
    from vllm_ascend.dfx.detector.manual_dump import MANUAL_DUMP_REQ_ID, ManualDumpDetector

    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(tp_rank=0, use_async_scheduling=False)
    dumper._debugger = MagicMock()
    dumper._pending_dump = False
    dumper._pending_dump_req_id = None
    dumper._pending_dump_skip_quota = False
    dumper._msprobe_dump_active = False
    dumper._msprobe_dumped_req_ids = set()
    dumper._msprobe_dump_total_count = 0
    dumper._dump_max_times = 0
    dumper._msprobe_last_dump_ts = time.time()
    dumper._dump_cooldown_seconds = 10_000
    dumper.set_msprobe_dump_state = MagicMock(return_value=True)
    dumper.dfx_config = MagicMock()
    dumper.dfx_config.dump_enabled.return_value = True
    dumper.full_log_requests_this_step = {}
    dumper._use_pending_dump_sync = MagicMock(return_value=False)

    detector = ManualDumpDetector(dfx_config=MagicMock(), runner=dumper.runner)
    detector._dfx_config.consume_dump_once.return_value = True

    with (
        patch("vllm_ascend.dfx.detector.manual_dump.get_pp_group") as get_pp_det,
        patch("vllm_ascend.dfx.dumper.get_pp_group") as get_pp_dump,
    ):
        get_pp_det.return_value.is_last_rank = True
        get_pp_dump.return_value.is_last_rank = True
        alerts = detector.check_all()
        assert len(alerts) == 1
        assert alerts[0].req_id == MANUAL_DUMP_REQ_ID
        assert alerts[0].consume_quota is False
        assert dumper.handle_anomaly_alert(alerts[0], detector=detector) is True

    assert dumper._msprobe_dump_active
    assert dumper._msprobe_dump_total_count == 0
    assert MANUAL_DUMP_REQ_ID not in dumper.full_log_requests_this_step


def test_consume_dump_once_persists_false(tmp_path: Path):
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.save({"dump": {"dump_once": True}})
    assert cfg.dump_once() is True
    assert cfg.consume_dump_once() is True
    assert cfg.dump_once() is False
    reloaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert reloaded["dump"]["dump_once"] is False
    assert cfg.consume_dump_once() is False
