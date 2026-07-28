from unittest.mock import MagicMock, patch

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.processor import DfxProcessor


def test_dfx_processor_check_spec_writes_report_on_arm():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dumper = MagicMock()
    proc.dumper.can_run_anomaly_detection.return_value = True
    proc.dumper.handle_anomaly_alert.return_value = True
    proc.dumper._dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.spec_detector = MagicMock()
    proc.save_sample_param = MagicMock()
    alert = AnomalyAlert(anomaly_type="spec_acceptance", req_id="r1", detail={"x": 1})
    proc.spec_detector.check_all.return_value = [alert]

    proc.check_spec_acceptance(sampled_tokens=None, accepted_token_nums=None)

    proc.dumper.handle_anomaly_alert.assert_called_once()
    proc.save_sample_param.assert_called_once_with("r1")
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["req_id"] == "r1"


def test_dfx_processor_refresh_runs_manual_dump_without_report_or_sample_log():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.sync_dfx_config.return_value = True
    proc.dumper = MagicMock()
    proc.dumper.handle_anomaly_alert.return_value = True
    proc.report_writer = MagicMock()
    proc.spec_detector = MagicMock()
    proc.token_logprob_detector = MagicMock()
    proc.manual_dump_detector = MagicMock()
    proc.save_sample_param = MagicMock()
    alert = AnomalyAlert(
        anomaly_type="manual_dump_once",
        req_id="__manual_dump_once__",
        consume_quota=False,
        mark_full_log=False,
    )
    proc.manual_dump_detector.check_all.return_value = [alert]

    assert proc.refresh_config() is True
    proc.dumper.apply_dfx_config.assert_called_once()
    proc.spec_detector.refresh_from_config.assert_called_once()
    proc.token_logprob_detector.refresh_from_config.assert_called_once()
    proc.manual_dump_detector.refresh_from_config.assert_called_once()
    proc.dumper.handle_anomaly_alert.assert_called_once()
    proc.save_sample_param.assert_not_called()
    proc.report_writer.write.assert_not_called()


def test_dfx_processor_refresh_no_change_skips_apply():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.dfx_config.sync_dfx_config.return_value = False
    proc.dumper = MagicMock()
    proc.spec_detector = MagicMock()
    proc.token_logprob_detector = MagicMock()
    proc.manual_dump_detector = MagicMock()

    assert proc.refresh_config() is False
    proc.dumper.apply_dfx_config.assert_not_called()
    proc.dumper._sync_dump_limits_from_config.assert_called_once()
    proc.spec_detector.refresh_from_config.assert_called_once()
    proc.token_logprob_detector.refresh_from_config.assert_called_once()
    proc.manual_dump_detector.refresh_from_config.assert_called_once()
    proc.manual_dump_detector.check_all.assert_not_called()


def test_handle_alert_calls_save_sample_param_when_mark_full_log():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dumper = MagicMock()
    proc.dumper.handle_anomaly_alert.return_value = True
    proc.dumper._dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.save_sample_param = MagicMock()
    alert = AnomalyAlert(
        anomaly_type="token_logprob",
        req_id="r2",
        mark_full_log=True,
    )
    assert proc._handle_alert(alert, write_report=True) is True
    proc.save_sample_param.assert_called_once_with("r2")
    proc.report_writer.write.assert_called_once()


def test_save_sample_param_skips_non_tp0():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=1, input_batch=MagicMock())
    with patch("vllm_ascend.dfx.processor.get_pp_group") as pp:
        pp.return_value.is_last_rank = True
        proc.save_sample_param("r1")
    # No crash; non-TP0 returns before needing sampling_metadata.


def test_ensure_logprobs_for_detection_bumps_v1_num_logprobs():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.dfx_config.dump_enabled.return_value = True
    proc.dumper = MagicMock()
    det = MagicMock()
    det.enabled = True
    det.topk = 20
    proc.token_logprob_detector = det

    input_batch = MagicMock()
    input_batch.req_ids = ["r1", "r2"]
    input_batch.num_logprobs = {"r2": 5}  # r1 missing; r2 too small
    input_batch._make_sampling_metadata.return_value = "meta"
    proc.runner = MagicMock(input_batch=input_batch)

    proc.ensure_logprobs_for_detection()

    assert input_batch.num_logprobs["r1"] == 20
    assert input_batch.num_logprobs["r2"] == 20
    input_batch._make_sampling_metadata.assert_called_once()
    assert input_batch.sampling_metadata == "meta"


def test_ensure_logprobs_noop_when_disabled():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    det = MagicMock()
    det.enabled = False
    proc.token_logprob_detector = det
    proc.runner = MagicMock()
    proc.dumper = MagicMock()

    proc.ensure_logprobs_for_detection()

    proc.dfx_config.dump_enabled.assert_not_called()
