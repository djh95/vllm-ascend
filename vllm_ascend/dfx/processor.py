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

"""Runner-side DFX processing (config sync, detectors, dump/log/report sinks).

Owns construction of ``Dumper`` / detectors / ``DfxReportWriter`` so v1 and v2
model runners only hang a ``DfxProcessor`` and call thin step hooks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.dfx.detector.manual_dump import ManualDumpDetector
from vllm_ascend.dfx.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.dfx.detector.token_logprob import TokenLogprobDetector
from vllm_ascend.dfx.dumper import Dumper
from vllm_ascend.dfx.report import DfxReportWriter
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)


class DfxProcessor:
    """Processes DFX step events: config → detect → dump/report (and later log/trace)."""

    def __init__(self, runner: Any) -> None:
        ascend = runner.ascend_config
        dfx_config: DfxRuntimeConfig = ascend.dfx_config
        dynamic_dump_config = ascend.dynamic_dump_config

        self.runner = runner
        self.dfx_config = dfx_config
        # All workers log the path; only the JSON writer rank materializes once.
        logger.info("[DFX] worker runtime config path=%s", dfx_config.config_path)
        dfx_config.ensure_persisted()
        self.dumper = Dumper(
            runner,
            dynamic_dump_config,
            dfx_config=dfx_config,
        )
        self.report_writer = DfxReportWriter(dfx_config.report_dir)
        self.spec_detector = SpecAcceptanceDetector(
            dfx_config=dfx_config,
            runner=runner,
            is_related_request=self.dumper.is_related_local_request,
            dynamic_dump_config=dynamic_dump_config,
        )
        self.token_logprob_detector = TokenLogprobDetector(
            dfx_config=dfx_config,
            runner=runner,
            dynamic_dump_config=dynamic_dump_config,
        )
        self.manual_dump_detector = ManualDumpDetector(
            dfx_config=dfx_config,
            runner=runner,
        )

    # ---- step entry (all ranks) --------------------------------------------

    def refresh_config(self) -> bool:
        """All-rank DFX config sync. Must not be skipped on early PP."""
        logger.debug("[DFX sync] enter stage=refresh_config")
        changed = self.dfx_config.sync_dfx_config()
        if changed:
            self.dumper.apply_dfx_config()
        else:
            # Keep dump limits aligned with live ``dfx_config`` even when this
            # step had no file mtime change (e.g. broadcast payload already applied).
            self.dumper._sync_dump_limits_from_config()
        # Drain dump_once whenever it is set in live config — not only on the
        # mtime-changed step. Otherwise a missed/racy ``changed`` leaves
        # dump_once stuck true on disk with no dump.
        if self.dfx_config.dump_once():
            for alert in self.manual_dump_detector.check_all():
                ok = self._handle_alert(alert, detector=self.manual_dump_detector, write_report=False)
                if not ok:
                    logger.error(
                        "[DFX manual_dump] dump_once consumed but dump arm failed "
                        "(check dump.enabled / dump_config_path / debugger). "
                        "Re-set dump.dump_once=true after fixing."
                    )
        # Always pull detector flags from live config. Init may have applied
        # ``dynamic_dump_config`` only; JSON hot-reload updates ``dfx_config``.
        self.spec_detector.refresh_from_config()
        self.token_logprob_detector.refresh_from_config()
        self.manual_dump_detector.refresh_from_config()
        logger.debug("[DFX sync] leave stage=refresh_config changed=%s", changed)
        return changed

    def sync_dump_pending_or(self, *, allow_arm: bool = True) -> bool:
        """Last-PP TP dump OR only (see ``Dumper.sync_dump_pending_or``)."""
        logger.debug("[DFX sync] enter stage=sync_dump_pending_or allow_arm=%s", allow_arm)
        ok = self.dumper.sync_dump_pending_or(allow_arm=allow_arm)
        logger.debug("[DFX sync] leave stage=sync_dump_pending_or ok=%s", ok)
        return ok

    # ---- sample / get_output hooks ----------------------------------------

    def clear_finished(self, finished_req_ids: Any) -> None:
        if not finished_req_ids:
            return
        for req_id in finished_req_ids:
            self.spec_detector.clear_finished(req_id)
            self.token_logprob_detector.clear_finished(req_id)

    def check_spec_acceptance(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
    ) -> None:
        if not self.dumper.can_run_anomaly_detection():
            reason = self.dumper.anomaly_check_skip_reason()
            # Only TP0, throttled: non-TP0 always skips under pending-OR and would flood.
            if reason and int(getattr(self.runner, "tp_rank", 0)) == 0:
                now = time.time()
                last = getattr(self, "_spec_skip_log_ts", 0.0)
                if now - last >= 2.0:
                    self._spec_skip_log_ts = now
                    logger.info(
                        "[Anomaly spec short] skip gate: %s (enable_spec_acceptance_check=%s dump.enabled=%s)",
                        reason,
                        self.dfx_config.detector_get("enable_spec_acceptance_check", False),
                        self.dfx_config.dump_enabled(),
                    )
            return
        for alert in self.spec_detector.check_all(sampled_tokens, accepted_token_nums):
            self._handle_alert(alert, detector=self.spec_detector)

    def check_token_logprobs(
        self,
        sampled_token_ids: Any,
        logprobs_lists: Any,
        req_ids: list[str] | None = None,
    ) -> None:
        if not self.dumper.can_run_anomaly_detection():
            # Spec short logs can still print while token check is gated off
            # (pending/active dump, non-output rank under async, etc.).
            reason = self.dumper.anomaly_check_skip_reason()
            if reason:
                logger.info_once(
                    "[Anomaly token_logprob short] skip gate: %s (enable_token_logprob_check=%s dump.enabled=%s)",
                    reason,
                    self.dfx_config.detector_get("enable_token_logprob_check", False),
                    self.dfx_config.dump_enabled(),
                )
            return
        for alert in self.token_logprob_detector.check_all(
            sampled_token_ids=sampled_token_ids,
            logprobs_lists=logprobs_lists,
            req_ids=req_ids,
        ):
            self._handle_alert(alert, detector=self.token_logprob_detector)

    def ensure_logprobs_for_detection(self) -> None:
        """Bump per-request top-k logprobs so TokenLogprobDetector can run.

        Clients need not set ``logprobs`` on the request when
        ``enable_token_logprob_check`` is on and ``dump.enabled``.
        ``dump.max_times`` only gates dump arming, not this force path.
        Safe no-op when the check is disabled or the batch has no sampling state.
        """
        det = self.token_logprob_detector
        det.refresh_from_config()
        if not det.enabled:
            return
        # Force top-k whenever the detector is on; dump quota is separate.
        if not self.dfx_config.dump_enabled():
            return
        topk = det.topk
        if topk <= 0:
            return
        runner = self.runner
        # v1 InputBatch: dict[str, int] num_logprobs + SamplingMetadata
        input_batch = getattr(runner, "input_batch", None)
        num_logprobs = getattr(input_batch, "num_logprobs", None) if input_batch is not None else None
        if isinstance(num_logprobs, dict):
            req_ids = getattr(input_batch, "req_ids", None) or []
            changed = False
            for req_id in req_ids:
                cur = num_logprobs.get(req_id)
                # None / missing: no logprobs. Keep full-vocab requests as-is.
                if cur is None or (0 <= cur < topk):
                    num_logprobs[req_id] = topk
                    changed = True
            if changed and hasattr(input_batch, "_make_sampling_metadata"):
                input_batch.sampling_metadata = input_batch._make_sampling_metadata()
                logger.info_once(
                    "[Anomaly token_logprob] forcing request top-k logprobs=%d for detection",
                    topk,
                )
            return

        # v2 SamplingStates: num_logprobs[req_state_idx], -1 == none
        sampler = getattr(runner, "sampler", None)
        states = getattr(sampler, "sampling_states", None) if sampler is not None else None
        if states is None or input_batch is None:
            return
        idx_mapping_np = getattr(input_batch, "idx_mapping_np", None)
        num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
        if idx_mapping_np is None or num_reqs <= 0:
            return
        arr = states.num_logprobs
        changed = False
        for batch_i in range(num_reqs):
            req_state_idx = int(idx_mapping_np[batch_i])
            cur = int(arr[req_state_idx])
            # v2: -1 means no logprobs; positive N means top-N.
            if cur < 0 or cur < topk:
                arr[req_state_idx] = topk
                changed = True
        if changed:
            logger.info_once(
                "[Anomaly token_logprob] forcing request top-k logprobs=%d for detection (v2)",
                topk,
            )

    def _handle_alert(
        self,
        alert: AnomalyAlert,
        *,
        detector: AnomalyDetector | None = None,
        write_report: bool = True,
    ) -> bool:
        ok = self.dumper.handle_anomaly_alert(alert, detector=detector)
        if not ok:
            return False
        # Log sink (not dump lifecycle): sampling meta for the anomalous request.
        if alert.mark_full_log:
            self.save_sample_param(alert.req_id)
        if write_report:
            self.report_writer.write(
                anomaly_type=alert.anomaly_type,
                req_id=alert.req_id,
                detail=alert.to_report_detail(),
                rank_tag=self.dumper._dump_rank_tag(),
            )
        return True

    def save_sample_param(self, target_req_id: str) -> None:
        """Log sampling metadata for ``target_req_id`` (TP0 + last PP only)."""
        runner = getattr(self, "runner", None)
        if runner is None or not target_req_id:
            return
        try:
            if int(getattr(runner, "tp_rank", 0)) != 0:
                return
            if not get_pp_group().is_last_rank:
                return
        except Exception:
            return

        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None:
            return
        sampling_metadata = getattr(input_batch, "sampling_metadata", None)
        if sampling_metadata is None:
            return
        req_ids = input_batch.req_ids
        for req_idx, req_id in enumerate(req_ids):
            if req_id != target_req_id:
                continue

            temp = sampling_metadata.temperature[req_idx].item() if sampling_metadata.temperature is not None else None
            topk = sampling_metadata.top_k[req_idx].item() if sampling_metadata.top_k is not None else None
            topp = sampling_metadata.top_p[req_idx].item() if sampling_metadata.top_p is not None else None

            freq_pen = sampling_metadata.frequency_penalties[req_idx].item()
            pres_pen = sampling_metadata.presence_penalties[req_idx].item()
            rep_pen = sampling_metadata.repetition_penalties[req_idx].item()

            req_bad_words = sampling_metadata.bad_words_token_ids.get(req_idx, [])
            req_output_tokens = (
                sampling_metadata.output_token_ids[req_idx]
                if sampling_metadata.output_token_ids and req_idx < len(sampling_metadata.output_token_ids)
                else []
            )
            req_spec_tokens = (
                sampling_metadata.spec_token_ids[req_idx]
                if sampling_metadata.spec_token_ids and req_idx < len(sampling_metadata.spec_token_ids)
                else None
            )
            if sampling_metadata.logprob_token_ids:
                req_logprob_tokens = sampling_metadata.logprob_token_ids.get(req_idx, [])
            else:
                req_logprob_tokens = None

            logger.info(
                "[SamplingMeta] req_id=%s req_idx=%d "
                "dp_rank=%d tp_rank=%d "
                "temperature=%.4f top_k=%s top_p=%.4f "
                "freq_pen=%.4f pres_pen=%.4f rep_pen=%.4f "
                "bad_words_group_num=%d output_tokens_len=%d spec_tokens_len=%s logprob_target_tokens_len=%s "
                "all_greedy=%s all_random=%s max_num_logprobs=%s",
                req_id,
                req_idx,
                runner.dp_rank,
                runner.tp_rank,
                temp if temp is not None else -1,
                topk,
                topp if topp is not None else 1.0,
                freq_pen,
                pres_pen,
                rep_pen,
                len(req_bad_words),
                len(req_output_tokens),
                len(req_spec_tokens) if req_spec_tokens else None,
                len(req_logprob_tokens) if req_logprob_tokens else None,
                sampling_metadata.all_greedy,
                sampling_metadata.all_random,
                sampling_metadata.max_num_logprobs,
            )
