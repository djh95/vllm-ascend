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

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)


class SpecAcceptanceDetector(AnomalyDetector):
    """Detect abnormal speculative-decoding acceptance rate / length."""

    anomaly_type = "spec_acceptance"

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
        runner: Any | None = None,
        is_related_request: Callable[[str, int | None], bool] | None = None,
        dynamic_dump_config: Any | None = None,
    ) -> None:
        enabled = True
        if dynamic_dump_config is not None:
            enabled = bool(getattr(dynamic_dump_config, "enable_spec_acceptance_check", True))
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=enabled)
        self._is_related_request = is_related_request
        # Per-req sliding window: (accepted_draft, draft_len, sampled_ids, accepted_ids)
        self._history: dict[str, deque[tuple[int, int, list[int], list[int]]]] = defaultdict(deque)
        self._window = 10
        self._low_threshold = 0.3
        self._len_low_threshold = 1.4
        self._high_threshold = 0.96
        self._len_high_threshold = 2.8
        # Throttle INFO short logs (per req) so TP0 is not flooded.
        self._short_log_ts: dict[str, float] = {}
        self._short_log_interval_s = 2.0
        # Prefer live DFX JSON; legacy dynamic_dump_config is fallback only.
        if dfx_config is not None:
            self.refresh_from_config()
        elif dynamic_dump_config is not None:
            self._apply_values(dynamic_dump_config)

    def _apply_values(self, src: Any) -> None:
        getter = src.get if isinstance(src, dict) else lambda k, d=None: getattr(src, k, d)
        self._enabled = bool(getter("enable_spec_acceptance_check", self._enabled))
        self._window = int(getter("spec_acceptance_window", self._window))
        self._low_threshold = float(getter("spec_acceptance_low_threshold", self._low_threshold))
        self._len_low_threshold = float(getter("spec_acceptance_len_low_threshold", self._len_low_threshold))
        self._high_threshold = float(getter("spec_acceptance_high_threshold", self._high_threshold))
        self._len_high_threshold = float(getter("spec_acceptance_len_high_threshold", self._len_high_threshold))

    def refresh_from_config(self) -> None:
        if self._dfx_config is None:
            return
        self._apply_values(self._dfx_config.detector)

    def clear_finished(self, req_id: str) -> None:
        self._history.pop(req_id, None)
        self._short_log_ts.pop(req_id, None)

    def check_all(
        self,
        sampled_tokens: torch.Tensor,
        accepted_token_nums: Any,
    ) -> list[AnomalyAlert]:
        """Batch entry: return alerts for the model runner to hand to Dumper."""
        if not self._precheck():
            runner = self._runner
            if int(getattr(runner, "tp_rank", 0) if runner is not None else 0) == 0:
                logger.info_once("[Anomaly spec short] skip: enable_spec_acceptance_check=false in live DFX config")
            return []
        runner = self._runner
        if runner is None:
            return []
        # Spec check needs speculative decoding, not only MambaSpec
        # (``need_accepted_tokens``). Plain MTP / Eagle also produce accept stats.
        if getattr(runner, "speculative_config", None) is None:
            if int(getattr(runner, "tp_rank", 0)) == 0:
                logger.info_once("[Anomaly spec short] skip: speculative_config is None")
            return []
        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None or not getattr(input_batch, "req_ids", None):
            return []

        num_reqs = len(input_batch.req_ids)
        if torch.is_tensor(accepted_token_nums):
            accepted_list = accepted_token_nums[:num_reqs].tolist()
        else:
            accepted_list = [int(x) for x in accepted_token_nums[:num_reqs]]

        sampled_token_rows = sampled_tokens[:num_reqs]
        requests = getattr(runner, "requests", None)
        draft_lens = getattr(input_batch, "num_draft_tokens_per_req", None)

        alerts: list[AnomalyAlert] = []
        for batch_idx, req_id in enumerate(input_batch.req_ids):
            accepted_token_num = int(accepted_list[batch_idx])
            sampled_ids = sampled_token_rows[batch_idx]

            if requests is not None and req_id in requests:
                req_state = requests[req_id]
            else:
                draft_len = int(draft_lens[batch_idx]) if draft_lens is not None else 0
                req_state = SimpleNamespace(
                    prev_num_draft_len=draft_len,
                    prompt_token_ids=None,
                    output_token_ids=None,
                )

            alert = self.check_one(
                req_idx=batch_idx,
                req_id=req_id,
                req_state=req_state,
                accepted_token_num=accepted_token_num,
                sampled_ids=sampled_ids,
            )
            if alert is not None:
                alerts.append(alert)
        return alerts

    def check_one(
        self,
        req_idx: int,
        req_id: str,
        req_state: Any,
        accepted_token_num: int,
        sampled_ids: list[int] | torch.Tensor | None = None,
    ) -> AnomalyAlert | None:
        if not req_id:
            return None
        if not get_pp_group().is_last_rank:
            return None
        runner = self._runner
        log_leader = int(getattr(runner, "tp_rank", 0) if runner is not None else 0) == 0
        draft_len = getattr(req_state, "prev_num_draft_len", 0) or 0
        sampled_norm = self._normalize_token_ids(sampled_ids)
        if draft_len <= 0:
            # Fallback when prev_num_draft_len was not populated (common on
            # non-hybrid MTP): treat last dim of sampled row as draft+bonus.
            draft_len = max(0, len(sampled_norm) - 1)
        if draft_len <= 0:
            if log_leader:
                logger.info_once(
                    "[Anomaly spec short] req_id=%s skip: draft_len=0 sampled_len=%d",
                    req_id,
                    len(sampled_norm),
                )
            return None
        # Related-request filter is for dump arming only; always emit short logs
        # for local batch rows so incorrect filters are visible.
        related_ok = True
        if self._is_related_request is not None and not self._is_related_request(req_id, req_idx):
            related_ok = False
        accepted_draft_tokens = max(0, accepted_token_num - 1)
        accepted_norm = sampled_norm[:accepted_token_num] if accepted_token_num > 0 else []
        history = self._history[req_id]
        prev_hist_len = len(history)
        history.append((accepted_draft_tokens, draft_len, sampled_norm, accepted_norm))
        while len(history) > self._window:
            history.popleft()

        prompt_token_ids_raw = getattr(req_state, "prompt_token_ids", None)
        input_batch = getattr(runner, "input_batch", None) if runner is not None else None
        req_output_token_ids = getattr(input_batch, "req_output_token_ids", None) if input_batch else None
        if req_output_token_ids is not None and 0 <= req_idx < len(req_output_token_ids):
            output_token_ids_raw = req_output_token_ids[req_idx]
        else:
            output_token_ids_raw = getattr(req_state, "output_token_ids", None)
        prompt_token_count = len(prompt_token_ids_raw) if prompt_token_ids_raw is not None else 0
        output_token_count = len(output_token_ids_raw) if output_token_ids_raw is not None else 0
        accepted_sum = sum(accepted for accepted, _, _, _ in history)
        draft_sum = sum(draft for _, draft, _, _ in history)
        acceptance_rate = accepted_sum / draft_sum if draft_sum > 0 else 0.0
        acceptance_len = accepted_sum / len(history) if history else 0.0

        window_ready = len(history) >= self._window
        just_filled = prev_hist_len < self._window <= len(history)
        should_alert = False
        if window_ready:
            should_alert = bool(
                (acceptance_rate < self._low_threshold and acceptance_len < self._len_low_threshold)
                or (acceptance_rate > self._high_threshold and acceptance_len > self._len_high_threshold)
            )

        # INFO on alert / related miss / window just filled, else throttled INFO
        # so hot-reload / mid-request enable is visible without per-step flood.
        if log_leader:
            now = time.time()
            last = self._short_log_ts.get(req_id, 0.0)
            interesting = bool(should_alert and related_ok) or (not related_ok) or just_filled
            due_info = interesting or (now - last >= self._short_log_interval_s)
            if due_info:
                self._short_log_ts[req_id] = now
                logger.info(
                    "[Anomaly spec short] req_id=%s draft_len=%d "
                    "accepted_count=%d accepted_draft_count=%d "
                    "accept_rate=%.4f accept_len=%.4f window=%d/%d accepted=%d drafted=%d "
                    "prompt_tokens=%d output_tokens=%d "
                    "low=(%.2f,%.2f) high=(%.2f,%.2f) related=%s alert=%s",
                    req_id,
                    draft_len,
                    accepted_token_num,
                    accepted_draft_tokens,
                    acceptance_rate,
                    acceptance_len,
                    len(history),
                    self._window,
                    accepted_sum,
                    draft_sum,
                    prompt_token_count,
                    output_token_count,
                    self._low_threshold,
                    self._len_low_threshold,
                    self._high_threshold,
                    self._len_high_threshold,
                    related_ok,
                    should_alert and related_ok,
                )
            else:
                logger.debug(
                    "[Anomaly spec short] req_id=%s window=%d/%d alert=%s",
                    req_id,
                    len(history),
                    self._window,
                    should_alert and related_ok,
                )

        if not related_ok or not window_ready or not should_alert:
            return None

        window_sampled_token_ids = [step_sampled for _, _, step_sampled, _ in history]
        window_accepted_token_ids = [step_accepted for _, _, _, step_accepted in history]
        return AnomalyAlert(
            anomaly_type=self.anomaly_type,
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=0,
            detail={
                "acceptance_rate": acceptance_rate,
                "acceptance_len": acceptance_len,
                "accepted_sum": accepted_sum,
                "draft_sum": draft_sum,
                "window": len(history),
                "window_sampled_token_ids": window_sampled_token_ids,
                "window_accepted_token_ids": window_accepted_token_ids,
            },
            skip_related_check=False,
            mark_full_log=log_leader,
            log_context={
                "sampled_ids": sampled_norm,
                "accepted_token_num": accepted_token_num,
                "prompt_token_ids_raw": prompt_token_ids_raw,
                "output_token_ids_raw": output_token_ids_raw,
                "window_sampled_token_ids": window_sampled_token_ids,
                "window_accepted_token_ids": window_accepted_token_ids,
            },
        )

    def on_alert_armed(self, alert: AnomalyAlert) -> None:
        ctx = alert.log_context
        if not ctx:
            return
        self._log_token_details(
            req_id=alert.req_id,
            sampled_ids=ctx.get("sampled_ids") or [],
            accepted_token_num=int(ctx.get("accepted_token_num") or 0),
            prompt_token_ids_raw=ctx.get("prompt_token_ids_raw"),
            output_token_ids_raw=ctx.get("output_token_ids_raw"),
            window_sampled_token_ids=ctx.get("window_sampled_token_ids") or [],
            window_accepted_token_ids=ctx.get("window_accepted_token_ids") or [],
        )

    def _log_token_details(
        self,
        req_id: str,
        sampled_ids: list[int],
        accepted_token_num: int,
        prompt_token_ids_raw: Any,
        output_token_ids_raw: Any,
        window_sampled_token_ids: list[list[int]] | None = None,
        window_accepted_token_ids: list[list[int]] | None = None,
    ) -> None:
        accepted_token_ids = sampled_ids[:accepted_token_num] if accepted_token_num > 0 else []
        prompt_token_ids = list(prompt_token_ids_raw) if prompt_token_ids_raw is not None else []
        output_token_ids = list(output_token_ids_raw) if output_token_ids_raw is not None else []
        output_token_ids = [
            output_token_id.item() if isinstance(output_token_id, torch.Tensor) else output_token_id
            for output_token_id in output_token_ids
        ]

        logger.info("[Anomaly spec] req_id=%s sampled_token_ids=%s", req_id, sampled_ids)
        logger.info("[Anomaly spec] req_id=%s accepted_token_ids=%s", req_id, accepted_token_ids)
        logger.info(
            "[Anomaly spec] req_id=%s window_sampled_token_ids=%s",
            req_id,
            window_sampled_token_ids or [],
        )
        logger.info(
            "[Anomaly spec] req_id=%s window_accepted_token_ids=%s",
            req_id,
            window_accepted_token_ids or [],
        )
        logger.info(
            "[Anomaly spec] req_id=%s prompt_token_count=%d prompt_token_ids=%s",
            req_id,
            len(prompt_token_ids),
            prompt_token_ids,
        )
        logger.info(
            "[Anomaly spec] req_id=%s output_token_count=%d output_token_ids=%s",
            req_id,
            len(output_token_ids),
            output_token_ids,
        )

    @staticmethod
    def _normalize_token_ids(token_ids: Any) -> list[int]:
        if token_ids is None:
            return []
        if torch.is_tensor(token_ids):
            return token_ids.tolist()
        return list(token_ids)
