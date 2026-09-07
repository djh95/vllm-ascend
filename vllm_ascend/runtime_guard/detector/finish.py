#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Request-finish detector: emit one report when a request is reaped."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from vllm_ascend.runtime_guard.detector.config_backed import ConfigBackedDetector
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.request_state import RequestGuardStore
from vllm_ascend.runtime_guard.types import ILL_TYPE_NONE
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


class FinishDetector(ConfigBackedDetector):
    """Lifecycle sensor: one non-ill incident per finished request at reap time.

    Not an anomaly detector — ``is_ill=False`` and ``consume_quota=False`` so
    finish reports do not arm dump quota or mark ``stop_after_alert``. Default
    ``on_trigger`` is ``["report"]`` (inherited from ``actions.defaults``).
    """

    incident_type = "finish"
    EST_COST_PER_STEP = 0.1  # once per finished req, not every decode step
    section_key = "finish"

    def __init__(self, *, runtime_config: Any | None = None, runner: Any | None = None) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        return

    def check_finished(self, req_ids: Iterable[str]) -> list[Incident]:
        """Build one finish incident per reapable request (placement-gated by caller)."""
        if not self._precheck():
            return []
        store = RequestGuardStore.get()
        out: list[Incident] = []
        for raw in req_ids:
            if not raw:
                continue
            req_id = str(raw)
            if not self._passes_input_filter(req_id, log=False):
                continue
            state = store.get_state(req_id)
            detail: dict[str, Any] = {"reason": "request_finished"}
            if state is not None and state.finish_mark_wave is not None:
                detail["finish_mark_wave"] = int(state.finish_mark_wave)
            out.append(
                Incident(
                    incident_type=self.incident_type,
                    req_id=req_id,
                    is_ill=False,
                    ill_type=ILL_TYPE_NONE,
                    detail=detail,
                    consume_quota=False,
                )
            )
        if out:
            logger.debug(
                "[Anomaly finish] emitting %d finish report(s) req_ids=%s",
                len(out),
                [a.req_id for a in out],
            )
        return out
