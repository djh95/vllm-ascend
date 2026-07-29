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

from typing import TYPE_CHECKING, Any

from vllm_ascend.dfx.detector.alert import AnomalyAlert

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig


class AnomalyDetector:
    """Base detector: inspects outputs and returns ``AnomalyAlert`` values.

    Does **not** arm dump — the model runner calls ``Dumper.handle_anomaly_alert``.
    """

    anomaly_type: str = "unknown"

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
        runner: Any | None = None,
        enabled: bool = True,
    ) -> None:
        self._dfx_config = dfx_config
        self._runner = runner
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind_runtime(self, *, dfx_config: DfxRuntimeConfig | None = None, runner: Any | None = None) -> None:
        if dfx_config is not None:
            self._dfx_config = dfx_config
        if runner is not None:
            self._runner = runner

    def refresh_from_config(self) -> None:
        """Pull enable flag / thresholds from live ``DfxRuntimeConfig``."""

    def clear_finished(self, req_id: str) -> None:
        """Drop per-request state when a request finishes."""

    def on_alert_armed(self, alert: AnomalyAlert) -> None:
        """Optional hook after dumper successfully arms / activates dump."""

    def _precheck(self) -> bool:
        self.refresh_from_config()
        return self._enabled
