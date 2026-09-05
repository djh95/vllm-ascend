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

from vllm_ascend.runtime_guard.detector.placement import ExecScope
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.io_snapshot import normalize_token_ids

if TYPE_CHECKING:
    from vllm_ascend.runtime_config.config import RuntimeConfig


class AnomalyDetector:
    """Base detector: inspects outputs and returns ``Incident`` values.

    Does **not** arm dump — the processor calls ``Dumper.handle_anomaly_alert``.
    Subclasses implement domain-specific ``check_all`` / ``check_one``; the base
    only defines shared lifecycle hooks (optional overrides, not ABC).
    """

    incident_type: str = "unknown"

    # Placement attributes (see detector/placement.py). Under pure TP all
    # detector inputs are rank-redundant (logits all-gathered, redundant
    # sampling, replicated scheduler metadata), so ANY is safe for every
    # current detector; the planner spreads them across ranks.
    EXEC_SCOPE: ExecScope = ExecScope.ANY
    # True when the detector's data is rank-local under CP/DP (planner then
    # runs it on every rank instead of exactly one).
    RANK_LOCAL_DATA: bool = False
    # Relative per-step cost weight used for load-balanced placement.
    EST_COST_PER_STEP: float = 1.0

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
        enabled: bool = True,
    ) -> None:
        self._runtime_config = runtime_config
        self._runner = runner
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def refresh_from_config(self) -> None:
        """Pull live knobs from ``RuntimeConfig`` (default: no-op).

        Config-threshold sensors override via :class:`ConfigBackedDetector`.
        One-shot triggers (e.g. manual dump) may override to keep ``_enabled``.
        """
        return

    def clear_finished(self, req_id: str) -> None:
        """Drop per-request state when a request finishes."""

    def on_alert_armed(self, alert: Incident) -> None:
        """Optional hook after dump arm or detect-only alert handling."""

    def _precheck(self) -> bool:
        """Refresh live thresholds then return whether this detector is enabled.

        Sole per-check pull from ``RuntimeConfig`` (processor no longer
        ``refresh_all`` every step). Config-backed subclasses inherit
        ``ConfigBackedDetector.refresh_from_config``; others may no-op or
        keep a fixed ``_enabled``.
        """
        self.refresh_from_config()
        return self._enabled

    def _passes_input_filter(
        self,
        req_id: str,
        req_idx: int | None = None,
        *,
        prompt_token_ids: Any | None = None,
        log: bool = True,
    ) -> bool:
        """Whether ``InputFilterManager`` allows detection for this request.

        Manual ``manual_trigger`` never calls this (unaffected by filters).
        """
        from vllm_ascend.runtime_guard.input_filters import InputFilterManager

        return InputFilterManager.get().allow(
            req_id,
            runner=self._runner,
            req_idx=req_idx,
            prompt_token_ids=prompt_token_ids,
            log=log,
        )

    @staticmethod
    def _normalize_token_ids(token_ids: Any) -> list[int]:
        return normalize_token_ids(token_ids)
