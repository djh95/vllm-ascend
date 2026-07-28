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

"""Manual one-shot dump trigger via JSON ``dump.dump_once``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.dfx.detector.alert import ILL_TYPE_NONE, AnomalyAlert
from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)

# Synthetic req id for dump routing; not a real scheduler request.
MANUAL_DUMP_REQ_ID = "__manual_dump_once__"


class ManualDumpDetector(AnomalyDetector):
    """Watches ``dump.dump_once`` and emits one alert when set to true.

    Requires ``additional_config.dfx_config_reload_interval > 0`` so the JSON
    change is picked up by hot-reload; otherwise ``dump_once`` never arms.

    Must be polled on **every** rank after config sync so ``consume_dump_once``
    clears in-memory state everywhere; only last-PP (async: TP0) returns an
    alert for the dumper to arm.
    """

    anomaly_type = "manual_dump_once"

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=True)

    def refresh_from_config(self) -> None:
        # Always enabled when constructed; gate is dump_once + dump.enabled on dumper.
        self._enabled = True

    def check_all(self) -> list[AnomalyAlert]:
        """Consume ``dump_once`` if set; return an alert only on arming ranks."""
        if self._dfx_config is None:
            return []
        if not self._dfx_config.consume_dump_once():
            return []
        if not self._should_arm():
            return []
        logger.info(
            "[DFX manual_dump] dump_once consumed → alert anomaly_type=%s",
            self.anomaly_type,
        )
        return [
            AnomalyAlert(
                anomaly_type=self.anomaly_type,
                req_id=MANUAL_DUMP_REQ_ID,
                is_ill=True,
                ill_type=ILL_TYPE_NONE,
                detail={"source": "dump.dump_once"},
                skip_related_check=True,
                mark_full_log=False,
                consume_quota=False,
            )
        ]

    def _should_arm(self) -> bool:
        try:
            if not get_pp_group().is_last_rank:
                return False
        except Exception:
            return False
        use_async = bool(getattr(self._runner, "use_async_scheduling", False))
        if use_async:
            return int(getattr(self._runner, "tp_rank", 0)) == 0
        return True
