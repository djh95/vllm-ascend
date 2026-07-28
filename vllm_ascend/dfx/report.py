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

"""Short anomaly reports under ``dfx/report``."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


class DfxReportWriter:
    """Append short anomaly records to a daily report file."""

    def __init__(self, report_dir: str | Path) -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        anomaly_type: str,
        req_id: str | None = None,
        detail: dict[str, Any] | None = None,
        rank_tag: str | None = None,
    ) -> Path | None:
        """Write one anomaly line. Returns report file path or None on failure."""
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            day = datetime.now().strftime("%Y%m%d")
            report_path = self.report_dir / f"anomaly_{day}.log"
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "unix_ts": round(time.time(), 3),
                "anomaly_type": anomaly_type,
                "req_id": req_id,
                "rank": rank_tag,
                "detail": detail or {},
            }
            line = json.dumps(record, ensure_ascii=False)
            with report_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            logger.info(
                "[DFX report] anomaly_type=%s req_id=%s path=%s",
                anomaly_type,
                req_id,
                report_path,
            )
            return report_path
        except Exception as exc:
            logger.error("[DFX report] write failed dir=%s error=%s", self.report_dir, exc)
            return None
