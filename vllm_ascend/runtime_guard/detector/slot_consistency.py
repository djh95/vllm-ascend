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

"""Slot-token consistency detector: block-slot meta tokens vs inference sequence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.detector.config_backed import ConfigBackedDetector
from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker
from vllm_ascend.runtime_guard.types import ILL_TYPE_NONE
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

_MAX_REPORTED_MISMATCHES = 8


class SlotConsistencyDetector(ConfigBackedDetector):
    """Verify the request's block-table slots hold the tokens being inferred.

    For every sequence position ``p`` below the current read horizon, the
    token id recorded at slot ``block_ids[p // bs] * bs + p % bs`` (stamped at
    write time by :class:`KvBlockMetaTracker`) must equal ``seq[p]`` — the
    token this request actually infers with. A mismatch means the KV living at
    that slot belongs to a different token / request: stale block reuse,
    wrong-block mapping, or cross-request contamination. Matching tokens with
    a different writer are CORRECT (prefix-cache import) and never alert.

    Metadata level only: the KV tensors themselves are verified offline
    (``verify_request_kv.py`` + dump_kv action); here we check addressing.
    """

    incident_type = "slot_consistency"
    RANK_LOCAL_DATA = True   # slot meta diverges across CP/DP ranks
    EST_COST_PER_STEP = 1.5  # step mode is O(prefix)/req/step
    section_key = "slot_consistency"

    def __init__(self, *, runtime_config: Any | None = None, runner: Any | None = None) -> None:
        super().__init__(runtime_config=runtime_config, runner=runner, enabled=False)
        # "first": check full prefix once per request (first note step);
        # "step": recheck the whole prefix every step (O(prefix)/req/step).
        self._mode = "first"
        self._checked: set[str] = set()
        if runtime_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        mode = str(getter("mode", self._mode)).lower()
        if mode not in ("first", "step"):
            logger.error("[Anomaly slot_consistency] unknown mode=%r; keeping %r", mode, self._mode)
            return
        self._mode = mode

    def clear_finished(self, req_id: str) -> None:
        self._checked.discard(req_id)

    def check_slots(
        self,
        *,
        req_id: str,
        req_idx: int | None,
        seq: list[int],
        block_ids: list[int],
        block_size: int,
        computed_before: int,
        scheduled: int,
    ) -> list[Incident]:
        """Check positions ``[0, computed_before + scheduled)`` (read horizon)."""
        if not self._precheck() or not req_id or not block_ids or block_size <= 0:
            return []
        if self._mode == "first":
            if req_id in self._checked:
                return []
            self._checked.add(req_id)
        if not seq or not self._passes_input_filter(req_id, log=False):
            if not seq and self._mode == "first":
                self._checked.discard(req_id)  # retry when snapshot has tokens
            return []
        end = min(int(computed_before) + int(scheduled), len(seq))
        if end <= 0:
            return []
        mismatches, unverified = KvBlockMetaTracker.get().find_slot_token_mismatches(
            seq,
            block_ids=block_ids,
            block_size=int(block_size),
            end_pos=end,
        )
        if not mismatches:
            logger.info_once(
                "[Anomaly slot_consistency] active mode=%s first check ok req_id=%s "
                "checked=%d unverified=%d",
                self._mode,
                req_id,
                end,
                unverified,
            )
            logger.debug(
                "[Anomaly slot_consistency] ok req_id=%s mode=%s checked=%d unverified=%d",
                req_id,
                self._mode,
                end,
                unverified,
            )
            return []
        detail = {
            "mode": self._mode,
            "checked_positions": end,
            "seq_len": len(seq),
            "unverified_slots": unverified,
            "num_mismatches": len(mismatches),
            "mismatches": mismatches[:_MAX_REPORTED_MISMATCHES],
        }
        logger.info(
            "[Anomaly slot_consistency] hit req_id=%s mode=%s mismatches=%d/%d unverified=%d "
            "first: pos=%s slot=%s expected=%s actual=%s writer=%s",
            req_id,
            self._mode,
            len(mismatches),
            end,
            unverified,
            mismatches[0]["pos"],
            mismatches[0]["slot"],
            mismatches[0]["expected_token"],
            mismatches[0]["actual_token"],
            mismatches[0]["last_writer_req_id"],
        )
        return [
            Incident(
                incident_type=self.incident_type,
                req_id=req_id,
                req_idx=req_idx,
                is_ill=True,
                ill_type=ILL_TYPE_NONE,
                detail=detail,
            )
        ]
