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

"""Slot-token consistency invariant: block-slot meta tokens vs inference sequence.

Offset-order anomalies are a separate invariant (``invariant.kv_slot_order`` →
``kv_slot_order``); this module only emits ``kv_slot_token``.
"""

from __future__ import annotations

from typing import Any, Literal

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.invariant._filter import allow_req
from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker
from vllm_ascend.runtime_guard.types import ILL_TYPE_NONE

logger = init_logger_ascend(__name__)

INCIDENT_TYPE_TOKEN = "kv_slot_token"
INCIDENT_TYPE_ORDER = "kv_slot_order"
_MAX_REPORTED_MISMATCHES = 8


class SlotConsistencyState:
    """Per-process first-check bookkeeping (cleared on request reap)."""

    def __init__(self) -> None:
        self._checked_first: set[str] = set()

    def clear_finished(self, req_id: str) -> None:
        self._checked_first.discard(req_id)

    def clear_all(self) -> None:
        """Drop first-check bookkeeping (when slot_consistency is disabled)."""
        self._checked_first.clear()

    def check_request(
        self,
        *,
        runner: Any,
        req_id: str,
        req_idx: int | None,
        seq: list[int],
        block_ids: list[int],
        block_size: int,
        computed_before: int = 0,
        scheduled: int = 0,
        phase: Literal["first", "finish"] = "first",
    ) -> list[Incident]:
        """Check sequence prefix against slot meta for ``phase``."""
        if not req_id or not block_ids or block_size <= 0:
            return []

        phase_l = str(phase or "first").lower()
        if phase_l == "first":
            if req_id in self._checked_first:
                return []
            self._checked_first.add(req_id)

        if not seq or not allow_req(req_id, runner=runner, req_idx=req_idx):
            if not seq and phase_l == "first":
                self._checked_first.discard(req_id)
            return []

        if phase_l == "finish":
            end = len(seq)
        else:
            end = min(int(computed_before) + int(scheduled), len(seq))
        if end <= 0:
            return []

        mismatches, unverified = KvBlockMetaTracker.get().check_and_fill(
            seq,
            block_ids=block_ids,
            block_size=int(block_size),
            end_pos=end,
        )
        if not mismatches:
            logger.info_once(
                "[invariant slot_consistency] active phase=%s first check ok req_id=%s "
                "checked=%d unverified=%d",
                phase_l,
                req_id,
                end,
                unverified,
            )
            logger.debug(
                "[invariant slot_consistency] ok req_id=%s phase=%s checked=%d unverified=%d",
                req_id,
                phase_l,
                end,
                unverified,
            )
            return []

        detail = {
            "phase": phase_l,
            "checked_positions": end,
            "seq_len": len(seq),
            "unverified_slots": unverified,
            "num_mismatches": len(mismatches),
            "mismatches": mismatches[:_MAX_REPORTED_MISMATCHES],
        }
        logger.info(
            "[invariant slot_consistency] hit req_id=%s phase=%s mismatches=%d/%d "
            "unverified=%d first: pos=%s slot=%s expected=%s actual=%s",
            req_id,
            phase_l,
            len(mismatches),
            end,
            unverified,
            mismatches[0]["pos"],
            mismatches[0]["slot"],
            mismatches[0]["expected_token"],
            mismatches[0]["actual_token"],
        )
        return [
            Incident(
                incident_type=INCIDENT_TYPE_TOKEN,
                req_id=req_id,
                req_idx=req_idx,
                is_ill=True,
                ill_type=ILL_TYPE_NONE,
                detail=detail,
            )
        ]
