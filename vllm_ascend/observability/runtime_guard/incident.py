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

"""Incident types emitted by runtime_guard detectors.

``ILL_TYPE_*`` codes align with legacy msprobe ILLDetector category ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Align with msprobe response_anomaly ILLDetector ill_type codes.
ILL_TYPE_NONE = 0
ILL_TYPE_RARE = 1
ILL_TYPE_GARBLED = 2
ILL_TYPE_REPEAT = 3
ILL_TYPE_NAN = 4

ILL_TYPE_NAME: dict[int, str] = {
    ILL_TYPE_NONE: "none",
    ILL_TYPE_RARE: "rare",
    ILL_TYPE_GARBLED: "garbled",
    ILL_TYPE_REPEAT: "repetition",
    ILL_TYPE_NAN: "nan",
}


@dataclass(slots=True)
class Incident:
    """One runtime finding handed to the action executor."""

    incident_type: str
    req_id: str
    is_ill: bool = True
    ill_type: int = ILL_TYPE_NONE
    req_idx: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    consume_quota: bool = True
    block_ids: list[int] = field(default_factory=list)
    wave: int | None = None
    log_context: dict[str, Any] = field(default_factory=dict)

    @property
    def ill_type_name(self) -> str:
        return ILL_TYPE_NAME.get(self.ill_type, f"unknown({self.ill_type})")

    def to_report_detail(self) -> dict[str, Any]:
        out = dict(self.detail)
        if self.ill_type != ILL_TYPE_NONE:
            out.setdefault("ill_type", self.ill_type)
            out.setdefault("ill_type_name", self.ill_type_name)
        out.setdefault("is_ill", self.is_ill)
        # ``block_ids`` is emitted solely by RuntimeGuardProcessor
        # ``_enrich_detail_with_block_meta`` (always attached). Emitting it
        # here would duplicate / race the processor enrichment path.
        return out


MANUAL_TRIGGER_REQ_ID = "__manual_trigger__"
MANUAL_TRIGGER_TYPE = "manual_trigger"


def iter_local_request_rows(
    runner: Any,
    scheduler_output: Any | None = None,
) -> list[tuple[str, int]]:
    """``(req_id, req_idx)`` for local live requests (v2 req_states / batch).

    Prefer ``input_batch.req_ids``; before prepare_inputs fall back to
    ``execute_model_state.input_batch``, ``req_states``, and
    ``scheduler_output.num_scheduled_tokens`` so manual_dump can arm on the
    first real prefill wave.
    """
    input_batch = getattr(runner, "input_batch", None)
    req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
    if req_ids:
        rows = [(str(req_id), idx) for idx, req_id in enumerate(req_ids) if req_id]
        if rows:
            return rows

    state = getattr(runner, "execute_model_state", None)
    state_batch = getattr(state, "input_batch", None) if state is not None else None
    state_ids = getattr(state_batch, "req_ids", None) if state_batch is not None else None
    if state_ids:
        rows = [(str(req_id), idx) for idx, req_id in enumerate(state_ids) if req_id]
        if rows:
            return rows

    requests = getattr(runner, "requests", None)
    if isinstance(requests, dict) and requests:
        return [(str(req_id), -1) for req_id in requests if req_id]

    req_states = getattr(runner, "req_states", None)
    id_map = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if isinstance(id_map, dict) and id_map:
        return sorted(
            ((str(rid), int(idx)) for rid, idx in id_map.items() if rid),
            key=lambda item: item[1],
        )

    if scheduler_output is not None:
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if isinstance(num_scheduled, dict) and num_scheduled:
            return [(str(req_id), -1) for req_id, n_tok in num_scheduled.items() if req_id and int(n_tok or 0) > 0]
    return []


@dataclass(slots=True)
class TriggerEvent:
    """One control-plane trigger consumed from runtime_config."""

    trigger_type: str
    req_id: str
    detail: dict[str, Any] = field(default_factory=dict)
    consume_quota: bool = False

    def to_report_detail(self) -> dict[str, Any]:
        out = dict(self.detail)
        out.setdefault("source", self.trigger_type)
        return out
