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

"""Pre-sample logits finite (NaN/Inf) invariant check."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.invariant._filter import allow_req
from vllm_ascend.runtime_guard.invariant.batch_locs import query_start_loc_np
from vllm_ascend.runtime_guard.types import ILL_TYPE_NAN

logger = init_logger_ascend(__name__)

INCIDENT_TYPE = "logits_finite"


def req_index_for_flat_token(flat_idx: int, qsl: np.ndarray, num_reqs: int) -> int | None:
    for r in range(num_reqs):
        if int(qsl[r]) <= flat_idx < int(qsl[r + 1]):
            return r
    return None


def check_logits_finite(
    *,
    runner: Any,
    logits: torch.Tensor | None,
    logits_indices: torch.Tensor | None = None,
    input_batch: Any = None,
) -> list[Incident]:
    """Return findings for non-finite sampling logits rows (caller writes report)."""
    if logits is None or not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        return []
    if runner is None:
        return []
    try:
        row_finite = torch.isfinite(logits).all(dim=-1)
        if bool(row_finite.all().item()):
            return []
        bad_rows = (~row_finite).nonzero(as_tuple=False).flatten()
    except Exception as exc:
        logger.warning("[invariant logits_finite] check failed: %s", exc)
        return []
    if input_batch is None:
        input_batch = getattr(runner, "input_batch", None)
    req_ids = list(getattr(input_batch, "req_ids", None) or [])
    num_reqs = len(req_ids)
    qsl = query_start_loc_np(runner, num_reqs, input_batch) if num_reqs > 0 else None
    if logits_indices is None:
        logits_indices = getattr(runner, "logits_indices", None)
    idx_list: list[int] = []
    if isinstance(logits_indices, torch.Tensor):
        try:
            idx_list = [int(x) for x in logits_indices.detach().cpu().tolist()]
        except Exception:
            idx_list = []
    alerts: list[Incident] = []
    seen_req: set[str] = set()
    unresolved_rows: list[int] = []
    for row_t in bad_rows.tolist():
        row = int(row_t)
        req_id: str | None = None
        req_idx: int | None = None
        if num_reqs > 0 and row < num_reqs and not idx_list:
            req_idx = row
            req_id = req_ids[row] if row < len(req_ids) else None
        elif qsl is not None and idx_list and row < len(idx_list):
            flat = idx_list[row]
            req_idx = req_index_for_flat_token(flat, qsl, num_reqs)
            if req_idx is not None and req_idx < len(req_ids):
                req_id = req_ids[req_idx]
        elif num_reqs > 0 and row < len(req_ids):
            req_idx = row
            req_id = req_ids[row]
        if not req_id:
            unresolved_rows.append(row)
            continue
        if req_id in seen_req:
            continue
        if not allow_req(req_id, runner=runner, req_idx=req_idx):
            continue
        seen_req.add(req_id)
        finite_kind = _finite_kind_for_row(logits[row])
        alerts.append(
            Incident(
                incident_type=INCIDENT_TYPE,
                req_id=req_id,
                req_idx=req_idx,
                is_ill=True,
                ill_type=ILL_TYPE_NAN,
                detail={
                    "logits_row": row,
                    "flat_token_index": idx_list[row] if row < len(idx_list) else None,
                    "violation": "non_finite_logits",
                    "finite_kind": finite_kind,
                },
            )
        )
    if unresolved_rows:
        alerts.append(
            Incident(
                incident_type=INCIDENT_TYPE,
                req_id=None,
                req_idx=None,
                is_ill=True,
                ill_type=ILL_TYPE_NAN,
                detail={
                    "logits_rows": unresolved_rows[:16],
                    "num_unresolved_rows": len(unresolved_rows),
                    "violation": "non_finite_logits",
                    "attribution": "unresolved_row_to_request",
                    "finite_kind": _finite_kind_for_row(logits[unresolved_rows[0]]),
                },
            )
        )
    return alerts


def _finite_kind_for_row(row: torch.Tensor) -> str:
    try:
        if bool(torch.isnan(row).any().item()):
            return "nan"
        if bool(torch.isposinf(row).any().item()):
            return "pos_inf"
        if bool(torch.isneginf(row).any().item()):
            return "neg_inf"
        if bool((~torch.isfinite(row)).any().item()):
            return "non_finite"
    except Exception:
        return "non_finite"
    return "non_finite"
