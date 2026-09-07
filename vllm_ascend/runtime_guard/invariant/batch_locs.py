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

"""Batch location helpers shared by invariant checks (query_start_loc / computed)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def query_start_loc_np(
    runner: Any,
    num_reqs: int,
    input_batch: Any = None,
) -> np.ndarray | None:
    if input_batch is None:
        input_batch = getattr(runner, "input_batch", None)
    if input_batch is not None:
        qsl_np = getattr(input_batch, "query_start_loc_np", None)
        if qsl_np is not None:
            try:
                return np.asarray(qsl_np[: num_reqs + 1], dtype=np.int64)
            except Exception:
                pass
    qsl = getattr(runner, "query_start_loc", None)
    if qsl is None and input_batch is not None:
        qsl = getattr(input_batch, "query_start_loc", None)
    if qsl is None:
        return None
    if hasattr(qsl, "np"):
        arr = qsl.np[: num_reqs + 1]
    elif isinstance(qsl, torch.Tensor):
        arr = qsl[: num_reqs + 1].detach().cpu().numpy()
    else:
        arr = np.asarray(qsl[: num_reqs + 1])
    return np.asarray(arr, dtype=np.int64)


def _int_from_batch(input_batch: Any, req_idx: int | None) -> int | None:
    if input_batch is None or req_idx is None:
        return None
    for attr in ("num_computed_tokens_np", "num_computed_tokens_cpu"):
        arr = getattr(input_batch, attr, None)
        if arr is None:
            continue
        try:
            return int(arr[int(req_idx)])
        except Exception:
            continue
    return None


def _int_from_requests(runner: Any, req_id: str) -> int | None:
    requests = getattr(runner, "requests", None)
    if requests is None:
        return None
    state = requests.get(req_id)
    if state is None:
        return None
    n = getattr(state, "num_computed_tokens", None)
    if n is None:
        return None
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


def num_computed_before(
    runner: Any,
    req_id: str,
    req_idx: int | None,
    scheduled: int,
    input_batch: Any = None,
) -> int | None:
    """Return wave-before ``num_computed_tokens``, or ``None`` if unknown.

    ``0`` is a valid first-prefill value — never treat it as "missing".
    Prefer ``input_batch`` over ``runner.requests``.
    At prepare-input / pre-sample time, counters are **before** this step's
    scheduled tokens. Do **not** subtract ``scheduled``.
    """
    del scheduled
    if input_batch is None:
        input_batch = getattr(runner, "input_batch", None)
    from_batch = _int_from_batch(input_batch, req_idx)
    if from_batch is not None:
        return from_batch
    return _int_from_requests(runner, req_id)
