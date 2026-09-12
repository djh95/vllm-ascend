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

"""Request KV location helpers for reports (``block_ids`` / ``slot_mapping``)."""

from __future__ import annotations

from contextlib import suppress
from typing import Any


def block_ids_for_request(
    runner: Any,
    req_id: str,
    req_idx: int | None = None,
    *,
    kv_cache_group: int = 0,
    input_batch: Any = None,
) -> list[int]:
    """Return logical GPU block ids for ``req_id`` (group 0 by default)."""
    if not req_id or runner is None:
        return []

    requests = getattr(runner, "requests", None)
    if requests is not None:
        state = requests.get(req_id)
        if state is not None:
            raw = getattr(state, "block_ids", None)
            parsed = _normalize_block_ids(raw, kv_cache_group=kv_cache_group)
            if parsed:
                return parsed

    if input_batch is None:
        # V2: runner.input_batch stays None; prefer execute_model_state batch
        # so report enrichment (and callers without an explicit batch) still work.
        input_batch = _runner_input_batch(runner)
    if input_batch is None:
        return []
    idx = req_idx
    if idx is None:
        mapping = getattr(input_batch, "req_id_to_index", None)
        if isinstance(mapping, dict) and req_id in mapping:
            idx = int(mapping[req_id])
        else:
            req_ids = list(getattr(input_batch, "req_ids", None) or [])
            try:
                idx = req_ids.index(req_id)
            except ValueError:
                return []
    idx = int(idx)
    table = _block_table_for_group(input_batch, kv_cache_group)
    if table is not None:
        try:
            num_blocks = int(table.num_blocks_per_row[idx])
            if num_blocks <= 0:
                return []
            row = table.block_table.np[idx, :num_blocks]
            return [int(x) for x in row.tolist()]
        except Exception:
            return []

    # ModelRunner V2 stores block rows by persistent request-state index.
    block_tables = getattr(runner, "block_tables", None)
    idx_mapping = getattr(input_batch, "idx_mapping_np", None)
    if block_tables is None or idx_mapping is None:
        return []
    try:
        state_idx = int(idx_mapping[idx])
        num_blocks = int(block_tables.num_blocks.np[kv_cache_group, state_idx])
        if num_blocks <= 0:
            return []
        # v2 rows live in StagedWriteTensor (``.gpu``, no host ``.np`` mirror):
        # prefer a host numpy mirror when present, else sync the device row.
        entry = block_tables.block_tables[kv_cache_group]
        host = getattr(entry, "np", None)
        if host is not None:
            row = host[state_idx, :num_blocks]
        else:
            row = entry.gpu[state_idx, :num_blocks].cpu()
        return [int(x) for x in row.tolist()]
    except Exception:
        return []


def slot_mapping_for_request(
    runner: Any,
    req_id: str,
    req_idx: int | None = None,
    *,
    kv_cache_group: int = 0,
    scheduler_output: Any | None = None,
) -> tuple[list[int], tuple[int, int]] | None:
    """D2H this wave's GPU ``slot_mapping`` slice for ``req_id``.

    Returns ``(values, (start, end))`` in the packed batch, or ``None`` if the
    live tensor / query span cannot be resolved. Never raises.
    """
    if not req_id or runner is None:
        return None
    try:
        batch = _runner_input_batch(runner)
        idx = _resolve_req_idx(runner, batch, req_id, req_idx)
        if idx is None:
            return None
        span = _query_span(runner, batch, idx, scheduler_output)
        if span is None:
            return None
        start, end = span
        gpu = _slot_mapping_gpu(batch, kv_cache_group)
        if gpu is None:
            return None
        values = _d2h_int_list(gpu[start:end])
        return values, (start, end)
    except Exception:
        return None


def _runner_input_batch(runner: Any) -> Any | None:
    batch = getattr(runner, "input_batch", None)
    if batch is not None:
        return batch
    state = getattr(runner, "execute_model_state", None)
    return getattr(state, "input_batch", None) if state is not None else None


def _resolve_req_idx(
    runner: Any,
    batch: Any | None,
    req_id: str,
    req_idx: int | None,
) -> int | None:
    if req_idx is not None and int(req_idx) >= 0:
        return int(req_idx)
    mapping = getattr(batch, "req_id_to_index", None) if batch is not None else None
    if isinstance(mapping, dict) and req_id in mapping:
        return int(mapping[req_id])
    req_ids = getattr(batch, "req_ids", None) if batch is not None else None
    if req_ids:
        try:
            return list(req_ids).index(req_id)
        except ValueError:
            pass
    req_states = getattr(runner, "req_states", None)
    id_map = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if isinstance(id_map, dict) and req_id in id_map:
        return int(id_map[req_id])
    return None


def _query_span(
    runner: Any,
    batch: Any | None,
    req_idx: int,
    scheduler_output: Any | None,
) -> tuple[int, int] | None:
    for qsl in (
        getattr(runner, "query_start_loc", None),
        getattr(getattr(runner, "input_buffers", None), "query_start_loc", None),
        getattr(getattr(runner, "execute_model_state", None), "query_start_loc", None),
    ):
        span = _span_from_qsl(qsl, req_idx)
        if span is not None:
            return span
    return _span_from_scheduler(batch, req_idx, scheduler_output)


def _span_from_qsl(qsl: Any, req_idx: int) -> tuple[int, int] | None:
    if qsl is None:
        return None
    arr = getattr(qsl, "np", None)
    if arr is None:
        cpu = getattr(qsl, "cpu", None)
        arr = cpu if cpu is not None else qsl
    try:
        start = int(arr[req_idx].item() if hasattr(arr[req_idx], "item") else arr[req_idx])
        nxt = arr[req_idx + 1]
        end = int(nxt.item() if hasattr(nxt, "item") else nxt)
    except Exception:
        return None
    if 0 <= start < end:
        return start, end
    return None


def _span_from_scheduler(
    batch: Any | None,
    req_idx: int,
    scheduler_output: Any | None,
) -> tuple[int, int] | None:
    if batch is None or scheduler_output is None:
        return None
    req_ids = getattr(batch, "req_ids", None)
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not req_ids or not isinstance(num_scheduled, dict):
        return None
    if req_idx < 0 or req_idx >= len(req_ids):
        return None
    start = 0
    for i, rid in enumerate(req_ids):
        n = int(num_scheduled.get(rid, 0) or 0)
        if i == req_idx:
            return (start, start + n) if n > 0 else None
        start += max(n, 0)
    return None


def _slot_mapping_gpu(input_batch: Any, kv_cache_group: int) -> Any | None:
    multi = getattr(input_batch, "block_table", None)
    if multi is None:
        return None
    slots = getattr(multi, "slot_mappings", None)
    if slots is not None:
        try:
            if int(getattr(slots, "ndim", 1) or 1) >= 2:
                return slots[int(kv_cache_group)]
            return slots
        except Exception:
            return None
    table = _block_table_for_group(input_batch, kv_cache_group)
    if table is None:
        return None
    sm = getattr(table, "slot_mapping", None)
    if sm is None:
        return None
    return getattr(sm, "gpu", sm)


def _d2h_int_list(gpu_slice: Any) -> list[int]:
    """Blocking copy of a GPU (or CPU) 1-D integer tensor to ``list[int]``."""
    if gpu_slice is None:
        return []
    if isinstance(gpu_slice, (list, tuple)):
        return [int(x) for x in gpu_slice]
    t = gpu_slice
    detach = getattr(t, "detach", None)
    if callable(detach):
        t = detach()
    to_fn = getattr(t, "to", None)
    if callable(to_fn):
        try:
            t = to_fn("cpu")
        except Exception:
            cpu_fn = getattr(t, "cpu", None)
            t = cpu_fn() if callable(cpu_fn) else t
    elif callable(getattr(t, "cpu", None)):
        t = t.cpu()
    reshape = getattr(t, "reshape", None)
    if callable(reshape):
        with suppress(Exception):
            t = reshape(-1)
    if hasattr(t, "tolist"):
        raw = t.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = [x for row in raw for x in row]
        return [int(x) for x in raw]
    return []


def _normalize_block_ids(raw: Any, *, kv_cache_group: int) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, tuple):
        if not raw or kv_cache_group >= len(raw):
            return []
        return [int(x) for x in raw[kv_cache_group]]
    if isinstance(raw, list):
        if not raw:
            return []
        if isinstance(raw[0], (list, tuple)):
            if kv_cache_group >= len(raw):
                return []
            return [int(x) for x in raw[kv_cache_group]]
        return [int(x) for x in raw]
    return []


def _block_table_for_group(input_batch: Any, kv_cache_group: int) -> Any | None:
    multi = getattr(input_batch, "block_table", None)
    if multi is None:
        return None
    tables = getattr(multi, "block_tables", None)
    if tables is not None:
        if kv_cache_group >= len(tables):
            return None
        return tables[kv_cache_group]
    try:
        return multi[kv_cache_group]
    except Exception:
        return multi if kv_cache_group == 0 else None
