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

"""Per-block KV write metadata for DFX reports (creation + last write)."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def _fmt_ts(ts: float | None) -> str | None:
    """Wall-clock stamp matching the report ``ts`` style (``%H:%M:%S.%f`` ms)."""
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) + f".{int(ts % 1 * 1000):03d}"


# Per-slot meta cap: slots = block_id * block_size + offset, bounded by total
# KV capacity; cap keeps worker memory flat (drop-oldest-half on overflow).
_SLOT_META_CAP = 1 << 18


def resolve_block_size(runner: Any) -> int:
    """Best-effort block size from the runner / vllm_config (fallback 16)."""
    bs = int(getattr(runner, "block_size", 0) or 0)
    if bs <= 0:
        cache_cfg = getattr(getattr(runner, "vllm_config", None), "cache_config", None)
        bs = int(getattr(cache_cfg, "block_size", 0) or 0)
    return bs if bs > 0 else 16


def slots_for_block_ids(block_ids: list[int], *, block_size: int) -> list[int]:
    """All slot ids covered by ``block_ids`` (block_size slots per block)."""
    if block_size <= 0:
        return []
    out: list[int] = []
    for bid in block_ids:
        base = int(bid) * int(block_size)
        out.extend(range(base, base + int(block_size)))
    return out


@dataclass(frozen=True, slots=True)
class BlockWriteViolation:
    block_id: int
    violation: str
    prev_wave: int | None
    new_wave: int
    prev_writer: str | None
    new_writer: str


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


def touched_block_ids(
    block_ids: list[int],
    *,
    block_size: int,
    num_computed_before: int,
    num_scheduled: int,
) -> list[int]:
    """Block ids whose KV slots are written in ``[computed, computed+scheduled)``."""
    if not block_ids or num_scheduled <= 0 or block_size <= 0:
        return []
    start = max(0, int(num_computed_before))
    end = start + int(num_scheduled)
    first = start // int(block_size)
    last = (end - 1) // int(block_size)
    # Do not fall back to the table tail: that can mark shared prefix blocks
    # as writes and false-trigger same_wave_writer_conflict.
    if first >= len(block_ids) or last < first:
        logger.debug(
            "touched_block_ids: write range [%s, %s) maps to block indices [%s, %s] outside table len=%s",
            start,
            end,
            first,
            last,
            len(block_ids),
        )
        return []
    last = min(last, len(block_ids) - 1)
    if last < first:
        return []
    return list(block_ids[first : last + 1])


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


class KvBlockMetaTracker:
    """Sparse per-block write meta: creation (first write) + last write.

    Per block: writer req_id, wave, and wall-clock timestamp for both the
    first write (creation) and the most recent write. Process-local.
    """

    _instance: KvBlockMetaTracker | None = None

    def __init__(self) -> None:
        self._wave: dict[int, int] = {}
        self._writer: dict[int, str] = {}
        self._first_wave: dict[int, int] = {}
        self._first_writer: dict[int, str] = {}
        self._first_ts: dict[int, float] = {}
        self._last_ts: dict[int, float] = {}
        # slot -> (writer req_id, ts, token id written at that slot). Per-slot
        # wave is deliberately not kept: block-level ``_wave`` already covers
        # step attribution, and no consumer read it.
        self._slot_meta: dict[int, tuple[str, float, int | None]] = {}

    @classmethod
    def get(cls) -> KvBlockMetaTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    def preview_write_checks(
        self,
        req_id: str,
        block_ids: list[int],
        wave: int,
        *,
        check_wave_regression: bool = True,
        check_same_wave_writer: bool = True,
    ) -> list[BlockWriteViolation]:
        """Return violations that would occur if ``record_writes`` ran (non-mutating)."""
        if not req_id or not block_ids:
            return []
        w = int(wave)
        rid = str(req_id)
        out: list[BlockWriteViolation] = []
        for bid in block_ids:
            b = int(bid)
            prev_w = self._wave.get(b)
            prev_writer = self._writer.get(b)
            if check_wave_regression and prev_w is not None and w < prev_w:
                out.append(
                    BlockWriteViolation(
                        block_id=b,
                        violation="wave_regression",
                        prev_wave=prev_w,
                        new_wave=w,
                        prev_writer=prev_writer,
                        new_writer=rid,
                    )
                )
            if (
                check_same_wave_writer
                and prev_w is not None
                and prev_w == w
                and prev_writer is not None
                and prev_writer != rid
            ):
                out.append(
                    BlockWriteViolation(
                        block_id=b,
                        violation="same_wave_writer_conflict",
                        prev_wave=prev_w,
                        new_wave=w,
                        prev_writer=prev_writer,
                        new_writer=rid,
                    )
                )
        return out

    def record_writes(self, req_id: str, block_ids: list[int], wave: int) -> None:
        if not req_id or not block_ids:
            return
        w = int(wave)
        rid = str(req_id)
        now = time.time()
        for bid in block_ids:
            b = int(bid)
            self._wave[b] = w
            self._writer[b] = rid
            self._last_ts[b] = now
            if b not in self._first_wave:
                self._first_wave[b] = w
                self._first_writer[b] = rid
                self._first_ts[b] = now

    def record_slot_writes(
        self,
        req_id: str,
        entries: list[tuple[int, int | None]],
    ) -> None:
        """Stamp per-slot last write; ``entries`` are ``(slot, token_id)`` pairs.

        ``token_id`` may be ``None`` when the sequence token for that position
        is unavailable (slot is still stamped, token left unknown). Updating an
        existing slot overwrites writer/ts/token. Insertion order is
        first-write order, so overflow eviction drops the oldest-written half.
        """
        if not req_id or not entries:
            return
        rid = str(req_id)
        now = time.time()
        for slot, token_id in entries:
            tok = int(token_id) if token_id is not None else None
            self._slot_meta[int(slot)] = (rid, now, tok)
        if len(self._slot_meta) > _SLOT_META_CAP:
            drop = len(self._slot_meta) - _SLOT_META_CAP // 2
            for k in list(self._slot_meta)[:drop]:
                del self._slot_meta[k]

    def last_write_wave(self, block_id: int) -> int | None:
        return self._wave.get(int(block_id))

    def last_writer_req_id(self, block_id: int) -> str | None:
        return self._writer.get(int(block_id))

    def blocks_detail(
        self,
        block_ids: list[int],
        *,
        include_wave: bool,
        include_writer: bool,
        include_creation: bool = False,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bid in block_ids:
            b = int(bid)
            entry: dict[str, Any] = {"block_id": b}
            if include_wave:
                entry["last_write_at"] = _fmt_ts(self._last_ts.get(b))
            if include_writer:
                entry["last_writer_req_id"] = self._writer.get(b)
            if include_creation:
                entry["created_by_req_id"] = self._first_writer.get(b)
                entry["created_at"] = _fmt_ts(self._first_ts.get(b))
            out.append(entry)
        return out

    def slots_detail(self, slots: list[int]) -> list[dict[str, Any]]:
        """Per-slot last-write entries for slots that have meta (sparse)."""
        out: list[dict[str, Any]] = []
        for s in slots:
            meta = self._slot_meta.get(int(s))
            if meta is None:
                continue
            rid, ts, tok = meta
            out.append(
                {
                    "slot": int(s),
                    "token_id": tok,
                    "last_writer_req_id": rid,
                    "last_write_at": _fmt_ts(ts),
                }
            )
        return out

    def find_slot_token_mismatches(
        self,
        seq: list[int],
        *,
        block_ids: list[int],
        block_size: int,
        end_pos: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Verify positions ``[0, end_pos)``: slot meta token vs ``seq[pos]``.

        Position ``p`` maps to slot ``block_ids[p // block_size] * block_size
        + p % block_size``. A slot whose recorded token differs from the
        sequence token at that position means the KV stored there belongs to
        another token / request (stale reuse, wrong block, contamination).
        Slots without meta (tracking off at write time / evicted) count as
        unverified and never alert. Returns ``(mismatches, unverified)``.
        """
        mismatches: list[dict[str, Any]] = []
        unverified = 0
        n_blocks = len(block_ids)
        bs = int(block_size)
        for pos in range(max(0, min(int(end_pos), len(seq)))):
            bi = pos // bs
            if bi >= n_blocks:
                break
            slot = int(block_ids[bi]) * bs + pos % bs
            m = self._slot_meta.get(slot)
            if m is None or m[2] is None:
                unverified += 1
                continue
            if m[2] != seq[pos]:
                mismatches.append(
                    {
                        "pos": pos,
                        "slot": slot,
                        "expected_token": seq[pos],
                        "actual_token": m[2],
                        "last_writer_req_id": m[0],
                        "last_write_at": _fmt_ts(m[1]),
                    }
                )
        return mismatches, unverified
