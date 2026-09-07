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

"""Per-block KV slot meta state machine for DFX reports and invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


# Per-slot meta cap: slots = block_id * block_size + offset, bounded by total
# KV capacity; cap keeps worker memory flat (drop-oldest-half on overflow).
_SLOT_META_CAP = 1 << 18


class BlockState(IntEnum):
    UNKNOWN = 0
    SLOT_PARTIAL = 1
    # Sequential slot writes filled 0‥block_size-1.
    BLOCK_SEALED_FILL = 2
    # Whole-block H2D / PD / prefix load (may lack per-slot tokens).
    BLOCK_SEALED_LOAD = 3
    # Alias kept for older call sites / docs (same as FILL).
    BLOCK_SEALED = 2


_BLOCK_STATE_NAMES = {
    BlockState.UNKNOWN: "UNKNOWN",
    BlockState.SLOT_PARTIAL: "SLOT_PARTIAL",
    BlockState.BLOCK_SEALED_FILL: "BLOCK_SEALED_FILL",
    BlockState.BLOCK_SEALED_LOAD: "BLOCK_SEALED_LOAD",
}


def is_sealed(state: BlockState) -> bool:
    return state in (BlockState.BLOCK_SEALED_FILL, BlockState.BLOCK_SEALED_LOAD)


@dataclass
class _BlockMeta:
    state: BlockState = BlockState.UNKNOWN
    next_offset: int = 0
    source: str | None = None


@dataclass(frozen=True, slots=True)
class SlotOrderViolation:
    block_id: int
    violation: str
    expected_offset: int
    offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SealedRewriteViolation:
    """Sealed block then a slot batch that does not start at offset 0."""

    block_id: int
    violation: str
    offsets: tuple[int, ...]
    prev_source: str | None
    # Parallel to ``offsets`` when at least one token is known; else None.
    token_ids: tuple[int | None, ...] | None
    prev_state_name: str | None = None

@dataclass
class SlotWriteFindings:
    order: list[SlotOrderViolation] = field(default_factory=list)
    state: list[SealedRewriteViolation] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.order) or bool(self.state)

    def extend(self, other: SlotWriteFindings) -> None:
        self.order.extend(other.order)
        self.state.extend(other.state)


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
    # as writes and false-trigger order violations.
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
        from contextlib import suppress

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
    """Sparse per-block slot meta state machine. Process-local."""

    _instance: KvBlockMetaTracker | None = None

    def __init__(self) -> None:
        self._blocks: dict[int, _BlockMeta] = {}
        self._slot_tokens: dict[int, int] = {}

    @classmethod
    def get(cls) -> KvBlockMetaTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    def clear(self) -> None:
        """Drop all block / slot meta (call when KV tracking is disabled)."""
        self._blocks.clear()
        self._slot_tokens.clear()

    def _block(self, block_id: int) -> _BlockMeta:
        b = int(block_id)
        meta = self._blocks.get(b)
        if meta is None:
            meta = _BlockMeta()
            self._blocks[b] = meta
        return meta

    def _clear_block_slots(self, block_id: int, *, block_size: int) -> None:
        if block_size <= 0:
            return
        base = int(block_id) * int(block_size)
        for slot in range(base, base + int(block_size)):
            self._slot_tokens.pop(slot, None)

    def _degrade_sealed(self, block_id: int, *, block_size: int) -> None:
        meta = self._block(block_id)
        self._clear_block_slots(block_id, block_size=block_size)
        meta.state = BlockState.SLOT_PARTIAL
        meta.next_offset = 0

    def _block_is_physically_full(self, block_id: int, *, block_size: int) -> bool:
        bs = int(block_size)
        if bs <= 0:
            return False
        meta = self._block(block_id)
        if meta.next_offset >= bs:
            return True
        base = int(block_id) * bs
        return all((base + off) in self._slot_tokens for off in range(bs))

    def _finalize_block_state(self, block_id: int, *, block_size: int) -> None:
        meta = self._block(block_id)
        if self._block_is_physically_full(block_id, block_size=block_size):
            meta.state = BlockState.BLOCK_SEALED_FILL
            meta.next_offset = int(block_size)
        elif meta.next_offset > 0 or any(
            slot // int(block_size) == int(block_id) for slot in self._slot_tokens
        ):
            meta.state = BlockState.SLOT_PARTIAL

    def _store_slot_token(self, slot: int, token_id: int | None) -> None:
        if token_id is None:
            return
        self._slot_tokens[int(slot)] = int(token_id)

    def _evict_slot_tokens_if_needed(self) -> None:
        if len(self._slot_tokens) <= _SLOT_META_CAP:
            return
        drop = len(self._slot_tokens) - _SLOT_META_CAP // 2
        for k in list(self._slot_tokens)[:drop]:
            del self._slot_tokens[k]

    def on_block_copy(
        self,
        copies: list[tuple[int, int]],
        *,
        block_size: int,
    ) -> None:
        """Clone ledger state + known slot tokens for CoW ``src → dst`` pairs."""
        bs = int(block_size)
        if bs <= 0 or not copies:
            return
        for src_raw, dst_raw in copies:
            src = int(src_raw)
            dst = int(dst_raw)
            if src == dst:
                continue
            self._clear_block_slots(dst, block_size=bs)
            src_meta = self._blocks.get(src)
            if src_meta is None:
                dst_meta = self._blocks.get(dst)
                if dst_meta is not None:
                    dst_meta.state = BlockState.UNKNOWN
                    dst_meta.next_offset = 0
                    dst_meta.source = None
                continue
            dst_meta = self._block(dst)
            dst_meta.state = src_meta.state
            dst_meta.next_offset = int(src_meta.next_offset)
            dst_meta.source = src_meta.source
            base_s = src * bs
            base_d = dst * bs
            for off in range(bs):
                tok = self._slot_tokens.get(base_s + off)
                if tok is not None:
                    self._slot_tokens[base_d + off] = tok
        self._evict_slot_tokens_if_needed()

    def on_block_load(
        self,
        block_ids: list[int],
        *,
        block_size: int,
        source: str | None = None,
        num_tokens: int | None = None,
    ) -> None:
        """Mark blocks as loaded (prefix cache / H2D / PD recv).

        When ``num_tokens`` is None, every block is treated as fully filled
        (``BLOCK_SEALED_LOAD``). When set, blocks before the last are load-sealed;
        the last block uses ``num_tokens % block_size`` as ``next_offset``
        (``SLOT_PARTIAL`` if partial, ``BLOCK_SEALED_LOAD`` if the remainder is 0).
        """
        bs = int(block_size)
        if bs <= 0 or not block_ids:
            return
        ids = [int(b) for b in block_ids]
        n = len(ids)
        last_partial = 0
        if num_tokens is not None:
            nt = max(0, int(num_tokens))
            last_partial = nt % bs
        for i, b in enumerate(ids):
            self._clear_block_slots(b, block_size=bs)
            meta = self._block(b)
            is_last = i == n - 1
            if (
                num_tokens is not None
                and is_last
                and int(num_tokens) > 0
                and last_partial != 0
            ):
                meta.state = BlockState.SLOT_PARTIAL
                meta.next_offset = last_partial
            else:
                meta.state = BlockState.BLOCK_SEALED_LOAD
                meta.next_offset = bs
            if source is not None:
                meta.source = str(source)
    def invalidate(
        self,
        block_ids: list[int],
        *,
        block_size: int = 0,
    ) -> None:
        """Drop block/slot meta after zero or other content-clearing mutations."""
        if not block_ids:
            return
        bs = int(block_size) if block_size and block_size > 0 else 0
        for bid in block_ids:
            b = int(bid)
            if bs > 0:
                self._clear_block_slots(b, block_size=bs)
            meta = self._blocks.get(b)
            if meta is not None:
                meta.state = BlockState.UNKNOWN
                meta.next_offset = 0
                meta.source = None

    def apply_slot_writes(
        self,
        entries: list[tuple[int, int | None]],
        *,
        block_size: int,
        alert_sealed_nonzero: bool = True,
    ) -> SlotWriteFindings:
        """Apply sequential slot writes; return order / sealed-rewrite findings.

        ``BLOCK_SEALED_LOAD`` + start offset 0 → silent degrade then apply.
        ``BLOCK_SEALED_LOAD`` + start offset > 0 → degrade, prime cursor, apply;
        alert when ``alert_sealed_nonzero`` (gated by caller for ``output_len==0``).

        ``BLOCK_SEALED_FILL`` + start offset 0 → silent degrade then apply (reuse).
        ``BLOCK_SEALED_FILL`` + start offset > 0 → always emit finding and **skip**
        the batch (true anomaly after sequential fill).
        """
        findings = SlotWriteFindings()
        bs = int(block_size)
        if bs <= 0 or not entries:
            return findings

        by_block: dict[int, list[tuple[int, int | None]]] = {}
        for slot, token_id in entries:
            s = int(slot)
            bid = s // bs
            off = s % bs
            by_block.setdefault(bid, []).append((off, token_id))

        for bid, batch in by_block.items():
            batch.sort(key=lambda x: x[0])
            offsets = [off for off, _ in batch]
            meta = self._block(bid)
            if meta.state == BlockState.BLOCK_SEALED_FILL:
                if offsets[0] != 0:
                    toks = tuple(tok for _, tok in batch)
                    findings.state.append(
                        SealedRewriteViolation(
                            block_id=bid,
                            violation="sealed_fill_nonzero_rewrite",
                            offsets=tuple(offsets),
                            prev_source=meta.source,
                            token_ids=toks if any(t is not None for t in toks) else None,
                            prev_state_name="BLOCK_SEALED_FILL",
                        )
                    )
                    continue
                self._degrade_sealed(bid, block_size=bs)
                meta = self._block(bid)
            elif meta.state == BlockState.BLOCK_SEALED_LOAD:
                prev_source = meta.source
                if offsets[0] != 0 and alert_sealed_nonzero:
                    toks = tuple(tok for _, tok in batch)
                    findings.state.append(
                        SealedRewriteViolation(
                            block_id=bid,
                            violation="sealed_nonzero_rewrite",
                            offsets=tuple(offsets),
                            prev_source=prev_source,
                            token_ids=toks if any(t is not None for t in toks) else None,
                            prev_state_name="BLOCK_SEALED_LOAD",
                        )
                    )
                self._degrade_sealed(bid, block_size=bs)
                meta = self._block(bid)
                if offsets[0] != 0:
                    meta.next_offset = int(offsets[0])
                    meta.state = BlockState.SLOT_PARTIAL

            expected_start = 0 if meta.state == BlockState.UNKNOWN else meta.next_offset
            if offsets[0] != expected_start:
                findings.order.append(
                    SlotOrderViolation(
                        block_id=bid,
                        violation="wrong_start",
                        expected_offset=expected_start,
                        offsets=tuple(offsets),
                    )
                )
                continue

            prev = offsets[0] - 1
            gap = False
            for off in offsets:
                if off != prev + 1:
                    findings.order.append(
                        SlotOrderViolation(
                            block_id=bid,
                            violation="gap",
                            expected_offset=prev + 1,
                            offsets=tuple(offsets),
                        )
                    )
                    gap = True
                    break
                prev = off
            if gap:
                continue
            for off, token_id in batch:
                slot = bid * bs + off
                self._store_slot_token(slot, token_id)
            meta.next_offset = offsets[-1] + 1
            if meta.state == BlockState.UNKNOWN:
                meta.state = BlockState.SLOT_PARTIAL
            self._finalize_block_state(bid, block_size=bs)

        self._evict_slot_tokens_if_needed()
        return findings
    def stamp_slot_tokens(self, entries: list[tuple[int, int | None]]) -> None:
        """Overwrite token ids for already-accounted slots (no order / state change)."""
        if not entries:
            return
        for slot, token_id in entries:
            self._store_slot_token(int(slot), token_id)
        self._evict_slot_tokens_if_needed()

    def merge_slot_writes(
        self,
        entries: list[tuple[int, int | None]],
        *,
        block_size: int,
        alert_sealed_nonzero: bool = True,
    ) -> SlotWriteFindings:
        """Apply new sequential writes, or stamp tokens if offsets already advanced.

        Used when reshape_and_cache may have recorded the same offsets with
        ``token_id=None`` before note_kv refreshes tokens from the sequence.
        """
        findings = SlotWriteFindings()
        bs = int(block_size)
        if bs <= 0 or not entries:
            return findings

        by_block: dict[int, list[tuple[int, int | None]]] = {}
        for slot, token_id in entries:
            s = int(slot)
            by_block.setdefault(s // bs, []).append((s % bs, token_id))

        apply_batch: list[tuple[int, int | None]] = []
        stamp_batch: list[tuple[int, int | None]] = []
        for bid, batch in by_block.items():
            batch.sort(key=lambda x: x[0])
            offsets = [off for off, _ in batch]
            meta = self._blocks.get(bid)
            next_off = 0 if meta is None else int(meta.next_offset)
            state = BlockState.UNKNOWN if meta is None else meta.state
            if is_sealed(state):
                apply_batch.extend((bid * bs + off, tok) for off, tok in batch)
                continue
            expected_start = 0 if state == BlockState.UNKNOWN else next_off
            if offsets[0] == expected_start:
                # Let apply_slot_writes validate contiguity (gap → violation).
                apply_batch.extend((bid * bs + off, tok) for off, tok in batch)
                continue
            if next_off > 0 and offsets[-1] < next_off:
                stamp_batch.extend((bid * bs + off, tok) for off, tok in batch)
                continue
            findings.order.append(
                SlotOrderViolation(
                    block_id=bid,
                    violation="wrong_start",
                    expected_offset=expected_start,
                    offsets=tuple(offsets),
                )
            )
        if apply_batch:
            findings.extend(
                self.apply_slot_writes(
                    apply_batch,
                    block_size=bs,
                    alert_sealed_nonzero=alert_sealed_nonzero,
                )
            )
        if stamp_batch:
            self.stamp_slot_tokens(stamp_batch)
        return findings

    def check_and_fill(
        self,
        seq: list[int],
        *,
        block_ids: list[int],
        block_size: int,
        end_pos: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Verify ``[0, end_pos)`` against slot meta; fill missing slots unverified.

        Returns ``(token_mismatches, fill_unverified_count)``.

        Missing tokens on already-advanced offsets (e.g. reshape wrote
        ``token_id=None``) are stamped in place. Sealed empty blocks (H2D /
        PD) degrade via :meth:`apply_slot_writes` and rebuild sequentially.
        """
        mismatches: list[dict[str, Any]] = []
        fill_unverified = 0
        n_blocks = len(block_ids)
        bs = int(block_size)
        if bs <= 0 or n_blocks <= 0:
            return mismatches, fill_unverified

        end = max(0, min(int(end_pos), len(seq)))
        for pos in range(end):
            bi = pos // bs
            if bi >= n_blocks:
                break
            bid = int(block_ids[bi])
            off = pos % bs
            slot = bid * bs + off
            tok = self._slot_tokens.get(slot)
            if tok is None:
                fill_unverified += 1
                meta = self._blocks.get(bid)
                next_off = 0 if meta is None else int(meta.next_offset)
                state = BlockState.UNKNOWN if meta is None else meta.state
                if not is_sealed(state) and off < next_off:
                    self._store_slot_token(slot, seq[pos])
                else:
                    self.apply_slot_writes([(slot, seq[pos])], block_size=bs)
                continue
            if tok != seq[pos]:
                mismatches.append(
                    {
                        "pos": pos,
                        "slot": slot,
                        "expected_token": seq[pos],
                        "actual_token": tok,
                    }
                )
        self._evict_slot_tokens_if_needed()
        return mismatches, fill_unverified

    def block_state(self, block_id: int) -> BlockState:
        meta = self._blocks.get(int(block_id))
        return BlockState.UNKNOWN if meta is None else meta.state

    def slot_token(self, slot: int) -> int | None:
        return self._slot_tokens.get(int(slot))

    def blocks_detail(self, block_ids: list[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bid in block_ids:
            b = int(bid)
            meta = self._blocks.get(b)
            state = BlockState.UNKNOWN if meta is None else meta.state
            entry: dict[str, Any] = {
                "block_id": b,
                "state": int(state),
                "state_name": _BLOCK_STATE_NAMES[state],
                "next_offset": 0 if meta is None else int(meta.next_offset),
            }
            if meta is not None and meta.source is not None:
                entry["source"] = meta.source
            out.append(entry)
        return out

    def slots_detail(self, slots: list[int]) -> list[dict[str, Any]]:
        """Per-slot token entries for slots that have meta (sparse)."""
        out: list[dict[str, Any]] = []
        for s in slots:
            tok = self._slot_tokens.get(int(s))
            if tok is None:
                continue
            out.append({"slot": int(s), "token_id": tok})
        return out
