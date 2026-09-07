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

"""Lightweight KV mutation audit bus (edge hooks → KvBlockMetaTracker).

When ``report.kv_audit`` is off *and* no KV detector / block·slot report
tracking is on, every entry is a single bool check and returns.
Hooks must stay soft-fail: never raise into attention / offload / zero paths.
"""

from __future__ import annotations

from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

_enabled: bool = False
_pad_check: bool = True
_block_size: int = 16


def configure(
    *,
    enabled: bool,
    block_size: int | None = None,
    pad_check: bool = True,
) -> None:
    """Update process-local audit knobs (called from runtime_config refresh)."""
    global _enabled, _pad_check, _block_size
    _enabled = bool(enabled)
    _pad_check = bool(pad_check)
    if block_size is not None and int(block_size) > 0:
        _block_size = int(block_size)


def set_wave(wave: int) -> None:
    """No-op retained for call-site compatibility (wave tracking removed)."""
    del wave


def enabled() -> bool:
    return _enabled


def reset_for_tests() -> None:
    configure(enabled=False, block_size=16, pad_check=True)


def on_block_load(
    block_ids: list[int] | None,
    *,
    tag: str,
    num_tokens: int | None = None,
) -> None:
    """Mark blocks as loaded (H2D / PD recv / prefix).

    Do **not** use for reshape_and_cache slot scatters — those go through
    :func:`on_write_from_slot_mapping` → sequential :meth:`apply_slot_writes`.

    ``num_tokens`` (optional): total tokens covered by ``block_ids`` in order.
    The last block becomes ``SLOT_PARTIAL`` when ``num_tokens % block_size != 0``.
    """
    if not _enabled or not block_ids:
        return
    try:
        from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker

        KvBlockMetaTracker.get().on_block_load(
            [int(b) for b in block_ids],
            block_size=_block_size,
            source=str(tag),
            num_tokens=num_tokens,
        )
    except Exception:
        logger.exception("[kv_audit soft-fail] on_block_load tag=%s", tag)


def flatten_block_ids(block_ids: Any) -> list[int]:
    """Normalize scheduler ``local_block_ids`` (flat or per-group) to unique ints."""
    if block_ids is None:
        return []
    out: list[int] = []
    seen: set[int] = set()

    def _add(x: Any) -> None:
        try:
            b = int(x)
        except (TypeError, ValueError):
            return
        if b < 0 or b in seen:
            return
        seen.add(b)
        out.append(b)

    if isinstance(block_ids, (list, tuple)):
        for item in block_ids:
            if isinstance(item, (list, tuple)):
                for x in item:
                    _add(x)
            else:
                _add(item)
    else:
        try:
            for x in list(block_ids):
                _add(x)
        except TypeError:
            _add(block_ids)
    return out


def sequence_block_ids(block_ids: Any) -> list[int]:
    """Prefer first KV-cache group (sequence-aligned); else flatten unique."""
    if block_ids is None:
        return []
    if isinstance(block_ids, (list, tuple)) and block_ids:
        first = block_ids[0]
        if isinstance(first, (list, tuple)):
            out: list[int] = []
            seen: set[int] = set()
            for x in first:
                try:
                    b = int(x)
                except (TypeError, ValueError):
                    continue
                if b < 0 or b in seen:
                    continue
                seen.add(b)
                out.append(b)
            return out
    return flatten_block_ids(block_ids)


def on_pd_recv_load(
    block_ids: Any,
    *,
    num_tokens: int | None = None,
    tag: str = "pd_recv",
) -> None:
    """Mark sequence blocks after a successful PD / Mooncake receive.

    Soft-fail wrapper for connector call sites. Uses the first KV group when
    ``block_ids`` is per-group. Pass ``num_tokens`` so a partial last block
    becomes ``SLOT_PARTIAL`` instead of full ``BLOCK_SEALED``.
    """
    ids = sequence_block_ids(block_ids)
    if not ids:
        return
    nt: int | None
    if num_tokens is None:
        nt = None
    else:
        try:
            nt = int(num_tokens)
        except (TypeError, ValueError):
            nt = None
        if nt is not None and nt <= 0:
            nt = None
    on_block_load(ids, tag=str(tag), num_tokens=nt)


def on_block_copies(
    copies: Any,
    *,
    block_size: int | None = None,
) -> None:
    """Clone meta for v1 CoW ``(src, dst)`` pairs after physical block copy."""
    if not _enabled or not copies:
        return
    pairs: list[tuple[int, int]] = []
    for item in copies:
        try:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                src, dst = int(item[0]), int(item[1])
            else:
                src = int(getattr(item, "src_block_id", item[0]))  # type: ignore[index]
                dst = int(getattr(item, "dst_block_id", item[1]))  # type: ignore[index]
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        pairs.append((src, dst))
    if not pairs:
        return
    bs = int(block_size) if block_size is not None and int(block_size) > 0 else _block_size
    try:
        from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker

        KvBlockMetaTracker.get().on_block_copy(pairs, block_size=bs)
    except Exception:
        logger.exception("[kv_audit soft-fail] on_block_copies n=%s", len(pairs))


def on_invalidate(block_ids: list[int] | None, *, reason: str) -> None:
    """Drop meta after zero (or other content-clearing) mutations."""
    if not _enabled or not block_ids:
        return
    try:
        from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker

        KvBlockMetaTracker.get().invalidate(
            [int(b) for b in block_ids],
            block_size=_block_size,
        )
    except Exception:
        logger.exception("[kv_audit soft-fail] on_invalidate reason=%s", reason)


def on_write_from_slot_mapping(
    slot_mapping: Any,
    *,
    tag: str = "reshape_and_cache",
) -> None:
    """Reshape/slot scatter: sequential slot writes (``token_id=None``).

    Advances block state / ``next_offset`` without sealing whole blocks.
    Tokens are stamped later at note_kv via :meth:`merge_slot_writes`.
    Only runs when audit is enabled (accepts sync cost on that path).
    """
    if not _enabled or slot_mapping is None:
        return
    try:
        import torch

        from vllm_ascend.runtime_guard.kv_block_meta import KvBlockMetaTracker

        if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.numel() == 0:
            return
        flat = slot_mapping.detach().reshape(-1)
        host = flat.cpu().tolist()
        entries = [(int(s), None) for s in host if int(s) >= 0]
        if not entries:
            return
        # Stable unique by slot (first occurrence order) for order checks.
        seen: set[int] = set()
        uniq: list[tuple[int, int | None]] = []
        for slot, tok in entries:
            if slot in seen:
                continue
            seen.add(slot)
            uniq.append((slot, tok))
        KvBlockMetaTracker.get().apply_slot_writes(uniq, block_size=_block_size)
    except Exception:
        logger.exception("[kv_audit soft-fail] on_write_from_slot_mapping tag=%s", tag)


def check_pad_slots(slot_mapping: Any, *, where: str) -> None:
    """Log when a supposed pad region still has non-negative slot ids."""
    if not _enabled or not _pad_check or slot_mapping is None:
        return
    try:
        import torch

        if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.numel() == 0:
            return
        flat = slot_mapping.detach().reshape(-1)
        # Fast device reduce when possible; fall back to host.
        try:
            bad = int((flat >= 0).sum().item())
        except Exception:
            bad = sum(1 for s in flat.cpu().tolist() if int(s) >= 0)
        if bad > 0:
            logger.error(
                "[kv_audit] pad violation where=%s non_pad_slots=%d numel=%d "
                "(dummy/pad path should use slot_mapping=-1)",
                where,
                bad,
                int(flat.numel()),
            )
    except Exception:
        logger.exception("[kv_audit soft-fail] check_pad_slots where=%s", where)
