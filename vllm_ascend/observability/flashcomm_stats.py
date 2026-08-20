# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashComm per-forward accumulators and hookable gate APIs.

Custom ops are registered at import time, so ms_service_metric cannot wrap their
Python impls after the fact. Collectives therefore call into this module to
accumulate timing/bytes/failures; YAML handlers flush Prometheus once per
forward (and record decision/path counters).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Fixed decision / path / failure enums (no request_id / layer_name).
DECISION_ENABLED = "enabled"
DECISION_CONFIG_OFF = "config_off"
DECISION_DENSE_BELOW_THRESHOLD = "dense_below_threshold"
DECISION_DENSE_DRAFT = "dense_draft"
DECISION_NO_NUM_TOKENS = "no_num_tokens"

PATH_ALLREDUCE = "allreduce"
PATH_MM_RS_UNQUANT = "mm_reduce_scatter_unquant"
PATH_MM_RS_W8A8 = "mm_reduce_scatter_w8a8"
PATH_PLAIN_RS = "plain_reduce_scatter"

OP_ALL_GATHER = "all_gather"
OP_REDUCE_SCATTER = "reduce_scatter"
OP_MM_REDUCE_SCATTER = "mm_reduce_scatter"

REASON_SHAPE = "shape"
REASON_DTYPE = "dtype"
REASON_TIMEOUT = "timeout"
REASON_CONNECTION = "connection"
REASON_OTHER = "other"

FLASHCOMM_DENSE_TOKEN_THRESHOLD = 1000


@dataclass
class FlashCommForwardStats:
    decision: str | None = None
    flash_comm_v1_enabled: bool = False
    num_tokens: int = 0
    pad_size: int = 0
    all_gather_ms: float = 0.0
    reduce_scatter_ms: float = 0.0
    bytes_all_gather: int = 0
    bytes_reduce_scatter: int = 0
    bytes_mm_reduce_scatter: int = 0
    failures: dict[tuple[str, str], int] = field(default_factory=dict)

    def add_failure(self, op: str, reason: str) -> None:
        key = (op, reason)
        self.failures[key] = self.failures.get(key, 0) + 1


_current: FlashCommForwardStats | None = None


def _ensure_stats() -> FlashCommForwardStats:
    global _current
    if _current is None:
        _current = FlashCommForwardStats()
    return _current


def classify_decision(
    *,
    enable_sp: bool,
    is_moe_model: bool,
    is_draft_model: bool,
    num_tokens: int | None,
    flash_comm_v1_enabled: bool,
) -> str:
    if not enable_sp:
        return DECISION_CONFIG_OFF
    if num_tokens is None:
        return DECISION_NO_NUM_TOKENS
    if is_draft_model and not is_moe_model:
        return DECISION_DENSE_DRAFT
    if not is_moe_model and num_tokens <= FLASHCOMM_DENSE_TOKEN_THRESHOLD:
        return DECISION_DENSE_BELOW_THRESHOLD
    if flash_comm_v1_enabled:
        return DECISION_ENABLED
    return DECISION_CONFIG_OFF


def publish_forward_gate(
    *,
    decision: str,
    flash_comm_v1_enabled: bool,
    num_tokens: int | None,
    pad_size: int,
) -> FlashCommForwardStats:
    """Hookable per-forward gate publish (YAML wraps FlashCommMetricHooks.publish_forward_gate)."""
    global _current
    stats = FlashCommForwardStats(
        decision=decision,
        flash_comm_v1_enabled=bool(flash_comm_v1_enabled),
        num_tokens=int(num_tokens or 0),
        pad_size=int(pad_size or 0),
    )
    _current = stats
    return stats


def note_collective(
    op: str,
    *,
    elapsed_ms: float,
    nbytes: int = 0,
) -> None:
    """Accumulate one collective sample for the current forward."""
    stats = _ensure_stats()
    if op == OP_ALL_GATHER:
        stats.all_gather_ms += float(elapsed_ms)
        stats.bytes_all_gather += int(nbytes)
    elif op == OP_MM_REDUCE_SCATTER:
        stats.reduce_scatter_ms += float(elapsed_ms)
        stats.bytes_mm_reduce_scatter += int(nbytes)
    else:
        stats.reduce_scatter_ms += float(elapsed_ms)
        stats.bytes_reduce_scatter += int(nbytes)


def note_collective_failure(op: str, reason: str) -> None:
    """Accumulate failure; YAML wraps FlashCommMetricHooks.note_collective_failure."""
    _ensure_stats().add_failure(op, reason)


def classify_failure_reason(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return REASON_TIMEOUT
    if "shape" in text or "size" in text or "dimension" in text:
        return REASON_SHAPE
    if "dtype" in text or "type" in text:
        return REASON_DTYPE
    if "connect" in text or "hccl" in text or "comm" in text or "rank" in text:
        return REASON_CONNECTION
    return REASON_OTHER


def tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0


def snapshot_and_reset() -> FlashCommForwardStats | None:
    global _current
    stats = _current
    _current = None
    return stats


class FlashCommMetricHooks:
    """YAML-hookable entry points (must be ``Class.method`` symbols)."""

    @staticmethod
    def publish_forward_gate(
        *,
        decision: str,
        flash_comm_v1_enabled: bool,
        num_tokens: int | None,
        pad_size: int,
    ) -> FlashCommForwardStats:
        return publish_forward_gate(
            decision=decision,
            flash_comm_v1_enabled=flash_comm_v1_enabled,
            num_tokens=num_tokens,
            pad_size=pad_size,
        )

    @staticmethod
    def note_collective_failure(op: str, reason: str) -> None:
        note_collective_failure(op, reason)


class CollectiveTimer:
    """Context manager used inside custom-op impls to accumulate timing/bytes."""

    def __init__(self, op: str, tensor: Any | None = None):
        self.op = op
        self.tensor = tensor
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc is not None:
            FlashCommMetricHooks.note_collective_failure(self.op, classify_failure_reason(exc))
            return False
        note_collective(
            self.op,
            elapsed_ms=elapsed_ms,
            nbytes=tensor_nbytes(self.tensor) if self.tensor is not None else 0,
        )
        return False
