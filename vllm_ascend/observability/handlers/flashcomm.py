# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashComm metric handlers for MS Service Metric."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    histogram_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_DECISION_METRIC = "flashcomm:decision_total"
_FAILURE_METRIC = "flashcomm:collective_failures_total"
_ACTIVE_TOKENS_METRIC = "flashcomm:active_tokens"
_PADDING_RATIO_METRIC = "flashcomm:padding_ratio"
_BYTES_METRIC = "flashcomm:input_bytes_total"

_FLASHCOMM_METRICS = {
    _DECISION_METRIC: (counter_type(), ["decision"]),
    _FAILURE_METRIC: (counter_type(), ["op", "reason"]),
    _ACTIVE_TOKENS_METRIC: (histogram_type(), []),
    _PADDING_RATIO_METRIC: (histogram_type(), []),
    _BYTES_METRIC: (counter_type(), ["op"]),
}
_recorder_cache: dict[str, Any] = {}


def _metrics():
    return register_metrics(_FLASHCOMM_METRICS, _recorder_cache, "flashcomm")


def flashcomm_gate_handler(original_func, *args, **kwargs):
    """Record decision Counter; keep tokens/pad on the returned stats for flush."""
    stats = original_func(*args, **kwargs)
    try:
        decision = getattr(stats, "decision", None)
        if decision:
            _metrics().record_metric(
                _DECISION_METRIC,
                value=1.0,
                labels={"decision": str(decision)},
            )
    except Exception:
        logger.warning("Failed to record FlashComm decision metrics", exc_info=True)
    return stats


def flashcomm_failure_note_handler(original_func, *args, **kwargs):
    """Optional wrap around note_collective_failure to emit Counter immediately."""
    result = original_func(*args, **kwargs)
    try:
        op = args[0] if args else kwargs.get("op", "unknown")
        reason = args[1] if len(args) > 1 else kwargs.get("reason", "other")
        _metrics().record_metric(
            _FAILURE_METRIC,
            value=1.0,
            labels={"op": str(op), "reason": str(reason)},
        )
    except Exception:
        logger.warning("Failed to record FlashComm failure metrics", exc_info=True)
    return result


def flashcomm_forward_flush_handler(original_func, runner, *args, **kwargs):
    """Flush per-forward histograms/counters after execute_model."""
    from vllm_ascend.observability.flashcomm_stats import snapshot_and_reset

    try:
        result = original_func(runner, *args, **kwargs)
    except Exception:
        snapshot_and_reset()
        raise

    try:
        stats = snapshot_and_reset()
        if stats is None:
            return result

        metrics = _metrics()
        if stats.flash_comm_v1_enabled:
            metrics.record_metric(
                _ACTIVE_TOKENS_METRIC,
                value=float(stats.num_tokens),
                labels={},
            )
            denom = max(int(stats.num_tokens), 1)
            metrics.record_metric(
                _PADDING_RATIO_METRIC,
                value=float(stats.pad_size) / float(denom),
                labels={},
            )

        for op, nbytes in (
            ("all_gather", stats.bytes_all_gather),
            ("reduce_scatter", stats.bytes_reduce_scatter),
            ("mm_reduce_scatter", stats.bytes_mm_reduce_scatter),
        ):
            if nbytes:
                metrics.record_metric(
                    _BYTES_METRIC,
                    value=float(nbytes),
                    labels={"op": op},
                )
    except Exception:
        logger.warning("Failed to flush FlashComm forward metrics", exc_info=True)

    return result
