# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async scheduling metric handlers."""

from __future__ import annotations

import logging
import time
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    gauge_type,
    histogram_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_STALE_DISCARD = "async_scheduling:stale_output_discard_total"
_PLACEHOLDER_UNDERFLOW = "async_scheduling:placeholder_underflow_total"
_OUTPUT_PLACEHOLDERS = "async_scheduling:output_placeholders_total"
_PENDING_STRUCTURED = "async_scheduling:pending_structured_output_tokens"
_SPEC_TOKENS = "async_scheduling:spec_tokens_scheduled_total"
_D2H_SYNC_MS = "async_scheduling:d2h_sync_ms"
_SEQ_LENS_BARRIER_MS = "async_scheduling:seq_lens_barrier_ms"
_OUTPUT_QUEUE_DEPTH = "async_scheduling:async_output_queue_depth"

_ASYNC_METRICS = {
    _STALE_DISCARD: (counter_type(), []),
    _PLACEHOLDER_UNDERFLOW: (counter_type(), []),
    _OUTPUT_PLACEHOLDERS: (gauge_type(), []),
    _PENDING_STRUCTURED: (gauge_type(), []),
    _SPEC_TOKENS: (gauge_type(), []),
    _D2H_SYNC_MS: (histogram_type(), ["path"]),
    _SEQ_LENS_BARRIER_MS: (histogram_type(), ["mtp"]),
    _OUTPUT_QUEUE_DEPTH: (gauge_type(), []),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_ASYNC_METRICS, _recorder_cache, "async_scheduling")


def async_note_stale_discard_handler(original_func, *args, **kwargs):
    original_func(*args, **kwargs)
    try:
        _metrics().record_metric(_STALE_DISCARD, value=1.0, labels={})
    except Exception:
        logger.warning("Failed to record async stale discard metric", exc_info=True)


def async_note_underflow_handler(original_func, *args, **kwargs):
    original_func(*args, **kwargs)
    try:
        _metrics().record_metric(_PLACEHOLDER_UNDERFLOW, value=1.0, labels={})
    except Exception:
        logger.warning("Failed to record async placeholder underflow metric", exc_info=True)


def async_after_schedule_handler(original_func, scheduler, scheduler_output, *args, **kwargs):
    """Gauge placeholder backlog after AsyncScheduler._update_after_schedule."""
    result = original_func(scheduler, scheduler_output, *args, **kwargs)

    try:
        total_placeholders = 0
        spec_tokens_scheduled = 0
        for req_id in scheduler_output.num_scheduled_tokens:
            request = scheduler.requests.get(req_id)
            if request is None or request.is_prefill_chunk:
                continue
            total_placeholders += int(getattr(request, "num_output_placeholders", 0) or 0)
            spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            spec_tokens_scheduled += len(spec_decode_tokens.get(req_id, ()))

        metrics = _metrics()
        metrics.record_metric(_OUTPUT_PLACEHOLDERS, value=float(total_placeholders), labels={})
        metrics.record_metric(
            _PENDING_STRUCTURED,
            value=float(int(getattr(scheduler_output, "pending_structured_output_tokens", False))),
            labels={},
        )
        metrics.record_metric(_SPEC_TOKENS, value=float(spec_tokens_scheduled), labels={})
    except Exception:
        logger.warning("Failed to record async after_schedule metrics", exc_info=True)

    return result


def async_output_get_output_handler(original_func, async_output, *args, **kwargs):
    """Measure D2H copy_event.synchronize() blocking in AsyncOutput.get_output."""
    start = time.perf_counter()
    try:
        return original_func(async_output, *args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            _metrics().record_metric(
                _D2H_SYNC_MS,
                value=elapsed_ms,
                labels={"path": "async_output"},
            )
        except Exception:
            logger.warning("Failed to record async D2H sync metric", exc_info=True)


def async_seq_lens_barrier_handler(original_func, runner, scheduler_output, req_ids, *args, **kwargs):
    """Measure num_computed_tokens_event.synchronize() in v2 NPUModelRunner."""
    mtp_enabled = getattr(runner, "speculator", None) is not None
    barrier_ms = 0.0
    if mtp_enabled:
        start = time.perf_counter()
        result = original_func(runner, scheduler_output, req_ids, *args, **kwargs)
        barrier_ms = (time.perf_counter() - start) * 1000.0
    else:
        result = original_func(runner, scheduler_output, req_ids, *args, **kwargs)

    try:
        if barrier_ms > 0.0:
            _metrics().record_metric(
                _SEQ_LENS_BARRIER_MS,
                value=barrier_ms,
                labels={"mtp": "1"},
            )
    except Exception:
        logger.warning("Failed to record seq_lens barrier metric", exc_info=True)

    return result


def async_output_queue_handler(original_func, worker_proc, output, *args, **kwargs):
    """Gauge async output queue depth when async scheduling enqueues worker output."""
    result = original_func(worker_proc, output, *args, **kwargs)

    try:
        queue = getattr(worker_proc, "async_output_queue", None)
        if queue is not None and getattr(worker_proc, "use_async_scheduling", False):
            depth = float(queue.qsize())
            _metrics().record_metric(_OUTPUT_QUEUE_DEPTH, value=depth, labels={})
    except Exception:
        logger.warning("Failed to record async output queue depth metric", exc_info=True)

    return result
