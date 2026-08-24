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
_OUTPUT_PLACEHOLDERS = "async_scheduling:output_placeholders"
_D2H_SYNC_MS = "async_scheduling:d2h_sync_ms"
_SEQ_LENS_UPDATE_MS = "async_scheduling:seq_lens_update_ms"
_OUTPUT_QUEUE_DEPTH = "async_scheduling:async_output_queue_depth"

_ASYNC_METRICS = {
    _STALE_DISCARD: (counter_type(), []),
    _PLACEHOLDER_UNDERFLOW: (counter_type(), []),
    _OUTPUT_PLACEHOLDERS: (gauge_type(), []),
    _D2H_SYNC_MS: (histogram_type(), []),
    _SEQ_LENS_UPDATE_MS: (histogram_type(), []),
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
        for req_id in scheduler_output.num_scheduled_tokens:
            request = scheduler.requests.get(req_id)
            if request is None or request.is_prefill_chunk:
                continue
            total_placeholders += int(getattr(request, "num_output_placeholders", 0) or 0)

        _metrics().record_metric(_OUTPUT_PLACEHOLDERS, value=float(total_placeholders), labels={})
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
            _metrics().record_metric(_D2H_SYNC_MS, value=elapsed_ms, labels={})
        except Exception:
            logger.warning("Failed to record async D2H sync metric", exc_info=True)


def async_seq_lens_update_handler(original_func, runner, scheduler_output, req_ids, *args, **kwargs):
    """Measure the v2 NPUModelRunner._update_seq_lens_cpu method wall time."""
    start = time.perf_counter()
    try:
        return original_func(runner, scheduler_output, req_ids, *args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            _metrics().record_metric(_SEQ_LENS_UPDATE_MS, value=elapsed_ms, labels={})
        except Exception:
            logger.warning("Failed to record seq_lens update metric", exc_info=True)


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
