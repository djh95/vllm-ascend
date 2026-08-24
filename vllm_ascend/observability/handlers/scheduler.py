# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler semantic metric handlers."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    gauge_type,
    register_metrics,
    scheduler_kind,
)

logger = logging.getLogger(__name__)

_PAUSED_REQUESTS = "scheduler:paused_requests"
_PREEMPT_TOTAL = "scheduler:preempt_total"

_SCHEDULER_METRICS = {
    _PAUSED_REQUESTS: (gauge_type(), ["scheduler"]),
    _PREEMPT_TOTAL: (counter_type(), ["scheduler"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_SCHEDULER_METRICS, _recorder_cache, "scheduler")


def scheduler_schedule_semantic_handler(original_func, scheduler, *args, **kwargs):
    """Record the Dyntra pause backlog after each schedule()."""
    output = original_func(scheduler, *args, **kwargs)

    try:
        kind = scheduler_kind(scheduler)
        metrics = _metrics()
        paused = getattr(scheduler, "_lb_paused_req_ids", None)
        if paused is not None:
            metrics.record_metric(
                _PAUSED_REQUESTS,
                value=float(len(paused)),
                labels={"scheduler": kind},
            )
    except Exception:
        logger.warning("Failed to record scheduler schedule metrics", exc_info=True)

    return output


def scheduler_preempt_handler(original_func, scheduler, request, *args, **kwargs):
    """Count calls to the scheduler preemption hook."""
    result = original_func(scheduler, request, *args, **kwargs)

    try:
        _metrics().record_metric(
            _PREEMPT_TOTAL,
            value=1.0,
            labels={"scheduler": scheduler_kind(scheduler)},
        )
    except Exception:
        logger.warning("Failed to record scheduler preempt metrics", exc_info=True)

    return result
