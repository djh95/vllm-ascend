# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler semantic metric handlers."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    gauge_type,
    histogram_type,
    register_metrics,
    scheduler_kind,
)
from vllm_ascend.observability.handlers.kv import count_waiting_for_remote_kvs

logger = logging.getLogger(__name__)

_SCHEDULED_TOKENS = "scheduler:scheduled_tokens"
_PAUSED_REQUESTS = "scheduler:paused_requests"
_PREEMPT_TOTAL = "scheduler:preempt_total"
_REMOTE_KV_WAITING = "scheduler:remote_kv_waiting_requests"
_PREFIX_QUERIES = "prefix_cache:connector_queries_total"
_PREFIX_HITS = "prefix_cache:connector_hits_total"

_SCHEDULER_METRICS = {
    _SCHEDULED_TOKENS: (histogram_type(), ["scheduler"]),
    _PAUSED_REQUESTS: (gauge_type(), ["scheduler"]),
    _PREEMPT_TOTAL: (counter_type(), ["scheduler", "reason"]),
    _REMOTE_KV_WAITING: (gauge_type(), ["scheduler"]),
    _PREFIX_QUERIES: (counter_type(), ["scheduler"]),
    _PREFIX_HITS: (counter_type(), ["scheduler"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_SCHEDULER_METRICS, _recorder_cache, "scheduler")


def _record_prefix_connector_stats(scheduler: Any) -> None:
    stats = getattr(scheduler, "connector_prefix_cache_stats", None)
    if stats is None:
        return
    requests = int(getattr(stats, "requests", 0) or 0)
    queries = int(getattr(stats, "queries", 0) or 0)
    hits = int(getattr(stats, "hits", 0) or 0)
    if requests == 0:
        return
    kind = scheduler_kind(scheduler)
    metrics = _metrics()
    metrics.record_metric(_PREFIX_QUERIES, value=float(queries), labels={"scheduler": kind})
    metrics.record_metric(_PREFIX_HITS, value=float(hits), labels={"scheduler": kind})


def scheduler_schedule_semantic_handler(original_func, scheduler, *args, **kwargs):
    """Record batch size, pause backlog, and connector prefix-cache stats per schedule()."""
    output = original_func(scheduler, *args, **kwargs)

    try:
        kind = scheduler_kind(scheduler)
        metrics = _metrics()
        total_tokens = int(getattr(output, "total_num_scheduled_tokens", 0) or 0)
        if total_tokens > 0:
            metrics.record_metric(
                _SCHEDULED_TOKENS,
                value=float(total_tokens),
                labels={"scheduler": kind},
            )

        paused = getattr(scheduler, "_lb_paused_req_ids", None)
        if paused is not None:
            metrics.record_metric(
                _PAUSED_REQUESTS,
                value=float(len(paused)),
                labels={"scheduler": kind},
            )

        metrics.record_metric(
            _REMOTE_KV_WAITING,
            value=float(count_waiting_for_remote_kvs(scheduler)),
            labels={"scheduler": kind},
        )
        _record_prefix_connector_stats(scheduler)
    except Exception:
        logger.warning("Failed to record scheduler schedule metrics", exc_info=True)

    return output


def scheduler_preempt_handler(original_func, scheduler, request, *args, **kwargs):
    """Count scheduler preemptions (Dyntra pause / generic preempt)."""
    result = original_func(scheduler, request, *args, **kwargs)

    try:
        reason = "preempt"
        if hasattr(scheduler, "_lb_paused_req_ids"):
            reason = "dyntra_lb"
        _metrics().record_metric(
            _PREEMPT_TOTAL,
            value=1.0,
            labels={"scheduler": scheduler_kind(scheduler), "reason": reason},
        )
    except Exception:
        logger.warning("Failed to record scheduler preempt metrics", exc_info=True)

    return result
