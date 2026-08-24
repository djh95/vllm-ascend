# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for Ascend MS Service Metric handlers."""

from __future__ import annotations

from typing import Any

from ms_service_metric.provider_api import (  # type: ignore[import-not-found]
    MetricType,
    get_metric_recorder,
)


def counter_type() -> Any:
    return getattr(MetricType, "COUNTER", MetricType.GAUGE)


def histogram_type() -> Any:
    return getattr(MetricType, "HISTOGRAM", MetricType.GAUGE)


def gauge_type() -> Any:
    return MetricType.GAUGE


def scheduler_kind(scheduler: Any) -> str:
    return type(scheduler).__name__


def register_metrics(
    metric_specs: dict[str, tuple[Any, list[str]]],
    cache: dict[str, Any],
    cache_key: str,
) -> Any:
    """Register metric specs once per recorder instance."""
    metrics = get_metric_recorder()
    if cache.get(cache_key) is metrics:
        return metrics
    for metric_name, (metric_type, label_names) in metric_specs.items():
        metrics.get_or_create_metric(
            metric_name,
            metric_type=metric_type,
            label_names=label_names,
        )
    cache[cache_key] = metrics
    return metrics


def count_waiting_for_remote_kvs(scheduler: Any) -> int:
    try:
        from vllm.v1.request import RequestStatus

        requests = getattr(scheduler, "requests", None)
        if not requests:
            return 0
        return sum(
            1
            for request in requests.values()
            if getattr(request, "status", None) == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
    except Exception:
        return 0
