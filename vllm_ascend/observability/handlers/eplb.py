# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend dynamic-EPLB metric handlers for MS Service Metric."""

from __future__ import annotations

import logging
from typing import Any

from ms_service_metric.provider_api import (  # type: ignore[import-not-found]
    MetricType,
    get_metric_recorder,
)

logger = logging.getLogger(__name__)

_HOTNESS_SUMMARY_METRICS = (
    "eplb:expert_hotness:current_mean",
    "eplb:expert_hotness:current_max",
    "eplb:expert_hotness:update_mean",
    "eplb:expert_hotness:update_max",
)
_IMBALANCE_METRIC = "eplb:expert_hotness:imbalance"
_LOAD_BALANCE_METRICS = (
    "eplb:load_balance:avg_tokens",
    "eplb:load_balance:max_tokens",
    "eplb:load_balance:balancedness",
)
_REBALANCE_RESULT_METRIC = "eplb:rebalance:result"
_MAP_CONSISTENCY_METRICS = (
    "eplb:map_consistency:fallback_layers",
    "eplb:map_consistency:duplicate_count",
    "eplb:map_consistency:missing_count",
    "eplb:map_consistency:num_valid_experts",
)
_TRANSFER_METRICS = (
    "eplb:transfer:send_experts",
    "eplb:transfer:recv_experts",
    "eplb:transfer:comm_ops",
    "eplb:transfer:est_bytes",
)
_ASYNC_WORKER_METRICS = (
    "eplb:async_worker:alive",
    "eplb:async_worker:pending_layers",
    "eplb:async_worker:seconds_since_progress",
    "eplb:async_worker:cur_iterations",
)

_eplb_hotness_recorder = None
_eplb_rebalance_recorder = None
_eplb_transfer_recorder = None
_eplb_async_recorder = None


def _counter_type():
    return getattr(MetricType, "COUNTER", MetricType.GAUGE)


def _register_hotness_metrics():
    global _eplb_hotness_recorder

    metrics = get_metric_recorder()
    if metrics is _eplb_hotness_recorder:
        return metrics

    for metric_name in _HOTNESS_SUMMARY_METRICS:
        metrics.get_or_create_metric(
            metric_name,
            metric_type=MetricType.GAUGE,
            label_names=["rank", "phase"],
        )
    metrics.get_or_create_metric(
        _IMBALANCE_METRIC,
        metric_type=MetricType.GAUGE,
        label_names=["rank", "phase", "layer"],
    )
    _eplb_hotness_recorder = metrics
    return metrics


def _register_rebalance_metrics():
    global _eplb_rebalance_recorder

    metrics = get_metric_recorder()
    if metrics is _eplb_rebalance_recorder:
        return metrics

    for metric_name in _LOAD_BALANCE_METRICS:
        metrics.get_or_create_metric(
            metric_name,
            metric_type=MetricType.GAUGE,
            label_names=["rank"],
        )
    metrics.get_or_create_metric(
        _REBALANCE_RESULT_METRIC,
        metric_type=_counter_type(),
        label_names=["rank", "result", "policy_type"],
    )
    for metric_name in _MAP_CONSISTENCY_METRICS:
        metrics.get_or_create_metric(
            metric_name,
            metric_type=MetricType.GAUGE,
            label_names=["rank"],
        )
    _eplb_rebalance_recorder = metrics
    return metrics


def _register_transfer_metrics():
    global _eplb_transfer_recorder

    metrics = get_metric_recorder()
    if metrics is _eplb_transfer_recorder:
        return metrics

    for metric_name in _TRANSFER_METRICS:
        metrics.get_or_create_metric(
            metric_name,
            metric_type=MetricType.GAUGE,
            label_names=["rank", "layer"],
        )
    _eplb_transfer_recorder = metrics
    return metrics


def _register_async_worker_metrics():
    global _eplb_async_recorder

    metrics = get_metric_recorder()
    if metrics is _eplb_async_recorder:
        return metrics

    for metric_name in _ASYNC_WORKER_METRICS:
        metrics.get_or_create_metric(
            metric_name,
            metric_type=MetricType.GAUGE,
            label_names=["rank"],
        )
    _eplb_async_recorder = metrics
    return metrics


def eplb_do_update_hotness_handler(original_func, worker, *args, **kwargs):
    """Record hotness, load balance, rebalance result, and map consistency."""
    result = original_func(worker, *args, **kwargs)

    try:
        rank_id = getattr(worker, "rank_id", -1)
        rank_label = {"rank": str(rank_id)}

        rebalance = getattr(worker, "latest_rebalance_result", None)
        if rebalance:
            metrics = _register_rebalance_metrics()
            metrics.record_metric(
                _REBALANCE_RESULT_METRIC,
                value=1.0,
                labels={
                    "rank": str(rank_id),
                    "result": str(rebalance.get("result", "unknown")),
                    "policy_type": str(rebalance.get("policy_type", -1)),
                },
            )

        map_consistency = getattr(worker, "latest_map_consistency", None)
        if map_consistency:
            metrics = _register_rebalance_metrics()
            for key in (
                "fallback_layers",
                "duplicate_count",
                "missing_count",
                "num_valid_experts",
            ):
                value = map_consistency.get(key)
                if value is not None:
                    metrics.record_metric(
                        f"eplb:map_consistency:{key}",
                        value=float(value),
                        labels=rank_label,
                    )

        if int(rank_id) != 0:
            return result

        hotness = getattr(worker, "latest_expert_hotness", None)
        if hotness:
            metrics = _register_hotness_metrics()
            rank_labels = {"rank": str(rank_id), "phase": "all"}
            for suffix in ("current_mean", "current_max", "update_mean", "update_max"):
                value = hotness.get(suffix)
                if value is not None:
                    metrics.record_metric(
                        f"eplb:expert_hotness:{suffix}",
                        value=float(value),
                        labels=rank_labels,
                    )

            for phase, key in (
                ("current", "current_imbalance_list"),
                ("update", "update_imbalance_list"),
            ):
                imbalance_values: Any = hotness.get(key)
                if imbalance_values is None:
                    continue
                for layer_index, value in enumerate(imbalance_values):
                    metrics.record_metric(
                        _IMBALANCE_METRIC,
                        value=float(value),
                        labels={
                            "rank": str(rank_id),
                            "phase": phase,
                            "layer": str(layer_index),
                        },
                    )
        else:
            logger.debug("Skip EPLB hotness metrics because latest_expert_hotness is missing")

        load_balance = getattr(worker, "latest_load_balance", None)
        if load_balance:
            metrics = _register_rebalance_metrics()
            for key in ("avg_tokens", "max_tokens", "balancedness"):
                value = load_balance.get(key)
                if value is not None:
                    metrics.record_metric(
                        f"eplb:load_balance:{key}",
                        value=float(value),
                        labels=rank_label,
                    )
    except Exception:
        logger.warning(
            "Failed to record EPLB do_update metrics; inference result is preserved",
            exc_info=True,
        )

    return result


def eplb_transfer_stats_handler(original_func, loader, *args, **kwargs):
    """Record expert migration volume after D2D task generation."""
    result = original_func(loader, *args, **kwargs)

    try:
        stats = getattr(loader, "latest_transfer_stats", None)
        if not stats:
            return result

        metrics = _register_transfer_metrics()
        labels = {
            "rank": str(getattr(getattr(loader, "comm_group", None), "rank_in_group", -1)),
            "layer": str(stats.get("layer_id", -1)),
        }
        for key in ("send_experts", "recv_experts", "comm_ops", "est_bytes"):
            value = stats.get(key)
            if value is not None:
                metrics.record_metric(
                    f"eplb:transfer:{key}",
                    value=float(value),
                    labels=labels,
                )
    except Exception:
        logger.warning(
            "Failed to record EPLB transfer metrics; inference result is preserved",
            exc_info=True,
        )

    return result


def eplb_async_worker_status_handler(original_func, updator, *args, **kwargs):
    """Record async EPLB worker liveness and progress after each forward_end."""
    result = original_func(updator, *args, **kwargs)

    try:
        status = getattr(updator, "latest_async_worker_status", None)
        if not status:
            return result

        metrics = _register_async_worker_metrics()
        labels = {"rank": str(getattr(updator, "rank_id", -1))}
        metric_keys = (
            ("worker_alive", "eplb:async_worker:alive"),
            ("pending_layers", "eplb:async_worker:pending_layers"),
            ("seconds_since_progress", "eplb:async_worker:seconds_since_progress"),
            ("cur_iterations", "eplb:async_worker:cur_iterations"),
        )
        for status_key, metric_name in metric_keys:
            value = status.get(status_key)
            if value is not None:
                metrics.record_metric(
                    metric_name,
                    value=float(value),
                    labels=labels,
                )
    except Exception:
        logger.warning(
            "Failed to record EPLB async worker metrics; inference result is preserved",
            exc_info=True,
        )

    return result
