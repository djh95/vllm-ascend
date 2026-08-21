# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashComm metric handlers for MS Service Metric."""

from __future__ import annotations

import logging
from typing import Any

from ms_service_metric.provider_api import (  # type: ignore[import-not-found]
    MetricType,
    get_metric_recorder,
)

logger = logging.getLogger(__name__)

PATH_ALLREDUCE = "allreduce"
PATH_MM_RS_UNQUANT = "mm_reduce_scatter_unquant"
PATH_MM_RS_W8A8 = "mm_reduce_scatter_w8a8"
PATH_PLAIN_RS = "plain_reduce_scatter"

_DECISION_METRIC = "flashcomm:decision_total"
_PATH_METRIC = "flashcomm:path_total"
_FAILURE_METRIC = "flashcomm:collective_failures_total"
_ACTIVE_TOKENS_METRIC = "flashcomm:active_tokens"
_PADDING_RATIO_METRIC = "flashcomm:padding_ratio"
_ALL_GATHER_MS_METRIC = "flashcomm:all_gather_ms"
_REDUCE_SCATTER_MS_METRIC = "flashcomm:reduce_scatter_ms"
_BYTES_METRIC = "flashcomm:communication_bytes_total"

_recorder = None


def _counter_type():
    return getattr(MetricType, "COUNTER", MetricType.GAUGE)


def _histogram_type():
    return getattr(MetricType, "HISTOGRAM", MetricType.GAUGE)


def _metrics():
    global _recorder
    metrics = get_metric_recorder()
    if metrics is _recorder:
        return metrics

    metrics.get_or_create_metric(
        _DECISION_METRIC,
        metric_type=_counter_type(),
        label_names=["decision"],
    )
    metrics.get_or_create_metric(
        _PATH_METRIC,
        metric_type=_counter_type(),
        label_names=["path"],
    )
    metrics.get_or_create_metric(
        _FAILURE_METRIC,
        metric_type=_counter_type(),
        label_names=["op", "reason"],
    )
    metrics.get_or_create_metric(
        _ACTIVE_TOKENS_METRIC,
        metric_type=_histogram_type(),
        label_names=[],
    )
    metrics.get_or_create_metric(
        _PADDING_RATIO_METRIC,
        metric_type=_histogram_type(),
        label_names=[],
    )
    metrics.get_or_create_metric(
        _ALL_GATHER_MS_METRIC,
        metric_type=_histogram_type(),
        label_names=[],
    )
    metrics.get_or_create_metric(
        _REDUCE_SCATTER_MS_METRIC,
        metric_type=_histogram_type(),
        label_names=[],
    )
    metrics.get_or_create_metric(
        _BYTES_METRIC,
        metric_type=_counter_type(),
        label_names=["op"],
    )
    _recorder = metrics
    return metrics


def _classify_matmul_path(op_self: Any) -> str:
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        flash_comm_v1_enabled = bool(_EXTRA_CTX.flash_comm_v1_enabled)
        mmrs_fusion = bool(_EXTRA_CTX.mmrs_fusion)
    except Exception:
        return PATH_ALLREDUCE

    if not flash_comm_v1_enabled:
        return PATH_ALLREDUCE

    quant_method = getattr(getattr(op_self, "layer", None), "quant_method", None)
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        from vllm_ascend.quantization.method_adapters import AscendLinearMethod
        from vllm_ascend.quantization.methods import AscendW8A8LinearMethod
    except Exception:
        return PATH_PLAIN_RS

    if mmrs_fusion and isinstance(quant_method, UnquantizedLinearMethod):
        return PATH_MM_RS_UNQUANT
    if mmrs_fusion and (
        isinstance(quant_method, AscendLinearMethod)
        and isinstance(getattr(quant_method, "quant_method", None), AscendW8A8LinearMethod)
    ):
        return PATH_MM_RS_W8A8
    return PATH_PLAIN_RS


def flashcomm_gate_handler(original_func, *args, **kwargs):
    """Record decision Counter; keep tokens/pad on the returned stats for flush."""
    stats = original_func(*args, **kwargs)
    try:
        decision = getattr(stats, "decision", None)
        if decision:
            _metrics().record_metric(_DECISION_METRIC, value=1.0, labels={"decision": str(decision)})
    except Exception:
        logger.warning("Failed to record FlashComm decision metrics", exc_info=True)
    return stats


def flashcomm_path_handler(original_func, op_self, *args, **kwargs):
    """Per-call path Counter around SequenceRowParallelOp.matmul_and_reduce."""
    path = PATH_ALLREDUCE
    try:
        path = _classify_matmul_path(op_self)
    except Exception:
        logger.debug("FlashComm path classify failed", exc_info=True)

    try:
        result = original_func(op_self, *args, **kwargs)
    except Exception as exc:
        # FlashComm collective paths already note failures via CollectiveTimer.
        if path == PATH_ALLREDUCE:
            try:
                from vllm_ascend.observability.flashcomm_stats import (
                    FlashCommMetricHooks,
                    classify_failure_reason,
                )

                FlashCommMetricHooks.note_collective_failure("allreduce", classify_failure_reason(exc))
            except Exception:
                logger.warning("Failed to note FlashComm allreduce failure", exc_info=True)
        raise

    try:
        _metrics().record_metric(_PATH_METRIC, value=1.0, labels={"path": path})
    except Exception:
        logger.warning("Failed to record FlashComm path metrics", exc_info=True)
    return result


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
            metrics.record_metric(_ACTIVE_TOKENS_METRIC, value=float(stats.num_tokens), labels={})
            denom = max(int(stats.num_tokens), 1)
            metrics.record_metric(
                _PADDING_RATIO_METRIC,
                value=float(stats.pad_size) / float(denom),
                labels={},
            )

        metrics.record_metric(_ALL_GATHER_MS_METRIC, value=float(stats.all_gather_ms), labels={})
        metrics.record_metric(_REDUCE_SCATTER_MS_METRIC, value=float(stats.reduce_scatter_ms), labels={})

        for op, nbytes in (
            ("all_gather", stats.bytes_all_gather),
            ("reduce_scatter", stats.bytes_reduce_scatter),
            ("mm_reduce_scatter", stats.bytes_mm_reduce_scatter),
        ):
            if nbytes:
                metrics.record_metric(_BYTES_METRIC, value=float(nbytes), labels={"op": op})
    except Exception:
        logger.warning("Failed to flush FlashComm forward metrics", exc_info=True)

    return result
