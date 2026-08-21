# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB metric handlers for MS Service Metric."""

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import gauge_type, register_metrics

logger = logging.getLogger(__name__)

_HOTNESS_SUMMARY_METRICS = (
    "eplb:expert_hotness:current_mean",
    "eplb:expert_hotness:current_max",
    "eplb:expert_hotness:update_mean",
    "eplb:expert_hotness:update_max",
)
_IMBALANCE_METRIC = "eplb:expert_hotness:imbalance"

_EPLB_METRICS = {
    **{name: (gauge_type(), ["rank", "phase"]) for name in _HOTNESS_SUMMARY_METRICS},
    _IMBALANCE_METRIC: (gauge_type(), ["rank", "phase", "layer"]),
}
_recorder_cache: dict[str, Any] = {}


def _register_eplb_metrics():
    return register_metrics(_EPLB_METRICS, _recorder_cache, "eplb")


def eplb_do_update_hotness_handler(original_func, worker, *args, **kwargs):
    """Record the aggregated expert hotness exposed by the EPLB rank-0 worker."""
    result = original_func(worker, *args, **kwargs)

    try:
        rank_id = getattr(worker, "rank_id", -1)
        if rank_id != 0:
            return result

        hotness = getattr(worker, "latest_expert_hotness", None)
        if not hotness:
            logger.debug("Skip EPLB hotness metrics because latest_expert_hotness is missing")
            return result

        metrics = _register_eplb_metrics()
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
    except Exception:
        logger.warning(
            "Failed to record EPLB hotness metrics; inference result is preserved",
            exc_info=True,
        )

    return result
