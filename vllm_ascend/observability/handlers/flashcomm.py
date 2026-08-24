# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashComm metric handlers for MS Service Metric."""

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import counter_type, register_metrics

logger = logging.getLogger(__name__)

_DECISION_METRIC = "flashcomm:decision_total"
_FLASHCOMM_METRICS = {
    _DECISION_METRIC: (counter_type(), ["decision"]),
}
_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_FLASHCOMM_METRICS, _recorder_cache, "flashcomm")


def flashcomm_gate_handler(original_func, *args, **kwargs):
    """Record the FlashComm decision for the current forward."""
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
