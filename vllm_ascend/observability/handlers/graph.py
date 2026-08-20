# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACL graph (graph mode) metric handlers."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    gauge_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_CAPTURE = "acl_graph:capture_total"
_REPLAY = "acl_graph:replay_total"
_EAGER = "acl_graph:eager_total"
_CAPTURE_FAILURE = "acl_graph:capture_failures_total"
_CACHE_ENTRIES = "acl_graph:cache_entries"

_GRAPH_METRICS = {
    _CAPTURE: (counter_type(), ["mode"]),
    _REPLAY: (counter_type(), ["mode"]),
    _EAGER: (counter_type(), ["mode"]),
    _CAPTURE_FAILURE: (counter_type(), ["mode", "reason"]),
    _CACHE_ENTRIES: (gauge_type(), ["mode"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_GRAPH_METRICS, _recorder_cache, "graph")


def _mode_label(wrapper: Any) -> str:
    runtime_mode = getattr(wrapper, "runtime_mode", None)
    if runtime_mode is None:
        return "unknown"
    return getattr(runtime_mode, "name", str(runtime_mode))


def acl_graph_call_handler(original_func, wrapper, *args, **kwargs):
    """Classify ACL graph capture / replay / eager dispatches."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    try:
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        runtime_mode = forward_context.cudagraph_runtime_mode
    except Exception:
        return original_func(wrapper, *args, **kwargs)

    mode_label = _mode_label(wrapper)
    metrics = _metrics()

    if runtime_mode == CUDAGraphMode.NONE or runtime_mode != wrapper.runtime_mode:
        try:
            metrics.record_metric(_EAGER, value=1.0, labels={"mode": mode_label})
        except Exception:
            logger.warning("Failed to record ACL graph eager metric", exc_info=True)
        return original_func(wrapper, *args, **kwargs)

    entry = wrapper.concrete_aclgraph_entries.get(batch_descriptor)
    is_capture = entry is None or entry.aclgraph is None

    try:
        if is_capture:
            result = original_func(wrapper, *args, **kwargs)
            metrics.record_metric(_CAPTURE, value=1.0, labels={"mode": mode_label})
            metrics.record_metric(
                _CACHE_ENTRIES,
                value=float(len(wrapper.concrete_aclgraph_entries)),
                labels={"mode": mode_label},
            )
            return result

        metrics.record_metric(_REPLAY, value=1.0, labels={"mode": mode_label})
        return original_func(wrapper, *args, **kwargs)
    except Exception as exc:
        try:
            reason = type(exc).__name__
            metrics.record_metric(
                _CAPTURE_FAILURE,
                value=1.0,
                labels={"mode": mode_label, "reason": reason},
            )
        except Exception:
            logger.warning("Failed to record ACL graph capture failure metric", exc_info=True)
        raise
