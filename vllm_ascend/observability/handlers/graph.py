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

_DISPATCH = "acl_graph:dispatch_total"
_CALL_FAILURE = "acl_graph:call_failures_total"
_CACHE_ENTRIES = "acl_graph:cache_entries"

_GRAPH_METRICS = {
    _DISPATCH: (counter_type(), ["path", "mode"]),
    _CALL_FAILURE: (counter_type(), ["phase", "mode", "reason"]),
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


def _record_dispatch(metrics: Any, path: str, mode: str) -> None:
    try:
        metrics.record_metric(_DISPATCH, value=1.0, labels={"path": path, "mode": mode})
    except Exception:
        logger.warning("Failed to record ACL graph dispatch metric", exc_info=True)


def _record_call_failure(metrics: Any, phase: str, mode: str, exc: BaseException) -> None:
    try:
        metrics.record_metric(
            _CALL_FAILURE,
            value=1.0,
            labels={"phase": phase, "mode": mode, "reason": type(exc).__name__},
        )
    except Exception:
        logger.warning("Failed to record ACL graph call failure metric", exc_info=True)


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
        _record_dispatch(metrics, "eager", mode_label)
        try:
            return original_func(wrapper, *args, **kwargs)
        except Exception as exc:
            _record_call_failure(metrics, "eager", mode_label, exc)
            raise

    entry = wrapper.concrete_aclgraph_entries.get(batch_descriptor)
    phase = "capture" if (entry is None or entry.aclgraph is None) else "replay"

    try:
        _record_dispatch(metrics, phase, mode_label)
        result = original_func(wrapper, *args, **kwargs)
        if phase == "capture":
            metrics.record_metric(
                _CACHE_ENTRIES,
                value=float(len(wrapper.concrete_aclgraph_entries)),
                labels={"mode": mode_label},
            )
        return result
    except Exception as exc:
        _record_call_failure(metrics, phase, mode_label, exc)
        raise
