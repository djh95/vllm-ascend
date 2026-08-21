# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sleep / wake / weight-update (RL ops) metric handlers."""

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

_SLEEP_MS = "lifecycle:sleep:duration_ms"
_WAKE_MS = "lifecycle:wake:duration_ms"
_SLEEP_FREED_GB = "lifecycle:sleep:freed_gb"
_UPDATE_WEIGHTS_MS = "lifecycle:update_weights:duration_ms"
_FAILURES = "lifecycle:sleep_wake_failures_total"
_ROUTED_EXPERTS_STATE = "rl:routed_experts_state"

_LIFECYCLE_METRICS = {
    _SLEEP_MS: (histogram_type(), ["level"]),
    _WAKE_MS: (histogram_type(), ["sleep_opt", "tags"]),
    _SLEEP_FREED_GB: (histogram_type(), ["level"]),
    _UPDATE_WEIGHTS_MS: (histogram_type(), []),
    _FAILURES: (counter_type(), ["op"]),
    _ROUTED_EXPERTS_STATE: (gauge_type(), ["state"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_LIFECYCLE_METRICS, _recorder_cache, "lifecycle")


def _wake_labels(worker: Any, tags: list[str] | None) -> dict[str, str]:
    sleep_opt = getattr(worker, "_observability_last_wake_sleep_opt", None)
    if sleep_opt is None:
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            rl_config = get_ascend_config().rl_config
            sleep_opt = bool(rl_config.enabled and rl_config.sleep_mode_extra_cleanup)
        except Exception:
            sleep_opt = False
    return {
        "sleep_opt": "1" if sleep_opt else "0",
        "tags": "all" if not tags else "partial",
    }


def worker_sleep_handler(original_func, worker, level=1, *args, **kwargs):
    """Record sleep duration and memory reclaimed."""
    start = time.perf_counter()
    try:
        return original_func(worker, level, *args, **kwargs)
    except Exception:
        try:
            _metrics().record_metric(_FAILURES, value=1.0, labels={"op": "sleep"})
        except Exception:
            logger.warning("Failed to record sleep failure metric", exc_info=True)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            metrics = _metrics()
            metrics.record_metric(
                _SLEEP_MS,
                value=elapsed_ms,
                labels={"level": str(level)},
            )
            freed_gb = getattr(worker, "_observability_last_sleep_freed_gb", None)
            if freed_gb is not None:
                metrics.record_metric(
                    _SLEEP_FREED_GB,
                    value=float(freed_gb),
                    labels={"level": str(level)},
                )
        except Exception:
            logger.warning("Failed to record sleep metrics", exc_info=True)


def worker_wake_handler(original_func, worker, tags=None, *args, **kwargs):
    """Record wake-up duration (CaMem / KV / sleep-opt path)."""
    start = time.perf_counter()
    try:
        return original_func(worker, tags, *args, **kwargs)
    except Exception:
        try:
            _metrics().record_metric(_FAILURES, value=1.0, labels={"op": "wake"})
        except Exception:
            logger.warning("Failed to record wake failure metric", exc_info=True)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            _metrics().record_metric(
                _WAKE_MS,
                value=elapsed_ms,
                labels=_wake_labels(worker, tags),
            )
        except Exception:
            logger.warning("Failed to record wake metrics", exc_info=True)


def worker_update_weights_handler(original_func, worker, update_info, *args, **kwargs):
    """Record HCCL/IPC weight-update chunk duration for RL."""
    start = time.perf_counter()
    try:
        return original_func(worker, update_info, *args, **kwargs)
    except Exception:
        try:
            _metrics().record_metric(_FAILURES, value=1.0, labels={"op": "update_weights"})
        except Exception:
            logger.warning("Failed to record update_weights failure metric", exc_info=True)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            _metrics().record_metric(_UPDATE_WEIGHTS_MS, value=elapsed_ms, labels={})
        except Exception:
            logger.warning("Failed to record update_weights metrics", exc_info=True)


def _set_routed_experts_state(state: str, value: float = 1.0) -> None:
    _metrics().record_metric(
        _ROUTED_EXPERTS_STATE,
        value=value,
        labels={"state": state},
    )


def routed_experts_init_handler(original_func, runner, *args, **kwargs):
    """Mark routed-experts capture enabled/ready after capturer init."""
    result = original_func(runner, *args, **kwargs)
    try:
        enabled = bool(getattr(getattr(runner, "model_config", None), "enable_return_routed_experts", False))
        ready = bool(getattr(runner, "routed_experts_initialized", False))
        _set_routed_experts_state("enabled", 1.0 if enabled else 0.0)
        _set_routed_experts_state("ready", 1.0 if ready else 0.0)
        _set_routed_experts_state("capturing", 0.0)
    except Exception:
        logger.warning("Failed to record routed experts init metrics", exc_info=True)
    return result


def routed_experts_capture_handler(original_func, capturer, layer_id, topk_ids, *args, **kwargs):
    """Mark capturing=1 while MoE router replay capture runs."""
    try:
        _set_routed_experts_state("capturing", 1.0)
    except Exception:
        logger.warning("Failed to record routed experts capturing metric", exc_info=True)
    return original_func(capturer, layer_id, topk_ids, *args, **kwargs)
