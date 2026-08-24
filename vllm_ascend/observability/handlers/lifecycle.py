# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle (RL ops) metric handlers."""

from __future__ import annotations

import logging
import time
from typing import Any

from vllm_ascend.observability.handlers.common import (
    gauge_type,
    histogram_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_WAKE_MS = "lifecycle:wake:duration_ms"
_UPDATE_WEIGHTS_MS = "lifecycle:update_weights:duration_ms"
_ROUTED_EXPERTS_STATE = "rl:routed_experts_state"

_LIFECYCLE_METRICS = {
    _WAKE_MS: (histogram_type(), ["sleep_opt", "tags"]),
    _UPDATE_WEIGHTS_MS: (histogram_type(), []),
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


def worker_wake_handler(original_func, worker, tags=None, *args, **kwargs):
    """Record wake-up duration (CaMem / KV / sleep-opt path)."""
    start = time.perf_counter()
    try:
        return original_func(worker, tags, *args, **kwargs)
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
    """Mark capturing=1 while MoE router replay capture runs, reset afterwards."""
    try:
        _set_routed_experts_state("capturing", 1.0)
        return original_func(capturer, layer_id, topk_ids, *args, **kwargs)
    finally:
        _set_routed_experts_state("capturing", 0.0)
