# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle (RL ops) metric handlers."""

from __future__ import annotations

import logging
import time
from typing import Any

from vllm_ascend.observability.handlers.common import (
    histogram_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_WAKE_MS = "lifecycle:wake:duration_ms"

_LIFECYCLE_METRICS = {
    _WAKE_MS: (histogram_type(), ["sleep_opt", "tags"]),
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
