# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor / ModelRunner post-forward metric handlers."""

from __future__ import annotations

import logging
import time
from typing import Any

from vllm_ascend.observability.handlers.common import (
    histogram_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_SAMPLE_MS = "executor:sample_tokens:duration_ms"

_EXECUTOR_METRICS = {
    _SAMPLE_MS: (histogram_type(), ["runner"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_EXECUTOR_METRICS, _recorder_cache, "executor")


def sample_tokens_duration_handler(original_func, runner, *args, **kwargs):
    """Wall-clock duration for post-forward sampling (rejection / bonus tokens)."""
    start = time.perf_counter()
    try:
        return original_func(runner, *args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            _metrics().record_metric(
                _SAMPLE_MS,
                value=elapsed_ms,
                labels={"runner": type(runner).__name__},
            )
        except Exception:
            logger.warning("Failed to record sample_tokens duration metrics", exc_info=True)
