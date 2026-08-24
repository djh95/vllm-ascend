# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Pool (ascend_store) metric handlers."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_POOL_LOAD_REQUESTS = "kvpool:load_requests_total"
_POOL_LOAD_ERROR_BLOCKS = "kvpool:load_error_blocks_total"
_POOL_PUT_FAILURE_KEYS = "kvpool:put_failure_keys_total"

_KV_METRICS = {
    _POOL_LOAD_REQUESTS: (counter_type(), ["mode"]),
    _POOL_LOAD_ERROR_BLOCKS: (counter_type(), []),
    _POOL_PUT_FAILURE_KEYS: (counter_type(), ["backend"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_KV_METRICS, _recorder_cache, "kvpool")


def kv_pool_start_load_handler(original_func, worker, metadata, *args, **kwargs):
    """Count KV Pool load attempts per forward."""
    mode = (
        "layerwise"
        if getattr(worker, "use_layerwise", False)
        else ("async" if getattr(worker, "load_async", False) else "sync")
    )
    request_count = len(getattr(metadata, "requests", []) or [])

    result = original_func(worker, metadata, *args, **kwargs)

    try:
        if request_count:
            _metrics().record_metric(
                _POOL_LOAD_REQUESTS,
                value=float(request_count),
                labels={"mode": mode},
            )
    except Exception:
        logger.warning("Failed to record KV Pool start_load metrics", exc_info=True)

    return result


def kv_pool_load_errors_handler(original_func, pool_worker, *args, **kwargs):
    """Count invalid blocks reported by KV Pool worker load path."""
    invalid_blocks = original_func(pool_worker, *args, **kwargs)

    try:
        if invalid_blocks:
            _metrics().record_metric(
                _POOL_LOAD_ERROR_BLOCKS,
                value=float(len(invalid_blocks)),
                labels={},
            )
    except Exception:
        logger.warning("Failed to record KV Pool load error metrics", exc_info=True)

    return invalid_blocks


def kv_pool_put_failure_handler(original_func, backend_self, keys, *args, **kwargs):
    """Count Mooncake/memcache put failures (often capacity-related)."""
    result = original_func(backend_self, keys, *args, **kwargs)

    try:
        backend_name = type(backend_self).__name__
        failed = int(getattr(backend_self, "_latest_put_failure_keys", 0) or 0)
        backend_self._latest_put_failure_keys = 0
        if failed:
            _metrics().record_metric(
                _POOL_PUT_FAILURE_KEYS,
                value=float(failed),
                labels={"backend": backend_name},
            )
    except Exception:
        logger.warning("Failed to record KV Pool put failure metrics", exc_info=True)

    return result
