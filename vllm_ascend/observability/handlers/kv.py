# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV transfer (PD) and KV Pool metric handlers."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    gauge_type,
    register_metrics,
    scheduler_kind,
)

logger = logging.getLogger(__name__)

_INVALID_BLOCKS = "kv:invalid_blocks_total"
_LOAD_FAILURE_EVENTS = "kv:load_failure_events_total"
_XFER_FINISHED = "kv:xfer_finished_total"
_XFER_FAILED_REQUESTS = "kv:xfer_failed_requests"
_REMOTE_KV_WAITING = "kv:remote_kv_waiting_requests"
_POOL_LOAD_REQUESTS = "kvpool:load_requests_total"
_POOL_LOAD_ERROR_BLOCKS = "kvpool:load_error_blocks_total"
_POOL_PUT_FAILURE_KEYS = "kvpool:put_failure_keys_total"

_KV_METRICS = {
    _INVALID_BLOCKS: (counter_type(), ["connector"]),
    _LOAD_FAILURE_EVENTS: (counter_type(), ["connector"]),
    _XFER_FINISHED: (counter_type(), ["connector"]),
    _XFER_FAILED_REQUESTS: (gauge_type(), ["connector"]),
    _REMOTE_KV_WAITING: (gauge_type(), ["scheduler"]),
    _POOL_LOAD_REQUESTS: (counter_type(), ["mode"]),
    _POOL_LOAD_ERROR_BLOCKS: (counter_type(), []),
    _POOL_PUT_FAILURE_KEYS: (counter_type(), ["backend"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_KV_METRICS, _recorder_cache, "kv")


def _connector_label(scheduler: Any) -> str:
    connector = getattr(scheduler, "connector", None)
    if connector is None:
        return "none"
    return type(connector).__name__


def kv_update_from_output_handler(original_func, scheduler, scheduler_output, model_runner_output, *args, **kwargs):
    """Record KV load failures and remote-KV backlog after scheduler output processing."""
    kv_out = getattr(model_runner_output, "kv_connector_output", None)
    invalid_block_count = 0
    if kv_out is not None and getattr(kv_out, "invalid_block_ids", None):
        invalid_block_count = len(kv_out.invalid_block_ids)

    result = original_func(scheduler, scheduler_output, model_runner_output, *args, **kwargs)

    try:
        connector = _connector_label(scheduler)
        metrics = _metrics()
        if invalid_block_count:
            metrics.record_metric(
                _INVALID_BLOCKS,
                value=float(invalid_block_count),
                labels={"connector": connector},
            )
            metrics.record_metric(
                _LOAD_FAILURE_EVENTS,
                value=1.0,
                labels={"connector": connector},
            )

        failed_recving = getattr(scheduler, "failed_recving_kv_req_ids", None)
        if failed_recving is not None:
            metrics.record_metric(
                _XFER_FAILED_REQUESTS,
                value=float(len(failed_recving)),
                labels={"connector": connector},
            )

        waiting_remote = count_waiting_for_remote_kvs(scheduler)
        metrics.record_metric(
            _REMOTE_KV_WAITING,
            value=float(waiting_remote),
            labels={"scheduler": scheduler_kind(scheduler)},
        )
    except Exception:
        logger.warning("Failed to record KV update_from_output metrics", exc_info=True)

    return result


def count_waiting_for_remote_kvs(scheduler: Any) -> int:
    from vllm_ascend.observability.handlers.common import count_waiting_for_remote_kvs as _count

    return _count(scheduler)


def kv_xfer_finished_handler(original_func, scheduler, kv_connector_output, *args, **kwargs):
    """Count finished PD KV receive operations per scheduler step."""
    finished = getattr(kv_connector_output, "finished_recving", None) or ()
    finished_count = len(finished)

    result = original_func(scheduler, kv_connector_output, *args, **kwargs)

    try:
        if finished_count:
            _metrics().record_metric(
                _XFER_FINISHED,
                value=float(finished_count),
                labels={"connector": _connector_label(scheduler)},
            )
    except Exception:
        logger.warning("Failed to record KV xfer finished metrics", exc_info=True)

    return result


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
