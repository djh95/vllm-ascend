# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV transfer (PD) metric handlers."""

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
_XFER_FINISHED = "kv:xfer_finished_total"
_XFER_FAILED_REQUESTS = "kv:xfer_failed_requests"
_REMOTE_KV_WAITING = "kv:remote_kv_waiting_requests"

_KV_METRICS = {
    _INVALID_BLOCKS: (counter_type(), ["connector"]),
    _XFER_FINISHED: (counter_type(), ["connector"]),
    _XFER_FAILED_REQUESTS: (gauge_type(), ["connector"]),
    _REMOTE_KV_WAITING: (gauge_type(), ["scheduler"]),
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
    """Record KV invalid blocks and remote-KV backlog after scheduler output processing."""
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
