# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallelism (SP / CP / DP pad) metric handlers."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    histogram_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_SP_PADDING_RATIO = "parallel:sp_padding_ratio"
_DP_PAD_SIZE = "parallel:dp_pad_tokens"
_MC2_COMM = "parallel:moe_comm_selection_total"

_PARALLEL_METRICS = {
    _SP_PADDING_RATIO: (histogram_type(), []),
    _DP_PAD_SIZE: (histogram_type(), []),
    _MC2_COMM: (counter_type(), ["comm"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_PARALLEL_METRICS, _recorder_cache, "parallel")


def sp_pad_handler(original_func, runner, num_scheduled_tokens, *args, **kwargs):
    """Record sequence-parallel padding overhead."""
    before = int(num_scheduled_tokens)
    padded = original_func(runner, num_scheduled_tokens, *args, **kwargs)

    try:
        if padded > before:
            _metrics().record_metric(
                _SP_PADDING_RATIO,
                value=float(padded - before) / float(max(before, 1)),
                labels={},
            )
    except Exception:
        logger.warning("Failed to record SP padding metrics", exc_info=True)

    return padded


def moe_comm_selection_handler(original_func, num_tokens, vllm_config, *args, **kwargs):
    """Record MoE communication method selection (MC2 / multistream paths)."""
    comm_type = original_func(num_tokens, vllm_config, *args, **kwargs)

    try:
        label = "none"
        if comm_type is not None:
            label = getattr(comm_type, "name", str(comm_type))
        _metrics().record_metric(
            _MC2_COMM,
            value=1.0,
            labels={"comm": label},
        )
    except Exception:
        logger.warning("Failed to record MoE comm selection metrics", exc_info=True)

    return comm_type
