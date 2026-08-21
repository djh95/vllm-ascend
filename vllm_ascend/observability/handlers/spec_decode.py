# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative decoding / rejection sampling metrics."""

from __future__ import annotations

import logging
from typing import Any

from vllm_ascend.observability.handlers.common import (
    counter_type,
    histogram_type,
    register_metrics,
)

logger = logging.getLogger(__name__)

_DRAFT_TOKENS = "spec_decode:draft_tokens_total"
_ACCEPTED_TOKENS = "spec_decode:accepted_tokens_total"
_ACCEPTANCE_RATIO = "spec_decode:acceptance_ratio"
_SHAPE_MISMATCH = "spec_decode:shape_mismatch_total"

_SPEC_METRICS = {
    _DRAFT_TOKENS: (counter_type(), []),
    _ACCEPTED_TOKENS: (counter_type(), []),
    _ACCEPTANCE_RATIO: (histogram_type(), []),
    _SHAPE_MISMATCH: (counter_type(), ["kind"]),
}

_recorder_cache: dict[str, Any] = {}


def _metrics() -> Any:
    return register_metrics(_SPEC_METRICS, _recorder_cache, "spec_decode")


def _draft_token_count(metadata: Any) -> int:
    draft_tokens = getattr(metadata, "num_draft_tokens", None)
    if draft_tokens is None:
        return 0
    if hasattr(draft_tokens, "sum"):
        return int(draft_tokens.sum().item())
    return int(sum(draft_tokens))


def spec_rejection_forward_handler(original_func, sampler_self, metadata, logits, *args, **kwargs):
    """Record acceptance stats from AscendRejectionSampler.forward (MTP/Eagle/DSpark)."""
    try:
        if logits is not None and hasattr(logits, "shape") and metadata is not None:
            expected_rows = int(metadata.cu_num_draft_tokens.shape[0])
            if int(logits.shape[0]) != expected_rows:
                _metrics().record_metric(
                    _SHAPE_MISMATCH,
                    value=1.0,
                    labels={"kind": "logits_rows"},
                )
    except Exception:
        logger.debug("Spec decode shape pre-check failed", exc_info=True)

    output = original_func(sampler_self, metadata, logits, *args, **kwargs)

    try:
        from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

        sampled = getattr(output, "sampled_token_ids", None)
        if sampled is None or metadata is None:
            return output

        metrics = _metrics()
        num_drafts = _draft_token_count(metadata)
        if num_drafts > 0:
            metrics.record_metric(_DRAFT_TOKENS, value=float(num_drafts), labels={})

        valid_mask = sampled.ne(PLACEHOLDER_TOKEN_ID)
        num_accepted = int(valid_mask.sum().item())
        num_slots = int(sampled.numel())
        if num_accepted > 0:
            metrics.record_metric(_ACCEPTED_TOKENS, value=float(num_accepted), labels={})
        if num_slots > 0:
            metrics.record_metric(
                _ACCEPTANCE_RATIO,
                value=float(num_accepted) / float(num_slots),
                labels={},
            )
    except Exception:
        logger.warning("Failed to record spec decode rejection metrics", exc_info=True)

    return output
