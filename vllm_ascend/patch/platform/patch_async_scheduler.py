# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Harden AsyncScheduler placeholder accounting for production observability."""

from __future__ import annotations

from functools import wraps

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


def _patch_async_scheduler_update_request_with_output() -> None:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if getattr(
        AsyncScheduler._update_request_with_output,
        "_vllm_ascend_async_obs_patched",
        False,
    ):
        return

    original_update_request_with_output = AsyncScheduler._update_request_with_output

    @wraps(original_update_request_with_output)
    def _patched_update_request_with_output(
        self,
        request,
        new_token_ids,
    ):
        if request.async_tokens_to_discard > 0:
            logger.warning(
                "Async scheduling discarded stale output for request %s (async_tokens_to_discard=%d)",
                request.request_id,
                request.async_tokens_to_discard,
            )
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self,
            request,
            new_token_ids,
        )

        request.num_output_placeholders -= len(new_token_ids)
        if request.num_output_placeholders < 0:
            logger.warning(
                "Async scheduling placeholder underflow for request %s: consumed=%d, placeholders_after=%d",
                request.request_id,
                len(new_token_ids),
                request.num_output_placeholders,
            )
            request.num_output_placeholders = 0

        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders,
            )
        return new_token_ids, stopped

    _patched_update_request_with_output._vllm_ascend_async_obs_patched = True  # type: ignore[attr-defined]
    AsyncScheduler._update_request_with_output = _patched_update_request_with_output


def apply_async_scheduler_observability_patch() -> None:
    _patch_async_scheduler_update_request_with_output()


apply_async_scheduler_observability_patch()
