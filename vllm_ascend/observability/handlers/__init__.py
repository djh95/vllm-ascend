# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers.

Handlers are split by domain under this package. Re-export symbols here so YAML
can keep using ``vllm_ascend.observability.handlers:function_name``.
"""

from vllm_ascend.observability.handlers.async_scheduling import (
    async_after_schedule_handler,
    async_note_stale_discard_handler,
    async_note_underflow_handler,
    async_output_get_output_handler,
    async_output_queue_handler,
    async_seq_lens_update_handler,
)
from vllm_ascend.observability.handlers.eplb import eplb_do_update_hotness_handler

__all__ = [
    "async_after_schedule_handler",
    "async_note_stale_discard_handler",
    "async_note_underflow_handler",
    "async_output_get_output_handler",
    "async_output_queue_handler",
    "async_seq_lens_update_handler",
    "eplb_do_update_hotness_handler",
]
