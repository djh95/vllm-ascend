# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers.

Handlers are split by domain under this package. Re-export symbols here so YAML
can keep using ``vllm_ascend.observability.handlers:function_name``.
"""

from vllm_ascend.observability.handlers.eplb import (
    eplb_async_worker_status_handler,
    eplb_do_update_hotness_handler,
    eplb_transfer_stats_handler,
)
from vllm_ascend.observability.handlers.executor import sample_tokens_duration_handler

__all__ = [
    "eplb_async_worker_status_handler",
    "eplb_do_update_hotness_handler",
    "eplb_transfer_stats_handler",
    "sample_tokens_duration_handler",
]
