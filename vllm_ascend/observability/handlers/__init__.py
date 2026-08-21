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
from vllm_ascend.observability.handlers.flashcomm import (
    flashcomm_failure_note_handler,
    flashcomm_forward_flush_handler,
    flashcomm_gate_handler,
    flashcomm_path_handler,
)
from vllm_ascend.observability.handlers.graph import acl_graph_call_handler
from vllm_ascend.observability.handlers.kv import (
    kv_pool_load_errors_handler,
    kv_pool_put_failure_handler,
    kv_pool_start_load_handler,
    kv_update_from_output_handler,
    kv_xfer_finished_handler,
)
from vllm_ascend.observability.handlers.lifecycle import (
    worker_sleep_handler,
    worker_wake_handler,
)
from vllm_ascend.observability.handlers.parallel import (
    moe_comm_selection_handler,
    sp_pad_handler,
)
from vllm_ascend.observability.handlers.scheduler import (
    scheduler_preempt_handler,
    scheduler_schedule_semantic_handler,
)
from vllm_ascend.observability.handlers.spec_decode import spec_rejection_forward_handler

__all__ = [
    "acl_graph_call_handler",
    "eplb_async_worker_status_handler",
    "eplb_do_update_hotness_handler",
    "eplb_transfer_stats_handler",
    "flashcomm_failure_note_handler",
    "flashcomm_forward_flush_handler",
    "flashcomm_gate_handler",
    "flashcomm_path_handler",
    "kv_pool_load_errors_handler",
    "kv_pool_put_failure_handler",
    "kv_pool_start_load_handler",
    "kv_update_from_output_handler",
    "kv_xfer_finished_handler",
    "moe_comm_selection_handler",
    "sample_tokens_duration_handler",
    "scheduler_preempt_handler",
    "scheduler_schedule_semantic_handler",
    "sp_pad_handler",
    "spec_rejection_forward_handler",
    "worker_sleep_handler",
    "worker_wake_handler",
]
