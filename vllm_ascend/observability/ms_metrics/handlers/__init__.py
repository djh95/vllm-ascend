# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers, one module per feature.

Shared building blocks live in :mod:`.common`; feature modules only depend on
them. Re-export new feature handlers here so the YAML reference
``vllm_ascend.observability.ms_metrics.handlers:function_name`` stays stable.
"""

from vllm_ascend.observability.ms_metrics.handlers.acl_graph import graph_replay_handler
from vllm_ascend.observability.ms_metrics.handlers.eplb import eplb_do_update_hotness_handler
from vllm_ascend.observability.ms_metrics.handlers.schedule_execute import (
    lb_pause_preempt_handler,
    recompute_preempt_handler,
    sample_tokens_handler,
    schedule_preempt_handler,
)

__all__ = [
    "eplb_do_update_hotness_handler",
    "graph_replay_handler",
    "lb_pause_preempt_handler",
    "recompute_preempt_handler",
    "sample_tokens_handler",
    "schedule_preempt_handler",
]
