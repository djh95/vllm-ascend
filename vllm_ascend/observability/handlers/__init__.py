# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers.

Handlers are split by domain under this package. Re-export symbols here so YAML
can keep using ``vllm_ascend.observability.handlers:function_name``.
"""

from vllm_ascend.observability.handlers.eplb import eplb_do_update_hotness_handler
from vllm_ascend.observability.handlers.flashcomm import (
    flashcomm_failure_note_handler,
    flashcomm_forward_flush_handler,
    flashcomm_gate_handler,
)

__all__ = [
    "eplb_do_update_hotness_handler",
    "flashcomm_failure_note_handler",
    "flashcomm_forward_flush_handler",
    "flashcomm_gate_handler",
]
