#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft-assert invariant checks (not anomaly detectors / not LPT-placed).

Call sites gate on ``runtime_config.invariant.<name>.enabled``, run a plain
check, and ``emit_finding`` → report / optional dump_kv.
"""

from __future__ import annotations

from vllm_ascend.runtime_config._defaults import INVARIANT_SECTIONS
from vllm_ascend.runtime_guard.invariant.batch_locs import (
    num_computed_before,
    query_start_loc_np,
)
from vllm_ascend.runtime_guard.invariant.logits_finite import check_logits_finite
from vllm_ascend.runtime_guard.invariant.policy import (
    rank_local_world,
    resolve_check_scope,
    should_run_invariant_check,
)
from vllm_ascend.runtime_guard.invariant.slot_consistency import SlotConsistencyState

__all__ = [
    "INVARIANT_SECTIONS",
    "SlotConsistencyState",
    "check_logits_finite",
    "num_computed_before",
    "query_start_loc_np",
    "rank_local_world",
    "resolve_check_scope",
    "should_run_invariant_check",
]
