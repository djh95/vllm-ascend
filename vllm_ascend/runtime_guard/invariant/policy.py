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

"""Which ranks run invariant checks (no cross-rank Incident shipping).

Without gathering findings to the leader, report is always written by the same
rank that ran the check. Therefore:

- ``leader``: one check + one report (TP-redundant data).
- ``all``: every detect-eligible rank checks and may write a rank-tagged report
  (rank-local meta under CP/DP).

``INVARIANT_SECTIONS`` lives in ``runtime_config._defaults`` (single source).
"""

from __future__ import annotations

from typing import Any

from vllm_ascend.runtime_guard.rank_gate import (
    is_action_leader_rank,
    should_run_anomaly_check_on_rank,
)


def rank_local_world(runner: Any) -> bool:
    """True when scheduler / KV addressing can diverge across ranks (CP or DP)."""
    pc = getattr(getattr(runner, "vllm_config", None), "parallel_config", None)
    if pc is None:
        return False
    cp = int(getattr(pc, "prefill_context_parallel_size", 1) or 1)
    dp = int(getattr(pc, "dp_size", 1) or 1)
    return cp > 1 or dp > 1


def resolve_check_scope(
    name: str,
    raw: str | None,
    *,
    rank_local_world: bool,
) -> str:
    """Map config ``check_scope`` to ``leader`` or ``all``."""
    scope = str(raw or "auto").strip().lower()
    if scope in ("leader", "all"):
        return scope
    # auto: rank-local block/slot meta needs every rank; otherwise leader once.
    if name in ("slot_consistency", "kv_slot_order", "kv_state") and rank_local_world:
        return "all"
    return "leader"


def should_run_invariant_check(runner: Any, *, check_scope: str) -> bool:
    """Whether this process should execute an invariant check this step."""
    if not should_run_anomaly_check_on_rank(runner):
        return False
    if check_scope == "all":
        return True
    return is_action_leader_rank(runner)
