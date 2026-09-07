#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Rank gating for detection and incident actions."""

from __future__ import annotations

from typing import Any

from vllm.distributed.parallel_state import get_pp_group, get_tp_group


def anomaly_check_rank_skip_reason(runner: Any) -> str | None:
    """None if this rank may run detectors; otherwise a short skip reason.

    Every last-PP TP rank may detect: logits are all-gathered (full-vocab on
    each rank), sampling is redundant per rank, and async outputs materialize
    on each WorkerProc. Which detectors actually run on this rank is decided
    per detector by ``DetectorManager``'s placement plan (detector/
    placement.py); this gate no longer pins detection to TP0.
    """
    if runner is None:
        return "no runner"
    try:
        if not get_pp_group().is_last_rank:
            return "not last PP rank"
    except Exception:
        return "PP group unavailable"
    return None


def should_run_anomaly_check_on_rank(runner: Any) -> bool:
    return anomaly_check_rank_skip_reason(runner) is None


def runner_tp_rank(runner: Any) -> int:
    """TP rank within this worker's TP group.

    Process-group truth first: v1 model runners do NOT carry a ``tp_rank``
    attribute (vllm GPUModelRunner has none), so attribute fallbacks silently
    resolve 0 on every worker. ``get_tp_group().rank_in_group`` is set up by
    the time any hook fires.
    """
    try:
        return int(get_tp_group().rank_in_group)
    except Exception:
        return int(getattr(runner, "tp_rank", 0) or 0)


def runner_dp_rank(runner: Any) -> int:
    try:
        from vllm.distributed.parallel_state import get_dp_group

        return int(get_dp_group().rank_in_group)
    except Exception:
        return int(getattr(runner, "dp_rank", 0) or 0)


def dump_rank_tag(runner: Any) -> str:
    dp = runner_dp_rank(runner)
    tp = runner_tp_rank(runner)
    try:
        pp = get_pp_group().rank_in_group
    except Exception:
        pp = "?"
    return f"dp{dp}_tp{tp}_pp{pp}"


def is_action_leader_rank(runner: Any) -> bool:
    """Rank that writes reports / performs KV dump for this worker."""
    try:
        if not get_pp_group().is_last_rank:
            return False
    except Exception:
        return False
    try:
        if get_tp_group().world_size > 1:
            return runner_tp_rank(runner) == 0
    except Exception:
        pass
    return True
