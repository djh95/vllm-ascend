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

"""Detector execution placement: which rank runs which detector.

Data-availability facts this builds on (vllm 0.27.x, verified on Ascend):

- ``LogitsProcessor`` gathers logits with ``tensor_model_parallel_all_gather``
  (``use_all_gather()`` defaults True and is not overridden on Ascend), so
  **every TP rank holds full-vocab logits** at ``check_before_sample``.
- v1 samples redundantly per rank (identical results; no post-sample
  broadcast), so sampled ids / logprobs exist on every rank at
  ``check_after_sample``.
- Scheduler metadata (block tables / slot mapping) is replicated across TP
  ranks; it only becomes rank-local under CP / DP attention.
- In async-scheduling mode every WorkerProc materializes its own async output
  (``enqueue_output`` calls ``get_output()`` on each rank), so post-sample
  hooks fire on all ranks.

Therefore no detector is inherently bound to TP0; only single-writer actions
(report persistence, manual-dump quota consumption) are leader-only. This
module spreads detectors across TP ranks so the hot path does not pile every
check onto rank 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

_LEADER_RANK = 0


class ExecScope(str, Enum):
    """Where a detector is allowed to run.

    - ``LEADER``: last-PP TP0 only (single-writer semantics; no detector
      currently needs this — reserved).
    - ``ANY``: data is rank-redundant; exactly one rank runs it (schedulable).
    - ``ALL``: data is rank-local (CP/DP); every rank runs its own view.
    - ``EXTERNAL``: off the worker process (api-server / offline). Reserved;
      not scheduled by the planner yet.
    """

    LEADER = "leader"
    ANY = "any"
    ALL = "all"
    EXTERNAL = "external"


# Config-facing value "auto" (default) resolves from the detector's static
# attributes: RANK_LOCAL_DATA + topology decide ANY vs ALL.
_VALID_SCOPE_VALUES = frozenset({"auto", "leader", "any", "all", "external"})


@dataclass(frozen=True)
class DetectorSpec:
    """Planner input for one detector (all static / config-derived)."""

    incident_type: str
    exec_scope: ExecScope
    rank_local_data: bool
    cost: float
    enabled: bool


@dataclass(frozen=True)
class PlacementPlan:
    """Planner output: detector → rank assignment for this process's view.

    ``assignment`` maps incident_type → tp_rank for LEADER/ANY detectors;
    ``all_ranks`` lists incident types that run on every TP rank (ALL).
    """

    assignment: dict[str, int] = field(default_factory=dict)
    all_ranks: frozenset[str] = frozenset()

    def rank_of(self, incident_type: str) -> int | None:
        """Rank assigned to a single-rank detector; None for ALL/unknown."""
        return self.assignment.get(incident_type)

    def runs_here(self, incident_type: str, tp_rank: int) -> bool:
        if incident_type in self.all_ranks:
            return True
        return self.assignment.get(incident_type) == int(tp_rank)

    def summary(self) -> str:
        parts = [
            f"{name}->rank{rank}"
            for name, rank in sorted(self.assignment.items(), key=lambda kv: (kv[1], kv[0]))
        ]
        if self.all_ranks:
            parts.append("ALL:" + ",".join(sorted(self.all_ranks)))
        return " ".join(parts) if parts else "(none)"


def resolve_scope(
    *,
    exec_scope: ExecScope,
    rank_local_data: bool,
    rank_local_world: bool,
) -> ExecScope:
    """Effective scope: explicit config wins; else topology decides."""
    if exec_scope in (ExecScope.LEADER, ExecScope.ALL, ExecScope.EXTERNAL):
        return exec_scope
    # ANY (or auto): rank-local detectors must run everywhere under CP/DP.
    if rank_local_data and rank_local_world:
        return ExecScope.ALL
    return ExecScope.ANY


def plan_placement(
    specs: list[DetectorSpec],
    *,
    tp_size: int,
    rank_local_world: bool,
    mode: str = "auto",
    manual: dict[str, int] | None = None,
    pin: bool = False,
    previous: PlacementPlan | None = None,
) -> PlacementPlan:
    """Assign detectors to TP ranks.

    - LEADER → rank 0; ALL → every rank (no assignment entry).
    - ANY (enabled only): LPT bin-packing by ``cost`` (deterministic: ties
      broken by name, then by previous rank to minimize churn).
    - ``mode="manual"``: entries in ``manual`` win (validated against
      ``tp_size``; out-of-range entries fall back to auto with the caller's
      warning hook responsible for logging).
    - ``pin=True``: keep ``previous`` assignments for still-enabled detectors;
      place only the rest.
    """
    tp_size = max(1, int(tp_size))
    assignment: dict[str, int] = {}
    all_ranks: set[str] = set()
    manual_map = dict(manual or {})

    pinned: dict[str, int] = {}
    if pin and previous is not None:
        for spec in specs:
            if not spec.enabled:
                continue
            prev = previous.rank_of(spec.incident_type)
            if prev is not None and 0 <= prev < tp_size:
                pinned[spec.incident_type] = prev

    loads = [0] * tp_size
    for name, rank in pinned.items():
        if 0 <= rank < tp_size:
            assignment[name] = rank
            loads[rank] += 1  # cost unknown here; count-based is enough for pin

    any_specs: list[DetectorSpec] = []
    for spec in specs:
        scope = resolve_scope(
            exec_scope=spec.exec_scope,
            rank_local_data=spec.rank_local_data,
            rank_local_world=rank_local_world,
        )
        if scope is ExecScope.EXTERNAL:
            continue
        if not spec.enabled:
            continue
        if scope is ExecScope.ALL:
            all_ranks.add(spec.incident_type)
            continue
        if spec.incident_type in assignment:
            continue  # pinned
        if scope is ExecScope.LEADER:
            assignment[spec.incident_type] = _LEADER_RANK
            loads[_LEADER_RANK] += max(spec.cost, 0.0)
            continue
        any_specs.append(spec)

    # Manual overrides win for not-yet-placed ANY detectors.
    for spec in any_specs:
        rank = manual_map.pop(spec.incident_type, None)
        if rank is not None and 0 <= int(rank) < tp_size:
            assignment[spec.incident_type] = int(rank)
            loads[int(rank)] += max(spec.cost, 0.0)
    any_specs = [s for s in any_specs if s.incident_type not in assignment]

    # LPT: most expensive first, least-loaded rank wins; tie → lower rank,
    # then keep previous rank if it ties, else name order decides sequence.
    for spec in sorted(any_specs, key=lambda s: (-s.cost, s.incident_type)):
        cost = max(float(spec.cost), 0.0)
        prev_rank = previous.rank_of(spec.incident_type) if previous is not None else None
        best = min(
            range(tp_size),
            key=lambda r: (loads[r], 0 if (prev_rank is not None and r == prev_rank) else 1, r),
        )
        assignment[spec.incident_type] = best
        loads[best] += cost

    return PlacementPlan(assignment=dict(assignment), all_ranks=frozenset(all_ranks))
