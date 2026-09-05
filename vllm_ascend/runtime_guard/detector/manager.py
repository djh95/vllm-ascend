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

"""Detector manager facade: stage hooks over a private detector registry.

Concrete detectors are private references held by ``DetectorManager``; callers
(``RuntimeGuardProcessor`` / model runners) only use the stage hooks, never the detector
instances. The internal ``DetectorRegistry`` keeps iteration / clear-finished
without exposing its public surface to the outside.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.detector.base import AnomalyDetector
from vllm_ascend.runtime_guard.detector.block_kv import BlockKvDetector
from vllm_ascend.runtime_guard.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.runtime_guard.detector.output_substring import OutputSubstringDetector
from vllm_ascend.runtime_guard.detector.placement import DetectorSpec, ExecScope, PlacementPlan, plan_placement
from vllm_ascend.runtime_guard.detector.position_alignment import PositionAlignmentDetector
from vllm_ascend.runtime_guard.detector.registry import DetectorRegistry
from vllm_ascend.runtime_guard.detector.slot_consistency import SlotConsistencyDetector
from vllm_ascend.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.runtime_guard.detector.token_logprob import TokenLogprobDetector
from vllm_ascend.runtime_guard.detector.token_repeat import TokenRepeatDetector
from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.runtime_guard.rank_gate import runner_tp_rank
from vllm_ascend.runtime_guard.request_state import RequestGuardStore
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


class DetectorManager:
    """Owns detectors and exposes stage hooks only.

    Callers (``RuntimeGuardProcessor`` / runners) use ``check_after_spec`` /
    ``check_before_sample`` / ``check_after_sample`` / ``check_kv_block_writes`` /
    ``clear_finished`` (reap path) only.
    Concrete detectors stay private (``_spec_det`` & co.); ``get`` exists solely
    for alert routing in ``RuntimeGuardProcessor._handle_alert``.
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        runner: Any,
        is_related_request: Callable[[str, int | None], bool] | None = None,
        tokenizer_provider: Callable[[], Any | None] | None = None,
        detection_gate: Callable[[], bool] | None = None,
        detection_skip_reason: Callable[[], str | None] | None = None,
    ) -> None:
        self._runner = runner
        self._runtime_config = runtime_config
        # Anomaly detection gate (rank / dump / detector-on), owned by the
        # caller (``RuntimeGuardProcessor`` → ``Dumper``). None = always run.
        self._detection_gate = detection_gate
        self._detection_skip_reason = detection_skip_reason
        # Private concrete references (constructed once; no registry get+assert).
        self._spec_det = SpecAcceptanceDetector(
            runtime_config=runtime_config,
            runner=runner,
            is_related_request=is_related_request,
        )
        self._token_det = TokenLogprobDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._output_substring_det = OutputSubstringDetector(
            runtime_config=runtime_config,
            runner=runner,
            tokenizer_provider=tokenizer_provider,
        )
        self._token_repeat_det = TokenRepeatDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._block_kv_det = BlockKvDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._slot_consistency_det = SlotConsistencyDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._position_det = PositionAlignmentDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        self._logits_finite_det = LogitsFiniteDetector(
            runtime_config=runtime_config,
            runner=runner,
        )
        # Internal ordered registry: iterate for clear_finished; not public.
        self._registry = DetectorRegistry()
        for det in (
            self._spec_det,
            self._token_det,
            self._output_substring_det,
            self._token_repeat_det,
            self._block_kv_det,
            self._slot_consistency_det,
            self._position_det,
            self._logits_finite_det,
        ):
            self._registry.register(det)
        # stop_after_alert flags live on RequestGuardStore (RequestGuardState).

        # Per-rank placement: which detectors this process actually runs.
        self._tp_rank = runner_tp_rank(runner)
        self._plan = PlacementPlan()
        self._replan(initial=True)

    def rebind_runner(self, runner: Any) -> None:
        """Point this manager and all detectors at a new runner."""
        self._runner = runner
        for det in self._registry:
            det._runner = runner

    def get(self, incident_type: str) -> AnomalyDetector | None:
        """Resolve a detector for alert routing (``RuntimeGuardProcessor._handle_alert`` only)."""
        return self._registry.get(incident_type)

    def clear_finished(self, req_id: str) -> None:
        """Drop per-request detector state when a request finishes.

        Shared fields (IO / filter / waves) are cleared by
        :meth:`RequestGuardStore.clear`. This also clears ``stopped_after_alert``
        so direct callers / tests can re-detect without popping the whole state.
        Prefer Store.clear from ``RuntimeGuardProcessor._reap_finished_requests``.
        """
        state = RequestGuardStore.get().get_state(req_id)
        if state is not None:
            state.stopped_after_alert = False
        for det in self._registry:
            det.clear_finished(req_id)

    def token_logprob_topk_if_enabled(self) -> int | None:
        """Return token-logprob top-k when that detector is enabled; else None.

        With hot-reload off, skip per-sample ``refresh_from_config`` when the
        detector is already known disabled (default service path).
        """
        if self._runtime_config is not None and self._runtime_config.hot_reload_enabled:
            self._token_det.refresh_from_config()
        elif not self._token_det.enabled:
            return None
        if not self._token_det.enabled:
            return None
        topk = int(self._token_det.topk)
        return topk if topk > 0 else None

    # ---- placement ---------------------------------------------------------

    @staticmethod
    def _topology(runner: Any) -> tuple[int, bool]:
        """(tp_size, rank_local_world) for the planner."""
        vllm_config = getattr(runner, "vllm_config", None)
        pc = getattr(vllm_config, "parallel_config", None)
        if pc is not None:
            tp = int(getattr(pc, "tensor_parallel_size", 1) or 1)
            cp = int(getattr(pc, "prefill_context_parallel_size", 1) or 1)
            dp = int(getattr(pc, "dp_size", 1) or 1)
            return tp, (cp > 1 or dp > 1)
        try:
            from vllm.distributed.parallel_state import get_tp_group

            tp = int(get_tp_group().world_size)
        except Exception:
            tp = int(getattr(runner, "tp_size", 1) or 1)
        return tp, False

    def _detector_specs(self) -> list[DetectorSpec]:
        specs: list[DetectorSpec] = []
        for det in self._registry:
            raw = "auto"
            if self._runtime_config is not None:
                raw = self._runtime_config.detector_get(det.incident_type, "exec_scope", "auto")
            try:
                scope = ExecScope(str(raw)) if str(raw) != "auto" else ExecScope.ANY
            except ValueError:
                scope = ExecScope.ANY
            enabled = bool(getattr(det, "enabled", False))
            if self._runtime_config is not None:
                enabled = bool(self._runtime_config.detector_get(det.incident_type, "enabled", False))
            specs.append(
                DetectorSpec(
                    incident_type=det.incident_type,
                    exec_scope=scope,
                    rank_local_data=bool(getattr(det, "RANK_LOCAL_DATA", False)),
                    cost=float(getattr(det, "EST_COST_PER_STEP", 1.0)),
                    enabled=enabled,
                )
            )
        return specs

    def _replan(self, *, initial: bool = False) -> None:
        """(Re)compute which detectors run on this rank; log moves."""
        if self._runtime_config is None:
            self._plan = plan_placement([], tp_size=1, rank_local_world=False)
            return
        tp_size, rank_local_world = self._topology(self._runner)
        prev = None if initial else getattr(self, "_plan", None)
        new = plan_placement(
            self._detector_specs(),
            tp_size=tp_size,
            rank_local_world=rank_local_world,
            mode=self._runtime_config.detector_placement_mode(),
            manual=self._runtime_config.detector_placement_manual(),
            pin=self._runtime_config.detector_placement_pin(),
            previous=prev,
        )
        if not initial and prev is not None and prev != new:
            for spec in self._detector_specs():
                if not spec.enabled:
                    continue
                old_rank, new_rank = prev.rank_of(spec.incident_type), new.rank_of(spec.incident_type)
                if old_rank != new_rank:
                    logger.warning(
                        "[runtime_guard placement] %s moved rank%s->rank%s "
                        "(per-request history restarts on the new rank)",
                        spec.incident_type,
                        old_rank,
                        new_rank,
                    )
        self._plan = new
        logger.info(
            "[runtime_guard placement] tp_rank=%s plan: %s",
            self._tp_rank,
            new.summary(),
        )

    def _here(self, det: AnomalyDetector) -> bool:
        """Whether this rank runs ``det`` per the placement plan."""
        return self._plan.runs_here(det.incident_type, self._tp_rank)

    def _any_here(self) -> bool:
        return any(self._plan.runs_here(det.incident_type, self._tp_rank) for det in self._registry)

    def apply_runtime_config(self) -> None:
        """All-rank hook after DFX JSON sync — refresh deps that may force flags off.

        ``token_logprob`` needs msprobe; if missing, force ``enabled=false`` and
        persist on the JSON writer. Must run on every rank (including early PP
        writers that never sample), not only on the detect / sample path.
        Also refresh the newer native detectors so hot-reload flips take effect.
        """
        self._token_det.refresh_from_config()
        self._block_kv_det.refresh_from_config()
        self._slot_consistency_det.refresh_from_config()
        self._position_det.refresh_from_config()
        self._logits_finite_det.refresh_from_config()
        # Spec / substring / repeat also pull knobs from JSON on enable flips.
        self._spec_det.refresh_from_config()
        self._output_substring_det.refresh_from_config()
        self._token_repeat_det.refresh_from_config()
        # Enable flips may rebalance rank assignment (auto mode).
        self._replan()

    # ---- detection gating -------------------------------------------------

    def _gated(self, stage: str, *, ignore_dump_busy: bool = False) -> bool:
        """True when anomaly detection is gated off this step; logs skip reason once.

        ``stage`` is a short tag (``after_spec`` / ``after_sample`` / ``kv_block``)
        for the once-per-process skip log. Gate checks live here so callers never
        re-implement them per hook.

        ``ignore_dump_busy``: still run when pending/active dump (same-step
        follow-on detectors such as block_kv after logits/position already armed).
        """
        if self._detection_gate is None:
            return False

        def _call_gate() -> bool:
            try:
                return bool(self._detection_gate(ignore_dump_busy=ignore_dump_busy))  # type: ignore[misc]
            except TypeError:
                if not ignore_dump_busy:
                    return bool(self._detection_gate())
                # Legacy gate without kwarg: treat dump-busy as not gated.
                if self._detection_skip_reason is not None:
                    try:
                        reason = self._detection_skip_reason()
                    except TypeError:
                        reason = None
                    if reason in (
                        "pending_dump already armed",
                        "msprobe dump already active",
                    ):
                        return True
                return bool(self._detection_gate())

        if _call_gate():
            return False
        reason = None
        if self._detection_skip_reason is not None:
            try:
                reason = self._detection_skip_reason(ignore_dump_busy=ignore_dump_busy)  # type: ignore[misc]
            except TypeError:
                reason = self._detection_skip_reason()
        if reason and int(getattr(self._runner, "tp_rank", 0)) == 0:
            logger.info_once(
                "[Anomaly detect short] skip gate (%s): %s (any_detector=%s dump.enabled=%s)",
                stage,
                reason,
                self._runtime_config.any_detector_enabled(),
                self._runtime_config.dump_enabled(),
            )
        return True

    # ---- stage hooks ------------------------------------------------------

    def _stop_after_alert(self) -> bool:
        return bool(self._runtime_config.stop_after_alert())

    def any_enabled_for_spec(self) -> bool:
        if self._runtime_config is not None and self._runtime_config.hot_reload_enabled:
            self._spec_det.refresh_from_config()
        return bool(self._spec_det.enabled)

    def _mark_alerted(self, alerts: list[Incident]) -> None:
        """Stop detecting requests that just produced an anomaly."""
        store = RequestGuardStore.get()
        for alert in alerts:
            if alert.req_id:
                store.get_or_create(alert.req_id).stopped_after_alert = True

    def check_after_spec(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
    ) -> list[Incident]:
        """Run spec-acceptance detect only (no cumulative IO append).

        Accepted tokens are recorded once in :meth:`check_after_sample` from
        the engine's validated sampled ids. Appending here as well doubled
        MTP/Eagle output in reports (same-wave dedupe fails under async
        scheduling when ``clear_wave_cache`` runs before ``get_output``).
        """
        if self._gated("after_spec"):
            return []
        if not self._here(self._spec_det):
            return []
        skip = RequestGuardStore.get().stopped_req_ids() if self._stop_after_alert() else None
        alerts = self._spec_det.check_all(sampled_tokens, accepted_token_nums, skip_req_ids=skip)
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts

    def check_after_sample(
        self,
        sampled_token_ids: Any,
        logprobs_lists: Any,
        req_ids: list[str] | None = None,
    ) -> list[Incident]:
        """Append sample tokens to IO buffer; run post-sample detectors.

        Order: ``token_logprob`` → ``output_substring`` → ``token_repeat``.

        ``sampled_token_ids`` is the sole path that appends to cumulative IO
        (including MTP/Eagle accepted tokens). Append runs only when
        ``RuntimeConfig.needs_cumulative_io`` (print_output / IO detectors /
        sensitive reports). When detect is gated off but IO is still needed,
        TP0 appends so finish logs stay complete. Detection then skips
        ``stop_after_alert`` reqs by id — never by row-subsetting — so
        ``req_idx`` stays aligned with ``input_batch`` (filters / dump related).

        ``OutputSubstringDetector`` and ``TokenRepeatDetector`` are called with
        ``sampled_token_ids=None`` so they read the shared cumulative IO buffer
        instead of re-appending (avoids double count).

        With ``detector.stop_after_alert`` (default true) a request keeps being
        checked on every step until it produces an anomaly; afterwards it is
        skipped entirely so the same anomaly does not write endless reports.
        """
        resolved_ids = req_ids
        if resolved_ids is None:
            input_batch = getattr(self._runner, "input_batch", None)
            resolved_ids = list(getattr(input_batch, "req_ids", None) or [])

        need_io = True
        if self._runtime_config is not None:
            need_io = bool(self._runtime_config.needs_cumulative_io())

        # Append only when a consumer needs cumulative IO (print_output /
        # substring/repeat / sensitive reports). When detect is gated off,
        # TP0 still appends if need_io so finish logs stay complete.
        if self._gated("after_sample"):
            if need_io and int(getattr(self._runner, "tp_rank", 0)) == 0:
                RequestIoSnapshotManager.get().append_batch(resolved_ids, sampled_token_ids)
            return []

        # IO authority stays on TP0 (reports / finish logs); other ranks
        # append only when a detector assigned here reads cumulative IO.
        io_owner = int(getattr(self._runner, "tp_rank", 0)) == 0
        if need_io and (io_owner or self._any_here()):
            RequestIoSnapshotManager.get().append_batch(resolved_ids, sampled_token_ids)

        skip = RequestGuardStore.get().stopped_req_ids() if self._stop_after_alert() else None
        if skip is not None and resolved_ids and all(rid in skip for rid in resolved_ids if rid):
            # Entire batch already alerted: IO updated; no further detect / reports.
            return []

        alerts: list[Incident] = []
        if self._here(self._token_det):
            alerts.extend(
                self._token_det.check_all(
                    sampled_token_ids=sampled_token_ids,
                    logprobs_lists=logprobs_lists,
                    req_ids=resolved_ids,
                    skip_req_ids=skip,
                )
            )
        # Substring + token_repeat share cumulative IO (already appended); pass
        # None to avoid a second append_batch inside each detector.check_all.
        if self._here(self._output_substring_det):
            alerts.extend(
                self._output_substring_det.check_all(
                    sampled_token_ids=None,
                    req_ids=resolved_ids,
                    skip_req_ids=skip,
                )
            )
        if self._here(self._token_repeat_det):
            alerts.extend(
                self._token_repeat_det.check_all(
                    sampled_token_ids=None,
                    req_ids=resolved_ids,
                    skip_req_ids=skip,
                )
            )
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts

    def check_before_sample(
        self,
        *,
        scheduler_output: Any,
        logits: Any,
        positions: Any = None,
        total_scheduled_tokens: int = 0,
        logits_indices: Any = None,
        input_batch: Any = None,
    ) -> list[Incident]:
        """Run pre-sample detectors (logits finite, then position alignment).

        With ``stop_after_alert``, a req that alerts in logits is marked before
        position runs in the same call, so the same step does not double-report.
        """
        logits_here = self._here(self._logits_finite_det)
        position_here = self._here(self._position_det)
        if not logits_here and not position_here:
            return []
        if self._gated("before_sample"):
            return []
        stop = self._stop_after_alert()
        skip = RequestGuardStore.get().stopped_req_ids() if stop else None
        alerts: list[Incident] = []
        logits_alerts: list[Incident] = []
        if logits_here:
            for alert in self._logits_finite_det.check_all(
                logits=logits,
                logits_indices=logits_indices,
                input_batch=input_batch,
            ):
                if skip is not None and alert.req_id in skip:
                    continue
                logits_alerts.append(alert)
                alerts.append(alert)
        if stop and logits_alerts:
            self._mark_alerted(logits_alerts)
            skip = RequestGuardStore.get().stopped_req_ids()
        if position_here:
            for alert in self._position_det.check_all(
                scheduler_output=scheduler_output,
                positions=positions,
                total_scheduled=total_scheduled_tokens,
                input_batch=input_batch,
            ):
                if skip is not None and alert.req_id in skip:
                    continue
                alerts.append(alert)
        if stop:
            # Position-only hits (logits already marked above).
            self._mark_alerted(alerts)
        return alerts

    def check_kv_block_writes(
        self,
        req_id: str,
        block_ids: list[int],
        wave: int,
    ) -> list[Incident]:
        """Run block KV integrity checks before ``record_writes``.

        Ignores dump-busy gate so same-step pending dump (armed by
        logits/position) does not skip KV checks.
        """
        if self._gated("kv_block", ignore_dump_busy=True):
            return []
        if not self._here(self._block_kv_det):
            return []
        skip = RequestGuardStore.get().stopped_req_ids() if self._stop_after_alert() else None
        if skip is not None and req_id in skip:
            return []
        alerts = self._block_kv_det.check_writes(req_id, block_ids, wave)
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts

    def slot_consistency_enabled(self) -> bool:
        """Cheap gate for the note loop (avoids snapshot cost when disabled).

        Also placement-gated: a rank the detector is not assigned to skips
        slot snapshot/record work entirely.
        """
        if not self._here(self._slot_consistency_det):
            return False
        if self._runtime_config is not None and self._runtime_config.hot_reload_enabled:
            self._slot_consistency_det.refresh_from_config()
        return bool(self._slot_consistency_det.enabled)

    def check_slot_consistency(
        self,
        *,
        req_id: str,
        req_idx: int | None,
        seq: list[int],
        block_ids: list[int],
        block_size: int,
        computed_before: int,
        scheduled: int,
    ) -> list[Incident]:
        """Run slot-token consistency checks after this step's slot record."""
        if self._gated("kv_block", ignore_dump_busy=True):
            return []
        if not self._here(self._slot_consistency_det):
            return []
        skip = RequestGuardStore.get().stopped_req_ids() if self._stop_after_alert() else None
        if skip is not None and req_id in skip:
            return []
        alerts = self._slot_consistency_det.check_slots(
            req_id=req_id,
            req_idx=req_idx,
            seq=seq,
            block_ids=block_ids,
            block_size=block_size,
            computed_before=computed_before,
            scheduled=scheduled,
        )
        if skip is not None:
            self._mark_alerted(alerts)
        return alerts
