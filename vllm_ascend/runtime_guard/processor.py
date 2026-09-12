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

"""Process-wide runtime-guard orchestration (config sync, detectors, actions).

Owns construction of detectors / ``ReportWriter`` / ``ActionExecutor``. Model
runners ``bind`` the process singleton once; other call sites use
:meth:`RuntimeGuardProcessor.get`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from vllm.distributed.parallel_state import get_pp_group, get_tp_group

from vllm_ascend.runtime_config._dist import (
    SYNC_BROADCAST,
    _runtime_config_sync_group_or_none,
)

from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard import inject
from vllm_ascend.runtime_guard.detector.base import AnomalyDetector
from vllm_ascend.runtime_guard.detector.manager import DetectorManager
from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.runtime_guard.kv_block_meta import (
    block_ids_for_request,
    slot_mapping_for_request,
)
from vllm_ascend.runtime_guard.manual_trigger import (
    ManualTriggerManager,
    TriggerEvent,
    iter_local_request_rows,
)
from vllm_ascend.runtime_guard.rank_gate import (
    dump_rank_tag,
    is_action_leader_rank,
    runner_tp_rank,
    should_dump_kv_on_rank,
)
from vllm_ascend.runtime_guard.report import ReportWriter
from vllm_ascend.runtime_guard.request_state import RequestGuardStore
from vllm_ascend.runtime_guard.util import load_model_tokenizer
from vllm_ascend.runtime_guard.util import decode_token_ids
from vllm_ascend.runtime_guard.action.executor import ActionExecutor
from vllm_ascend.runtime_guard.quota import DumpQuota
from vllm_ascend.runtime_guard.wave_tracker import WaveTracker
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


@dataclass
class SamplePhaseResult:
    """Runner-side sample-phase outputs needed by post-sample runtime_guard hooks.

    Returned by the ``sample_fn`` callback passed to
    :meth:`RuntimeGuardProcessor.run_sample_phase`. Bundles the values the runner
    already computes (``ModelRunnerOutput`` + ``sampler_output`` + the
    bookkeeping sync outputs) so hooks 3-8 don't need to re-fetch them.
    """

    scheduler_output: Any
    input_batch: Any
    model_runner_output: Any
    sampler_output: Any
    valid_sampled_token_ids: Any
    req_ids_output_copy: Any
    invalid_req_indices: Any
    finished_req_ids: Any
    hidden_states: Any = None
    spec_decode_metadata: Any = None


class RuntimeGuardProcessor:
    """Process-wide singleton: config sync → detect → report / dump_kv.

    Create / attach a runner with :meth:`bind` (or ``RuntimeGuardProcessor(runner)``).
    Retrieve with :meth:`get`. One worker process should bind at most one model runner.
    """

    _instance: ClassVar[RuntimeGuardProcessor | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, runner: Any | None = None):
        # ``RuntimeGuardProcessor(runner)`` is an alias of :meth:`bind`.
        if runner is None:
            raise TypeError("RuntimeGuardProcessor() requires a model runner; use bind(runner)")
        return cls.bind(runner)

    def __init__(self, runner: Any | None = None) -> None:
        # Initialization is done in :meth:`bind` / ``_init_from_runner``.
        return

    @classmethod
    def get(cls) -> RuntimeGuardProcessor:
        """Return the process singleton. Raises if :meth:`bind` has not run."""
        inst = cls._instance
        if inst is None:
            raise RuntimeError(
                "RuntimeGuardProcessor is not bound; call RuntimeGuardProcessor.bind(runner) first"
            )
        return inst

    @classmethod
    def try_get(cls) -> RuntimeGuardProcessor | None:
        """Return the process singleton, or None if not bound yet."""
        return cls._instance

    @classmethod
    def bind(cls, runner: Any) -> RuntimeGuardProcessor:
        """Create or rebind the process singleton to ``runner``."""
        if runner is None:
            raise ValueError("RuntimeGuardProcessor.bind requires a model runner")
        with cls._lock:
            if cls._instance is None:
                inst = object.__new__(cls)
                inst._init_from_runner(runner)
                cls._instance = inst
            else:
                cls._instance._rebind_runner(runner)
            return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop singleton (unit tests only)."""
        with cls._lock:
            inst = cls._instance
            cls._instance = None
        if inst is not None:
            try:
                inst.shutdown()
            except Exception:
                pass

    def _init_from_runner(self, runner: Any) -> None:
        ascend = runner.ascend_config
        runtime_config: RuntimeConfig = ascend.runtime_config

        self.runner = runner
        self.runtime_config = runtime_config
        # Leader materializes JSON once. Non-leaders no-op inside ensure_persisted.
        runtime_config.ensure_persisted()
        # Runtime config is solely ``runtime_config`` (JSON).
        self.wave_tracker = WaveTracker()
        # B1: reap must wait for the last async output of a finished request;
        # WaveTracker.pending is the drain signal (replaces the dead store FIFO).
        RequestGuardStore.get().set_drain_probe(self.wave_tracker.pending)
        self.quota = DumpQuota(runtime_config)
        # Detection + report only on last-PP TP0; all ranks share the same report root.
        self.report_writer = ReportWriter(
            runtime_config.report_dir,
            save_sensitive_info=runtime_config.report_save_sensitive_info(),
            max_prompt_token_ids=runtime_config.report_max_prompt_token_ids(),
            max_output_token_ids=runtime_config.report_max_output_token_ids(),
            decode_token_ids=runtime_config.report_decode_token_ids(),
            max_per_req=runtime_config.report_max_per_req(),
            dump_root_provider=runtime_config.dump_root,
        )
        self.action_executor = ActionExecutor(
            runner,
            runtime_config=runtime_config,
            report_writer=self.report_writer,
            quota=self.quota,
        )
        self.action_executor.start()
        self._report_tokenizer: Any | None = None
        self._report_tokenizer_failed = False
        self.manual_triggers = ManualTriggerManager(runtime_config=runtime_config, runner=runner)
        self._scheduler_output_for_step: Any | None = None
        self._kv_dump_jobs: list[dict[str, Any]] = []
        self.detectors = DetectorManager(
            runtime_config=runtime_config,
            runner=runner,
            tokenizer_provider=self._get_detector_tokenizer,
            detection_gate=self.action_executor.can_run_detection,
            detection_skip_reason=self.action_executor.anomaly_check_skip_reason,
        )

    def _rebind_runner(self, runner: Any) -> None:
        """Point nested components at a new runner (same process, rare rebuild)."""
        if runner is self.runner:
            return
        logger.info("[runtime_guard] rebinding processor singleton to a new runner")
        ascend = runner.ascend_config
        runtime_config: RuntimeConfig = ascend.runtime_config
        self.runner = runner
        self.runtime_config = runtime_config
        self.action_executor.rebind_runner(runner, runtime_config=runtime_config)
        self.manual_triggers.rebind_runner(runner)
        self.detectors.rebind_runner(runner)
        # Tokenizer may differ across runners; force lazy reload.
        self._report_tokenizer = None
        self._report_tokenizer_failed = False
        self.action_executor.start()

    def shutdown(self) -> None:
        """Stop the async action worker (process teardown / tests)."""
        try:
            self.action_executor.stop()
        except Exception:
            logger.debug("[runtime_guard] action worker stop failed", exc_info=True)

    # ---- step entry (all ranks) --------------------------------------------

    def refresh_config(
        self,
        *,
        allow_arm: bool = True,
        scheduler_output: Any | None = None,
    ) -> bool:
        """All-rank runtime_config sync. Must not be skipped on early PP.

        ``allow_arm``: False on idle ``execute_dummy_batch``. Config sync still
        runs. ``dump.manual_dump`` is **not** consumed on the dummy path — only a real
        ``allow_arm=True`` wave may ``check_all`` / arm (avoids clearing the
        JSON flag with no dump when the service is idle or a peer DP is busy).
        """
        logger.debug("[runtime_guard sync] enter stage=refresh_config allow_arm=%s", allow_arm)
        so = scheduler_output if scheduler_output is not None else getattr(self, "_scheduler_output_for_step", None)
        prev_so = getattr(self, "_scheduler_output_for_step", None)
        self._scheduler_output_for_step = so
        try:
            changed = self._refresh_config_body(allow_arm=allow_arm, scheduler_output=so)
            # Clear IO wave cache only when something may append this step.
            # Runs *after* sync so a hot-reload that turns IO consumers on
            # still clears before sample/detect. Idle A (detectors off) skips.
            if self.runtime_config.needs_cumulative_io():
                RequestIoSnapshotManager.get().clear_wave_cache()
            return changed
        finally:
            self._scheduler_output_for_step = prev_so

    def _refresh_config_body(
        self,
        *,
        allow_arm: bool,
        scheduler_output: Any | None,
    ) -> bool:
        # Hot-reload off: config is static after init. Skip sync; still honor
        # startup manual_trigger on real waves.
        if not self.runtime_config.hot_reload_enabled:
            trigger = self.manual_triggers.consume_once(allow_arm=allow_arm, scheduler_output=scheduler_output)
            if trigger is not None:
                self._handle_manual_trigger(trigger)
            logger.debug("[runtime_guard sync] leave stage=refresh_config changed=False hot_reload=off")
            return False

        dump_jobs: list[Any] = []
        if self._dump_hitchhikes_on_config():
            extra_due, extra = self._claim_kv_dump_jobs()
            changed, dump_jobs = self.runtime_config.sync_with_hitchhike(
                extra_due=extra_due, extra=extra
            )
        else:
            changed = self.runtime_config.sync_runtime_config()
        if dump_jobs:
            self._run_kv_dumps(dump_jobs)
        # Fast path: no detector features active AND config unchanged
        # -> skip the ``if changed`` re-apply cascade.
        if not self.runtime_config.any_detector_enabled() and not changed:
            trigger = self.manual_triggers.consume_once(allow_arm=allow_arm, scheduler_output=scheduler_output)
            if trigger is not None:
                self._handle_manual_trigger(trigger)
            logger.debug(
                "[runtime_guard sync] leave stage=refresh_config changed=%s guard-inactive",
                changed,
            )
            return changed
        # Active guard (or just-changed): refresh detector deps only when JSON
        # content actually changed — unchanged polls skip the apply cascade.
        if changed:
            self.action_executor.apply_runtime_config()
            # ascend_log.level / modules live in the same JSON; propagate to this
            # rank (workers included) on hot-reload so DEBUG/etc reach TP/PP
            # workers, not just the API/EngineCore non-worker reloader.
            self.runtime_config.apply_ascend_log_level()
            # All ranks (incl. early-PP JSON writers) must run detector dep
            # checks so force-disable can persist when optional deps are missing.
            self.detectors.apply_runtime_config()
            self.report_writer.save_sensitive_info = self.runtime_config.report_save_sensitive_info()
            self.report_writer.max_prompt_token_ids = self.runtime_config.report_max_prompt_token_ids()
            self.report_writer.max_output_token_ids = self.runtime_config.report_max_output_token_ids()
            self.report_writer.decode_token_ids = self.runtime_config.report_decode_token_ids()
            self.report_writer.max_per_req = self.runtime_config.report_max_per_req()
        # Dump limits sync only when config changed (via apply_runtime_config).
        trigger = self.manual_triggers.consume_once(allow_arm=allow_arm, scheduler_output=scheduler_output)
        if trigger is not None:
            self._handle_manual_trigger(trigger)
        # Detector thresholds / enable flags: pulled lazily in each detector's
        # ``_precheck`` so we do not refresh every detector twice per step here.
        logger.debug("[runtime_guard sync] leave stage=refresh_config changed=%s", changed)
        return changed

    def sync_for_step(
        self,
        *,
        allow_arm: bool = True,
        scheduler_output: Any | None = None,
    ) -> None:
        """Lockstep runtime_guard sync for one engine wave (real step or idle dummy).

        ``refresh_config`` uses the per-DP sync group (or local file poll) and
        must run on every rank of that EngineCore each wave — including idle
        DP ranks that take ``execute_dummy_batch`` and skip ``execute_model``.
        Do not put this inside ``_dummy_run``: ``execute_model`` may already
        sync then call ``_dummy_run``. Never use a cross-DP full-world
        collective for config hot-reload.

        ``scheduler_output`` (optional): lets MRV2 arm ``manual_trigger`` on the
        first prefill wave before ``prepare_inputs`` populates ``req_states``.
        """
        runner = self.runner
        dp = getattr(runner, "dp_rank", "?")
        tp = getattr(runner, "tp_rank", "?")
        try:
            pp = get_pp_group().rank_in_group
        except Exception:
            pp = "?"
        logger.debug(
            "[runtime_guard sync] enter sync_for_step allow_arm=%s dp=%s tp=%s pp=%s",
            allow_arm,
            dp,
            tp,
            pp,
        )
        self._scheduler_output_for_step = scheduler_output
        try:
            self.wave_tracker.advance(allow_arm=allow_arm)
            cfg = self.runtime_config
            if not self._dump_hitchhikes_on_config():
                # File mode / hot-reload off: dump has its own last-PP TP sync.
                self._drain_kv_dump()
            # Idle shell + hot-reload off (interval<=0): config is static →
            # advance only. When hot-reload is on, always refresh so broadcast
            # mode can all_reduce every step (no per-rank wall-clock skip).
            idle = (
                not cfg.manual_trigger()
                and not cfg.needs_sample_phase_hooks()
            )
            if idle and not cfg.hot_reload_enabled:
                return
            self.refresh_config(allow_arm=allow_arm, scheduler_output=scheduler_output)
            # When no feature needs prompt cache / finished-IO reap, skip the
            # extra work (reload path still syncs above).
            if self.needs_sample_phase_hooks():
                self._cache_prompt_token_ids_from_scheduler_output(scheduler_output)
                # Idle / early-return finishes never reach check_after_sample; reap
                # finished reqs whose sample-wave FIFO is already drained.
                # Final-request-before-idle: on the trailing empty batch the
                # runner returns EMPTY_MODEL_RUNNER_OUTPUT, so sample_tokens →
                # run_sample_phase (and its mark_finished hook) never fires and
                # the last request is never marked finished. Mark it here.
                finished = getattr(scheduler_output, "finished_req_ids", None)
                if finished and int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) == 0:
                    self.mark_finished(finished)
                self._reap_finished_requests()
        finally:
            self._scheduler_output_for_step = None
            logger.debug(
                "[runtime_guard sync] leave sync_for_step allow_arm=%s dp=%s tp=%s pp=%s",
                allow_arm,
                dp,
                tp,
                pp,
            )

    def _dump_hitchhikes_on_config(self) -> bool:
        """True when dump jobs can ride the config broadcast group (PP=1)."""
        cfg = self.runtime_config
        if not cfg.hot_reload_enabled or cfg.sync_mode != SYNC_BROADCAST:
            return False
        group = _runtime_config_sync_group_or_none()
        return group is not None and int(getattr(group, "world_size", 1) or 1) > 1

    def _claim_kv_dump_jobs(self) -> tuple[bool, list[dict[str, Any]]]:
        """TP0 last-PP: take queued dump jobs. Other ranks return empty."""
        jobs = list(getattr(self, "_kv_dump_jobs", None) or [])
        if hasattr(self, "_kv_dump_jobs"):
            self._kv_dump_jobs.clear()
        if not should_dump_kv_on_rank(self.runner):
            return False, []
        try:
            is_tp0 = runner_tp_rank(self.runner) == 0
        except Exception:
            is_tp0 = True
        if not is_tp0 or not jobs:
            return False, []
        return True, jobs

    def queue_kv_dump(self, job: dict[str, Any]) -> bool:
        """TP0: record a dump job for last-PP all TP next wave.

        Same pending list: each ``req_id`` at most once (first arm wins).
        Returns True when the job was appended.
        """
        if not job or not job.get("req_id"):
            return False
        rid = str(job["req_id"])
        pending = getattr(self, "_kv_dump_jobs", None)
        if pending is None:
            self._kv_dump_jobs = []
            pending = self._kv_dump_jobs
        if any(str(j.get("req_id") or "") == rid for j in pending):
            return False
        pending.append(dict(job))
        return True

    def _drain_kv_dump(self) -> None:
        """File-mode / hot-reload-off dump sync on last-PP TP group.

        ``all_reduce(has_job)`` every step; ``broadcast_object`` only when due.
        """
        if not should_dump_kv_on_rank(self.runner):
            if hasattr(self, "_kv_dump_jobs"):
                self._kv_dump_jobs.clear()
            return
        payload = list(getattr(self, "_kv_dump_jobs", None) or [])
        if hasattr(self, "_kv_dump_jobs"):
            self._kv_dump_jobs.clear()
        try:
            tp_group = get_tp_group()
        except Exception:
            if payload:
                self._run_kv_dumps(payload)
            return
        tp_size = int(getattr(tp_group, "world_size", 1) or 1)
        if tp_size <= 1:
            if payload:
                self._run_kv_dumps(payload)
            return
        try:
            rank = int(tp_group.rank_in_group)
        except Exception:
            rank = 0
        cpu_group = getattr(tp_group, "cpu_group", None)
        device_group = getattr(tp_group, "device_group", None)
        gate_group = cpu_group if cpu_group is not None else device_group
        if gate_group is not None:
            try:
                import torch

                due_local = 1.0 if rank == 0 and payload else 0.0
                due_t = torch.tensor([due_local], dtype=torch.float32)
                torch.distributed.all_reduce(
                    due_t,
                    op=torch.distributed.ReduceOp.MAX,
                    group=gate_group,
                )
                if float(due_t.item()) < 0.5:
                    return
            except Exception:
                logger.exception("[runtime_guard soft-fail] kv dump all_reduce failed")
                return
        # No usable process group for has_job: must still broadcast_object so
        # non-TP0 ranks stay in lockstep (may be empty most steps).
        try:
            src_obj = payload if rank == 0 else None
            jobs = tp_group.broadcast_object(src_obj, src=0)
        except Exception:
            logger.exception("[runtime_guard soft-fail] kv dump broadcast failed")
            return
        if jobs:
            self._run_kv_dumps(jobs)

    def _run_kv_dumps(self, jobs: list[dict[str, Any]]) -> None:
        from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader

        ex = getattr(self, "action_executor", None)
        reader = getattr(ex, "_kv_reader", None) or KvCacheReader(self.runner)
        submit = getattr(ex, "_submit_heavy", None)
        dump_root = Path(self.runtime_config.dump_root())
        rank_tag = dump_rank_tag(self.runner)
        store = RequestGuardStore.get()
        quota = getattr(self, "quota", None)
        try:
            is_tp0 = runner_tp_rank(self.runner) == 0
        except Exception:
            is_tp0 = True
        # One prepare may queue N jobs after a single try_consume (arm_id).
        # Refund once per arm if that arm produced no D2H snapshot. Later async
        # torch.save failure does not refund (see ops docs).
        from collections import defaultdict

        from vllm_ascend.runtime_guard.util import kv_dump_wave_dirname

        arms: dict[str, dict[str, bool]] = defaultdict(lambda: {"debited": False, "ok": False})
        seen_req: set[str] = set()
        for job in jobs:
            arm_id = str(job.get("arm_id") or f"job-{id(job)}")
            if job.get("consume_quota"):
                arms[arm_id]["debited"] = True
            req_id = str(job.get("req_id") or "")
            if not req_id:
                continue
            if req_id in seen_req:
                continue
            seen_req.add(req_id)
            incident_type = str(job.get("incident_type") or "unknown")
            wave = job.get("wave")
            try:
                wave_i = int(wave) if wave is not None else None
            except (TypeError, ValueError):
                wave_i = None
            wave_dir = kv_dump_wave_dirname(wave_i)
            if not store.kv_dump_allowed(req_id):
                if is_tp0:
                    from vllm_ascend.runtime_guard.util import write_kv_dump_skipped_finished

                    write_kv_dump_skipped_finished(
                        dump_root,
                        req_id=req_id,
                        incident_type=incident_type,
                        stage="drain",
                        rank_tag=rank_tag,
                    )
                continue
            block_ids = list(block_ids_for_request(self.runner, req_id, None) or [])
            if not block_ids:
                logger.warning(
                    "[runtime_guard dump_kv] skip empty local block_ids req_id=%s rank=%s",
                    req_id,
                    rank_tag,
                )
                continue
            out_dir = dump_root / incident_type / req_id / wave_dir / rank_tag
            produced = 0
            try:
                for snap in reader.iter_request_snapshots(
                    req_id=req_id,
                    block_ids=block_ids,
                    out_dir=out_dir,
                ):
                    produced += 1
                    if submit is not None:
                        submit(lambda s=snap: KvCacheReader.write_snapshots([s]))
                    else:
                        KvCacheReader.write_snapshots([snap])
            except Exception:
                logger.exception(
                    "[runtime_guard dump_kv] dump failed req_id=%s rank=%s",
                    req_id,
                    rank_tag,
                )
                continue
            if produced == 0:
                logger.warning(
                    "[runtime_guard dump_kv] no tensors req_id=%s rank=%s",
                    req_id,
                    rank_tag,
                )
                continue
            arms[arm_id]["ok"] = True
            logger.info(
                "[runtime_guard dump_kv] dumped req_id=%s wave=%s rank=%s files=%d dir=%s",
                req_id,
                wave_dir,
                rank_tag,
                produced,
                out_dir,
            )
        if is_tp0 and quota is not None:
            for meta in arms.values():
                if meta["debited"] and not meta["ok"]:
                    quota.refund(consume_quota=True)

    def _cache_prompt_token_ids_from_scheduler_output(
        self,
        scheduler_output: Any | None,
    ) -> None:
        """Capture prompt_token_ids from each scheduled new request.

        Bug #9 fix: v2 ``RequestState.all_token_ids`` is a StagedWriteTensor
        whose host mirror stays 0 until ``apply_staged_writes`` commits, and
        ``_remove_request`` pops the id from ``req_id_to_index`` on finish —
        both windows make the snapshot path return zeros or empty. Cache the
        ids here on the first prefill wave so later snapshots (crash or
        finish) read the real prompt. Idempotent per request.
        """
        if scheduler_output is None:
            return
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
        if not new_reqs:
            return
        store = RequestGuardStore.get()
        for req in new_reqs:
            req_id = getattr(req, "req_id", None)
            if not req_id:
                continue
            ids = getattr(req, "prompt_token_ids", None)
            if ids is None:
                ids = getattr(req, "prefill_token_ids", None)
            if ids is None:
                continue
            store.set_prompt_token_ids(str(req_id), ids)

    # ---- sample / get_output hooks ----------------------------------------

    def mark_finished(self, finished_req_ids: Any) -> None:
        """Mark requests finished; defer Store.clear until last sample is consumed.

        Runner order is ``mark_finished`` → (optional) ``record_sample_waves``
        → ``check_after_sample`` / async ``get_output``. Only Store
        :meth:`~RequestGuardStore.mark_finished` runs here so the last stamp /
        detect / append still see the same state. Sidecars + clear happen in
        :meth:`_reap_finished_requests`.
        """
        if not finished_req_ids:
            return
        store = RequestGuardStore.get()
        wave_tracker = getattr(self, "wave_tracker", None)
        wave = 0
        if wave_tracker is not None:
            try:
                wave = int(wave_tracker.current_wave())
            except (TypeError, ValueError):
                wave = 0
        store.mark_finished(finished_req_ids, wave=wave)

    def _reap_finished_requests(self) -> None:
        """Optionally log finish output and clear reqs that are finished and drained."""
        store = RequestGuardStore.get()
        wave_tracker = getattr(self, "wave_tracker", None)
        wave = 0
        if wave_tracker is not None:
            try:
                wave = int(wave_tracker.current_wave())
            except (TypeError, ValueError):
                wave = 0
        reapable = store.list_reapable(current_wave=wave)
        if not reapable:
            return
        io_mgr = RequestIoSnapshotManager.get()
        if self.runtime_config.log_print_output_on_finish():
            self._maybe_print_output_on_finish(reapable, io_mgr)
        store.clear_many(reapable, detectors=self.detectors)
        if wave_tracker is not None:
            wave_tracker.discard_many(reapable)

    def _maybe_print_output_on_finish(self, finished_req_ids: Any, io_mgr: RequestIoSnapshotManager) -> None:
        """Log output_token_ids + text for finished reqs (TP0 only).

        Content comes from runtime_guard cumulative IO accumulated while
        ``log.print_output_on_finish`` was true on sample steps (no historical
        backfill). Mid-request hot-enable may print a partial sequence or
        ``output_token_count=0`` / empty text if nothing was appended after
        enable. See ``RuntimeConfig.log_print_output_on_finish``.
        """
        runner = self.runner
        try:
            if int(getattr(runner, "tp_rank", 0)) != 0:
                return
        except Exception:
            return
        tokenizer = self._get_detector_tokenizer()
        max_ids = self.runtime_config.report_max_output_token_ids()
        for req_id in finished_req_ids:
            if not req_id:
                continue
            snap = io_mgr.snapshot(runner, req_id, None, include_token_ids=True, use_cache=False)
            ids = list(snap.output_token_ids or [])
            truncated = False
            if max_ids > 0 and len(ids) > max_ids:
                ids = ids[:max_ids]
                truncated = True
            text = ""
            if tokenizer is not None and ids:
                try:
                    text = decode_token_ids(tokenizer, ids)
                except Exception as exc:
                    text = f"<decode failed: {exc}>"
            elif tokenizer is None:
                text = "<tokenizer unavailable>"
            logger.info(
                "[runtime_guard print_output] req_id=%s output_token_count=%d truncated=%s output_token_ids=%s output_text=%r",
                req_id,
                snap.output_token_count,
                truncated,
                ids,
                text,
            )

    def should_check_after_spec(self) -> bool:
        if not self.action_executor.can_run_detection():
            return False
        return self.detectors.any_enabled_for_spec()

    def needs_sample_phase_hooks(self) -> bool:
        """True when sample-phase runtime_guard hooks must run (else pure ``sample_fn``).

        Missing ``runtime_config`` (bare test doubles) defaults to True so
        soft-fail / wiring tests still exercise the hook chain.
        """
        cfg = getattr(self, "runtime_config", None)
        if cfg is None:
            return True
        return bool(cfg.needs_sample_phase_hooks())

    def _soft_fail(self, hook: str, fn: Callable[[], Any]) -> Any:
        # Guard hooks are observational: any exception must stay inside the
        # guard and never reach the engine loop / async copy thread.
        try:
            return fn()
        except Exception:
            logger.exception("[runtime_guard soft-fail] hook=%s raised; skipped this step", hook)
            return None

    def check_after_spec(
        self,
        sampled_tokens: Any,
        accepted_token_nums: Any,
    ) -> None:
        """Speculative step hook: record accepted tokens + run registered spec detectors.

        Detection gating (rank / dump / detector-on) lives in ``DetectorManager``.
        """
        if inject.ENABLED:
            inject.inject_after_spec(accepted_token_nums)

        def _run() -> None:
            if not self.should_check_after_spec():
                return
            for alert in self.detectors.check_after_spec(sampled_tokens, accepted_token_nums):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

        self._soft_fail("check_after_spec", _run)

    def record_sample_waves(self, req_ids: list[str] | None) -> None:
        self.wave_tracker.record_sample_waves(req_ids)

    def _should_record_sample_waves(self, *, use_async: bool) -> bool:
        """Async: only TP0 (output rank) stamps — matches AscendAsync* wrap.

        Non-TP0 never runs ``get_output`` / ``take_sample_wave`` under async
        scheduling; recording there would leave ``pending`` stamps until
        ``max_deferred_waves`` force-reap.
        """
        if not use_async:
            return True
        try:
            return runner_tp_rank(self.runner) == 0
        except Exception:
            return True

    # ---- single sink for post-pre-sample runtime_guard hooks ---------------

    def run_sample_phase(
        self,
        *,
        sample_fn: Callable[[], "SamplePhaseResult"],
        speculative_config: Any,
        need_accepted_tokens: bool,
        use_async: bool,
        async_state_update_fn: Callable[["SamplePhaseResult"], None] | None = None,
        routed_experts_fn: Callable[["SamplePhaseResult"], Any] | None = None,
        accepted_token_nums_fn: Callable[["SamplePhaseResult"], Any] | None = None,
    ) -> tuple["SamplePhaseResult", Any]:
        """Single sink for post-pre-sample runtime_guard hooks.

        Replaces 7 inline ``self.runtime_guard.*`` calls scattered across
        ``NPUModelRunner.sample_tokens`` with one orchestration call so
        hook ordering is owned by ``RuntimeGuardProcessor`` rather than the runner.

        Hook 1 (``check_before_sample``) stays explicit in the runner because
        it must fire BEFORE ``apply_grammar_bitmask`` (a source-level contract
        enforced by ``test_v1_sample_tokens_checks_before_grammar_bitmask``).

        Hook sequence (``S1`` golden path):
            2. ``sample_fn()`` returns :class:`SamplePhaseResult`
            3. ``mark_finished``
            -> ``async_state_update_fn`` (only if ``need_accepted_tokens``)
            4. ``check_after_spec`` (spec only; ``accepted_token_nums_fn`` for branch)
            -> ``routed_experts_fn`` (async path: BEFORE wave stamp; sync: AFTER check_after_sample)
            5. ``record_sample_waves``
            6. ``check_after_sample`` (sync path only; async via AscendAsync* ``get_output``)

        Native KV capture uses ``dump_kv`` actions only (fully decoupled from
        Ascend/msprobe PrecisionDebugger dump).
        """
        # Idle fast-path: detectors / print_output all off →
        # skip soft-fail wrappers and observational hooks entirely.
        if not self.needs_sample_phase_hooks():
            result = sample_fn()
            if need_accepted_tokens and async_state_update_fn is not None:
                async_state_update_fn(result)
            routed_experts_result = None
            if routed_experts_fn is not None:
                routed_experts_result = routed_experts_fn(result)
            return result, routed_experts_result

        # Runner's sample work (sample + draft + bookkeeping + output + profiling + eplb)
        result = sample_fn()
        # Hook 3: mark_finished
        self._soft_fail("mark_finished", lambda: self.mark_finished(result.finished_req_ids))
        # Async state update callback (between mark_finished and check_after_spec)
        if need_accepted_tokens and async_state_update_fn is not None:
            async_state_update_fn(result)
        # Hook 4: check_after_spec (spec only)
        if speculative_config is not None and self.should_check_after_spec():
            if accepted_token_nums_fn is not None:
                accepted_token_nums = accepted_token_nums_fn(result)
            else:
                accepted_token_nums = None
            self._soft_fail(
                "check_after_spec",
                lambda: self.check_after_spec(
                    sampled_tokens=result.sampler_output.sampled_token_ids,
                    accepted_token_nums=accepted_token_nums,
                ),
            )
        # Async path: routed_experts computed BEFORE wave stamp
        routed_experts_result = None
        if use_async and routed_experts_fn is not None:
            routed_experts_result = routed_experts_fn(result)
        # Hook 5: record_sample_waves (sync: all ranks; async: output-rank TP0 only)
        if self._should_record_sample_waves(use_async=use_async):
            self._soft_fail(
                "record_sample_waves",
                lambda: self.record_sample_waves(result.req_ids_output_copy),
            )
        # Hook 6: check_after_sample (sync path only)
        if not use_async:
            self._soft_fail(
                "check_after_sample",
                lambda: self.check_after_sample(
                    sampled_token_ids=result.valid_sampled_token_ids,
                    req_ids=result.req_ids_output_copy,
                ),
            )
        # Sync path: routed_experts computed AFTER check_after_sample
        if not use_async and routed_experts_fn is not None:
            routed_experts_result = routed_experts_fn(result)
        return result, routed_experts_result

    def check_before_sample(
        self,
        *,
        scheduler_output: Any,
        logits: Any,
        logits_indices: Any = None,
        input_batch: Any = None,
        **_unused: Any,
    ) -> None:
        """Pre-sample hook: ``logits_finite`` (and future pre-sample detectors)."""
        del scheduler_output, _unused
        self._last_input_batch = input_batch
        if inject.ENABLED:
            inject.inject_before_sample(logits)

        def _run() -> None:
            for alert in self.detectors.check_before_sample(
                logits=logits,
                logits_indices=logits_indices,
                input_batch=input_batch,
            ):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

        self._soft_fail("check_before_sample", _run)

    def check_after_sample(
        self,
        sampled_token_ids: Any,
        req_ids: list[str] | None = None,
    ) -> None:
        """Sample-step hook: drain logits_finite + enqueue CPU detect.

        ``logits_finite`` already ``.item()``'d / resolved on before-sample;
        here we drain those incidents. Wave stamp + IO append stay on this
        thread (sync sample or async ``get_output``).
        ``token_repeat`` / ``output_substring`` run later on ActionQueue;
        request finish does not wait. ``dump_kv`` (any detector) is skipped
        if the request is already finished/reaped.
        """
        if inject.ENABLED:
            inject.inject_after_sample(sampled_token_ids, runner=self.runner)

        def _run() -> None:
            wave_by_req: dict[str, int] = {}
            wave_tracker = getattr(self, "wave_tracker", None)
            runner = getattr(self, "runner", None)
            async_sched = bool(getattr(runner, "use_async_scheduling", False)) if runner is not None else False
            if wave_tracker is not None:
                ids = list(req_ids) if req_ids else []
                if async_sched and not ids:
                    logger.warning_once(
                        "[runtime_guard wave] async check_after_sample without req_ids; "
                        "arm_wave will fall back to current_wave (may race advance_wave)"
                    )
                for rid in ids:
                    if not rid:
                        continue
                    rid_s = str(rid)
                    stamped = wave_tracker.take_sample_wave(rid_s)
                    if stamped is not None:
                        wave_by_req[rid_s] = stamped
                    elif async_sched:
                        logger.warning(
                            "[runtime_guard wave] missing sample-wave stamp for req_id=%s under async "
                            "scheduling; arm_wave falls back to current_wave (may be polluted)",
                            rid_s,
                        )
            logits_alerts, snap = self.detectors.after_sample_hot_path(
                sampled_token_ids,
                req_ids=req_ids,
            )
            self._log_sampling_meta_debug(req_ids)
            for alert in logits_alerts:
                arm_wave = wave_by_req.get(alert.req_id) if alert.req_id else None
                self._handle_alert(
                    alert,
                    detector=self.detectors.get(alert.incident_type),
                    arm_wave=arm_wave,
                )
            if snap is not None:
                self._enqueue_after_sample_cpu(snap, wave_by_req)
            self._reap_finished_requests()

        self._soft_fail("check_after_sample", _run)

    def _enqueue_after_sample_cpu(
        self,
        snap: Any,
        wave_by_req: dict[str, int],
    ) -> None:
        """Submit CPU after-sample detect; never run inline on get_output."""
        req_ids_job = [rid for rid in snap.req_ids if rid]
        store = RequestGuardStore.get()
        store.add_cpu_jobs(req_ids_job)

        def _cpu_job(
            snap: Any = snap,
            wave_by_req: dict[str, int] = wave_by_req,
            req_ids_job: list[str] = req_ids_job,
        ) -> None:
            try:
                for alert in self.detectors.run_after_sample_cpu(snap):
                    arm_wave = wave_by_req.get(alert.req_id) if alert.req_id else None
                    self._handle_alert(
                        alert,
                        detector=self.detectors.get(alert.incident_type),
                        arm_wave=arm_wave,
                    )
            except Exception:
                logger.exception("[runtime_guard] after-sample CPU detect failed")
            finally:
                RequestGuardStore.get().finish_cpu_jobs(req_ids_job)
                self._reap_finished_requests()

        queue = getattr(getattr(self, "action_executor", None), "action_queue", None)
        ok = False
        if queue is not None:
            ok = bool(queue.submit(_cpu_job, drop_on_full=True))
        if not ok:
            store.finish_cpu_jobs(req_ids_job)


    def _handle_alert(
        self,
        alert: Incident,
        *,
        detector: AnomalyDetector | None = None,
        write_report: bool = True,
        arm_wave: int | None = None,
        action_override: list[str] | None = None,
    ) -> None:
        if not write_report and action_override is None:
            return
        if alert.block_ids is None or not alert.block_ids:
            alert.block_ids = block_ids_for_request(
                self.runner,
                alert.req_id,
                alert.req_idx,
                input_batch=getattr(self, "_last_input_batch", None),
            )
        if arm_wave is not None:
            alert.wave = arm_wave
        elif alert.wave is None:
            alert.wave = self.wave_tracker.current_wave()
        if detector is not None:
            detector.on_alert_armed(alert)
        detail = alert.to_report_detail()
        include_ids = self.runtime_config.report_save_sensitive_info()
        io_mgr = RequestIoSnapshotManager.get()
        snap = io_mgr.snapshot(
            self.runner,
            alert.req_id,
            alert.req_idx,
            include_token_ids=include_ids,
            scheduler_output=getattr(self, "_scheduler_output_for_step", None),
        )
        detail = io_mgr.merge_into_detail(detail, snap)
        detail = self._enrich_detail_with_block_meta(
            detail,
            alert.req_id,
            alert.req_idx,
        )
        self.action_executor.handle(
            alert,
            detail=detail,
            tokenizer=self._get_report_tokenizer(),
            action_override=action_override,
            write_report=write_report,
        )

    def _handle_manual_trigger(
        self,
        trigger: TriggerEvent,
        *,
        write_report: bool = True,
    ) -> None:
        # Manual triggers stay leader-only: single-writer for the JSON quota
        # consume (#5) and for manual reports/dumps. Auto detection is also
        # last-PP TP0 (same rank as this leader).
        if not is_action_leader_rank(self.runner):
            return
        batch_rows = self._batch_request_io_rows()
        include_ids = self.runtime_config.report_save_sensitive_info()
        io_mgr = RequestIoSnapshotManager.get()
        so = getattr(self, "_scheduler_output_for_step", None)
        requests_detail: list[dict[str, Any]] = []
        for req_id, req_idx in batch_rows:
            snap = io_mgr.snapshot(
                self.runner,
                req_id,
                req_idx,
                include_token_ids=include_ids,
                scheduler_output=so,
            )
            entry = {"req_id": req_id, "req_idx": req_idx}
            entry.update(snap.as_detail_fields())
            entry = self._enrich_detail_with_block_meta(entry, req_id, req_idx)
            requests_detail.append(entry)
        detail = trigger.to_report_detail()
        detail["num_requests"] = len(requests_detail)
        detail["requests"] = requests_detail
        block_ids = []
        if batch_rows:
            block_ids = block_ids_for_request(self.runner, batch_rows[0][0], batch_rows[0][1])
        incident = Incident(
            incident_type=trigger.trigger_type,
            req_id=trigger.req_id,
            detail=detail,
            consume_quota=False,
            block_ids=block_ids,
            wave=self.wave_tracker.current_wave(),
        )
        # on_trigger: detector.manual_trigger.on_trigger, else actions.defaults.
        # dump_kv is always injected for manual (all_requests); see ActionExecutor.
        self.action_executor.handle(
            incident,
            detail=detail,
            tokenizer=self._get_report_tokenizer(),
            write_report=write_report,
        )
        # One scheduled wave → one consume (after prepare so dump still sees
        # dump_enabled during arm). Continuous ``manual_dump: true`` no-ops.
        if self.runtime_config.consume_manual_trigger():
            logger.info(
                "[runtime_guard manual_trigger] manual_dump consumed, remaining=%d",
                self.runtime_config.manual_trigger_count(),
            )

    def _enrich_detail_with_block_meta(
        self,
        detail: dict[str, Any],
        req_id: str,
        req_idx: int | None = None,
    ) -> dict[str, Any]:
        """Attach ``block_ids`` / ``slot_mapping`` per report.* flags."""
        include_ids = self.runtime_config.report_include_block_ids()
        include_slots = self.runtime_config.report_include_slot_mapping()
        if not include_ids and not include_slots:
            return detail
        out = dict(detail)
        if include_ids:
            out["block_ids"] = block_ids_for_request(self.runner, req_id, req_idx)
        if include_slots:
            got = slot_mapping_for_request(
                self.runner,
                req_id,
                req_idx,
                scheduler_output=getattr(self, "_scheduler_output_for_step", None),
            )
            if got is not None:
                values, span = got
                out["slot_mapping"] = values
                out["slot_mapping_span"] = [span[0], span[1]]
        return out

    def _batch_request_io_rows(self) -> list[tuple[str, int]]:
        """``(req_id, req_idx)`` for every request currently in the local batch."""
        return iter_local_request_rows(
            self.runner,
            getattr(self, "_scheduler_output_for_step", None),
        )

    def _get_detector_tokenizer(self) -> Any | None:
        """Tokenizer for detectors that need encode/decode (not gated on report flags)."""
        if self._report_tokenizer is not None:
            return self._report_tokenizer
        if self._report_tokenizer_failed:
            return None
        runner = getattr(self, "runner", None)
        try:
            tok = load_model_tokenizer(runner)
        except Exception as exc:
            self._report_tokenizer_failed = True
            logger.warning("[runtime_guard] tokenizer load failed error=%s", exc)
            return None
        if tok is None:
            # runner / model_config missing; retry on next call.
            return None
        self._report_tokenizer = tok
        return self._report_tokenizer

    def _get_report_tokenizer(self) -> Any | None:
        """Lazy-load tokenizer for report decode (detect rank only, once)."""
        if not self.runtime_config.report_save_sensitive_info() or not self.runtime_config.report_decode_token_ids():
            return None
        return self._get_detector_tokenizer()

    def _log_sampling_meta_debug(self, req_ids: list[str] | None) -> None:
        """DEBUG ``[SamplingMeta]`` for local-batch reqs (TP0 + last PP).

        Gated by logger level only (no JSON switch). Skips ``.item()`` work when
        DEBUG is off. Soft-fail: never raise into the sample path.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            self._emit_sampling_meta_debug(req_ids)
        except Exception as exc:
            logger.debug("[runtime_guard] SamplingMeta log failed: %s", exc, exc_info=True)

    def _emit_sampling_meta_debug(self, req_ids: list[str] | None) -> None:
        runner = getattr(self, "runner", None)
        if runner is None:
            return
        if int(getattr(runner, "tp_rank", 0)) != 0:
            return
        if not get_pp_group().is_last_rank:
            return

        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None:
            return
        sampling_metadata = getattr(input_batch, "sampling_metadata", None)
        if sampling_metadata is None:
            return

        batch_ids = list(getattr(input_batch, "req_ids", None) or [])
        want = {str(r) for r in (req_ids or []) if r} if req_ids else set(batch_ids)
        if not want:
            return

        for req_idx, req_id in enumerate(batch_ids):
            if not req_id or str(req_id) not in want:
                continue

            temp = sampling_metadata.temperature[req_idx].item() if sampling_metadata.temperature is not None else None
            topk = sampling_metadata.top_k[req_idx].item() if sampling_metadata.top_k is not None else None
            topp = sampling_metadata.top_p[req_idx].item() if sampling_metadata.top_p is not None else None

            freq_pen = sampling_metadata.frequency_penalties[req_idx].item()
            pres_pen = sampling_metadata.presence_penalties[req_idx].item()
            rep_pen = sampling_metadata.repetition_penalties[req_idx].item()

            bad_words = sampling_metadata.bad_words_token_ids
            req_bad_words = bad_words.get(req_idx, []) if bad_words else []
            req_output_tokens = (
                sampling_metadata.output_token_ids[req_idx]
                if sampling_metadata.output_token_ids and req_idx < len(sampling_metadata.output_token_ids)
                else []
            )
            req_spec_tokens = (
                sampling_metadata.spec_token_ids[req_idx]
                if sampling_metadata.spec_token_ids and req_idx < len(sampling_metadata.spec_token_ids)
                else None
            )
            if sampling_metadata.logprob_token_ids:
                req_logprob_tokens = sampling_metadata.logprob_token_ids.get(req_idx, [])
            else:
                req_logprob_tokens = None

            logger.debug(
                "[SamplingMeta] req_id=%s req_idx=%d "
                "dp_rank=%d tp_rank=%d "
                "temperature=%.4f top_k=%s top_p=%.4f "
                "freq_pen=%.4f pres_pen=%.4f rep_pen=%.4f "
                "bad_words_group_num=%d output_tokens_len=%d spec_tokens_len=%s logprob_target_tokens_len=%s "
                "all_greedy=%s all_random=%s max_num_logprobs=%s",
                req_id,
                req_idx,
                runner.dp_rank,
                runner.tp_rank,
                temp if temp is not None else -1,
                topk,
                topp if topp is not None else 1.0,
                freq_pen,
                pres_pen,
                rep_pen,
                len(req_bad_words),
                len(req_output_tokens),
                len(req_spec_tokens) if req_spec_tokens else None,
                len(req_logprob_tokens) if req_logprob_tokens else None,
                sampling_metadata.all_greedy,
                sampling_metadata.all_random,
                sampling_metadata.max_num_logprobs,
            )
