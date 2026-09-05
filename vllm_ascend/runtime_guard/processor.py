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

"""Process-wide runtime-guard processing (config sync, detectors, dump/report).

Owns construction of detectors / ``ReportWriter`` / actions. Model runners
``bind`` the process singleton once; other call sites use
:meth:`RuntimeGuardProcessor.get`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.detector.base import AnomalyDetector
from vllm_ascend.runtime_guard.detector.manager import DetectorManager
from vllm_ascend.runtime_guard.detector.position_alignment import num_computed_before

from vllm_ascend.runtime_guard.input_filters import InputFilterManager, iter_batch_prompt_token_ids
from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.runtime_guard.kv_block_meta import (
    KvBlockMetaTracker,
    block_ids_for_request,
    resolve_block_size,
    slot_mapping_for_request,
    slots_for_block_ids,
    touched_block_ids,
)
from vllm_ascend.runtime_guard.manual_trigger import (
    ManualTriggerManager,
    TriggerEvent,
    iter_local_request_rows,
)
from vllm_ascend.runtime_guard.rank_gate import dump_rank_tag, is_action_leader_rank
from vllm_ascend.runtime_guard.report import ReportWriter
from vllm_ascend.runtime_guard.request_state import RequestGuardStore
from vllm_ascend.runtime_guard.tokenizer import load_model_tokenizer
from vllm_ascend.runtime_guard.util import decode_token_ids
from vllm_ascend.runtime_guard.action.executor import ActionExecutor
from vllm_ascend.runtime_guard.quota import DumpQuota
from vllm_ascend.runtime_guard.wave_tracker import WaveTracker
from vllm_ascend.logger import init_logger_ascend

# How many leading prompt token ids to suggest as a filter prefix example.
_PRINT_INPUT_PREFIX_HINT_LEN = 8

if TYPE_CHECKING:
    from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


def _block_kv_violated_block_ids(detail: dict[str, Any]) -> list[int]:
    """Block ids that actually appear in ``block_kv`` ``violations`` (errored only)."""
    raw = detail.get("violations")
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for row in raw:
        if not isinstance(row, dict):
            continue
        if "block_id" not in row or not row.get("violation"):
            continue
        try:
            bid = int(row["block_id"])
        except (TypeError, ValueError):
            continue
        if bid in seen:
            continue
        seen.add(bid)
        out.append(bid)
    return out


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
    logprobs_lists: Any
    req_ids_output_copy: Any
    invalid_req_indices: Any
    finished_req_ids: Any
    hidden_states: Any = None
    spec_decode_metadata: Any = None


class RuntimeGuardProcessor:
    """Process-wide singleton: config → filter → detect → dump/report.

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
        # Leader materializes JSON once; path is already logged at AscendConfig /
        # Dumper ready. Non-leaders no-op inside ensure_persisted.
        runtime_config.ensure_persisted()
        # Sole owner of InputFilterManager refresh (init + refresh_config).
        InputFilterManager.get().apply_from_config(runtime_config)
        # Runtime config is solely ``runtime_config`` (JSON).
        self.wave_tracker = WaveTracker()
        # B1: reap must wait for the last async output of a finished request;
        # WaveTracker.pending is the drain signal (replaces the dead store FIFO).
        RequestGuardStore.get().set_drain_probe(self.wave_tracker.pending)
        self.quota = DumpQuota(runtime_config)
        # Plan A report routing: the leader keeps the legacy report dir; every
        # other detection-eligible rank writes into its own rank-tagged
        # subdir, so per-rank reports/dumps never interleave. Analysis tools
        # aggregate <report_dir>/<incident_type>/ plus <report_dir>/rank_*/<...>.
        report_dir = runtime_config.report_dir
        if not is_action_leader_rank(runner):
            report_dir = str(Path(report_dir) / f"rank_{dump_rank_tag(runner)}")
        self.report_writer = ReportWriter(
            report_dir,
            save_sensitive_info=runtime_config.report_save_sensitive_info(),
            max_prompt_token_ids=runtime_config.report_max_prompt_token_ids(),
            max_output_token_ids=runtime_config.report_max_output_token_ids(),
            decode_token_ids=runtime_config.report_decode_token_ids(),
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
        """All-rank DFX config sync. Must not be skipped on early PP.

        ``allow_arm``: False on idle ``execute_dummy_batch``. Config sync still
        runs. ``dump.manual_trigger`` is **not** consumed on the dummy path — only a real
        ``allow_arm=True`` wave may ``check_all`` / arm (avoids clearing the
        JSON flag with no dump when the service is idle or a peer DP is busy).
        """
        logger.debug("[runtime_guard sync] enter stage=refresh_config allow_arm=%s", allow_arm)
        # Always advance the IO append-wave frontier (same-wave dedupe).
        RequestIoSnapshotManager.get().clear_wave_cache()
        so = scheduler_output if scheduler_output is not None else getattr(self, "_scheduler_output_for_step", None)
        prev_so = getattr(self, "_scheduler_output_for_step", None)
        self._scheduler_output_for_step = so
        try:
            return self._refresh_config_body(allow_arm=allow_arm, scheduler_output=so)
        finally:
            self._scheduler_output_for_step = prev_so

    def _refresh_config_body(
        self,
        *,
        allow_arm: bool,
        scheduler_output: Any | None,
    ) -> bool:
        # Hot-reload off: config/filters are static after init. Skip sync +
        # filter rebuild; still honor startup manual_trigger /
        # print_input on real waves.
        if not self.runtime_config.hot_reload_enabled:
            trigger = self.manual_triggers.consume_once(allow_arm=allow_arm, scheduler_output=scheduler_output)
            if trigger is not None:
                self._handle_manual_trigger(trigger)
            self.maybe_print_input_token_ids_once(allow_arm=allow_arm)
            logger.debug("[runtime_guard sync] leave stage=refresh_config changed=False hot_reload=off")
            return False

        changed = self.runtime_config.sync_runtime_config()
        # Fast path: no detector/filter features active AND config unchanged
        # -> skip filter rebuild and the ``if changed`` re-apply cascade.
        # ``sync_runtime_config`` already ran so a hot-reload that turns a
        # detector on is observed next step and re-evaluates this gate.
        if not self.runtime_config.needs_filter_chain_apply() and not changed:
            trigger = self.manual_triggers.consume_once(allow_arm=allow_arm, scheduler_output=scheduler_output)
            if trigger is not None:
                self._handle_manual_trigger(trigger)
            self.maybe_print_input_token_ids_once(allow_arm=allow_arm)
            logger.debug(
                "[runtime_guard sync] leave stage=refresh_config changed=%s guard-inactive",
                changed,
            )
            return changed
        # Active guard (or just-changed): rebuild filter / detector deps only
        # when JSON content actually changed — unchanged polls skip
        # ``normalize_input_filter_configs`` and the apply cascade.
        if changed:
            InputFilterManager.get().apply_from_config(self.runtime_config)
            self.action_executor.apply_runtime_config()
            # All ranks (incl. early-PP JSON writers) must run detector dep
            # checks so force-disable can persist when msprobe is missing.
            self.detectors.apply_runtime_config()
            self.report_writer.save_sensitive_info = self.runtime_config.report_save_sensitive_info()
            self.report_writer.max_prompt_token_ids = self.runtime_config.report_max_prompt_token_ids()
            self.report_writer.max_output_token_ids = self.runtime_config.report_max_output_token_ids()
            self.report_writer.decode_token_ids = self.runtime_config.report_decode_token_ids()
        # Dump limits sync only when config changed (via apply_runtime_config).
        trigger = self.manual_triggers.consume_once(allow_arm=allow_arm, scheduler_output=scheduler_output)
        if trigger is not None:
            self._handle_manual_trigger(trigger)
        self.maybe_print_input_token_ids_once(allow_arm=allow_arm)
        # Detector thresholds / enable flags: pulled lazily in each detector's
        # ``_precheck`` (and ``ensure_logprobs_for_detection``) so we do not
        # refresh every detector twice per step here.
        logger.debug("[runtime_guard sync] leave stage=refresh_config changed=%s", changed)
        return changed

    def maybe_print_input_token_ids_once(self, *, allow_arm: bool = True) -> bool:
        """If ``input_filter.print_input_token_ids_once``, log prompts once then clear.

        Skipped on dummy waves (``allow_arm=False``) and when the batch has no
        printable prompts (flag stays true for the next real request).
        Logging is TP0-only; all ranks consume together when printing succeeds.
        """
        if not allow_arm or not self.runtime_config.print_input_token_ids_once():
            return False
        rows = iter_batch_prompt_token_ids(
            self.runner,
            scheduler_output=getattr(self, "_scheduler_output_for_step", None),
        )
        if not rows:
            logger.debug(
                "[runtime_guard print_input] print_input_token_ids_once set but no prompt_token_ids in batch; defer consume"
            )
            return False
        log_leader = int(getattr(self.runner, "tp_rank", 0)) == 0
        if log_leader:
            for req_id, req_idx, ids in rows:
                prompt_len = len(ids)
                hint_n = min(_PRINT_INPUT_PREFIX_HINT_LEN, prompt_len)
                prefix_hint = ids[:hint_n]
                logger.info(
                    "[runtime_guard print_input] req_id=%s req_idx=%s length=%d prompt_token_ids=%s",
                    req_id,
                    req_idx,
                    prompt_len,
                    ids,
                )
                if hint_n > 0:
                    logger.info(
                        "[runtime_guard print_input] filter hint req_id=%s length=%d "
                        "input_filter.filters example: "
                        '{"type":"input_token_id_prefix","mode":"include",'
                        '"prefixes":[%s]}',
                        req_id,
                        prompt_len,
                        prefix_hint,
                    )
        self.runtime_config.consume_print_input_token_ids_once()
        return True


    def sync_for_step(
        self,
        *,
        allow_arm: bool = True,
        scheduler_output: Any | None = None,
    ) -> None:
        """Lockstep DFX sync for one engine wave (real step or idle dummy).

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
            self.refresh_config(allow_arm=allow_arm, scheduler_output=scheduler_output)
            # When no feature needs prompt cache / finished-IO reap, skip the
            # extra work (reload path still syncs above).
            if self.needs_sample_phase_hooks():
                self._cache_prompt_token_ids_from_scheduler_output(scheduler_output)
                # Idle / early-return finishes never reach check_after_sample; reap
                # finished reqs whose sample-wave FIFO is already drained.
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

        Content comes from DFX cumulative IO accumulated while
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
        """True when sample-phase DFX hooks must run (else pure ``sample_fn``).

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
        def _run() -> None:
            if not self.should_check_after_spec():
                return
            for alert in self.detectors.check_after_spec(sampled_tokens, accepted_token_nums):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

        self._soft_fail("check_after_spec", _run)

    def record_sample_waves(self, req_ids: list[str] | None) -> None:
        self.wave_tracker.record_sample_waves(req_ids)


    # ---- single sink for the 7 post-pre-sample DFX hooks ---------------

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
        """Single sink for the 7 post-pre-sample DFX hooks (Option α refactor).

        Replaces 7 inline ``self.runtime_guard.*`` calls scattered across
        ``NPUModelRunner.sample_tokens`` with one orchestration call so
        hook ordering is owned by ``RuntimeGuardProcessor`` rather than the runner.

        Hook 1 (``check_before_sample``) stays explicit in the runner because
        it must fire BEFORE ``apply_grammar_bitmask`` (a source-level contract
        enforced by ``test_v1_sample_tokens_checks_before_grammar_bitmask``).

        Hook sequence (``S1`` golden path):
            2. ``ensure_logprobs_for_detection`` (moved out of ``_sample``)
            -> ``sample_fn()`` returns :class:`SamplePhaseResult`
            3. ``note_kv_block_writes``
            4. ``mark_finished``
            -> ``async_state_update_fn`` (only if ``need_accepted_tokens``)
            5. ``check_after_spec`` (spec only; ``accepted_token_nums_fn`` for branch)
            -> ``routed_experts_fn`` (async path: BEFORE wave stamp; sync: AFTER check_after_sample)
            6. ``record_sample_waves``
            7. ``check_after_sample`` (sync path only; async via AscendAsync* ``get_output``)

        Runner still owns post-sample dump finalization on v1 (msprobe debugger
        lifecycle). v2 KV capture uses native ``dump_kv`` actions only.
        """
        # Idle fast-path: detectors / print_output / block-meta all off →
        # skip soft-fail wrappers and observational hooks entirely.
        if not self.needs_sample_phase_hooks():
            result = sample_fn()
            if need_accepted_tokens and async_state_update_fn is not None:
                async_state_update_fn(result)
            routed_experts_result = None
            if routed_experts_fn is not None:
                routed_experts_result = routed_experts_fn(result)
            return result, routed_experts_result

        # Hook 2: ensure_logprobs_for_detection (moved out of _sample)
        self._soft_fail("ensure_logprobs_for_detection", self.ensure_logprobs_for_detection)
        # Runner's sample work (sample + draft + bookkeeping + output + profiling + eplb)
        result = sample_fn()
        # Hook 3: note_kv_block_writes
        self._soft_fail(
            "note_kv_block_writes",
            lambda: self.note_kv_block_writes(
                result.scheduler_output,
                input_batch=result.input_batch,
            ),
        )
        # Hook 5: mark_finished
        self._soft_fail("mark_finished", lambda: self.mark_finished(result.finished_req_ids))
        # Async state update callback (between hook 5 and hook 6)
        if need_accepted_tokens and async_state_update_fn is not None:
            async_state_update_fn(result)
        # Hook 6: check_after_spec (spec only)
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
        # Async path: routed_experts computed BEFORE hook 7
        routed_experts_result = None
        if use_async and routed_experts_fn is not None:
            routed_experts_result = routed_experts_fn(result)
        # Hook 7: record_sample_waves (always)
        self._soft_fail(
            "record_sample_waves",
            lambda: self.record_sample_waves(result.req_ids_output_copy),
        )
        # Hook 8: check_after_sample (sync path only)
        if not use_async:
            self._soft_fail(
                "check_after_sample",
                lambda: self.check_after_sample(
                    sampled_token_ids=result.valid_sampled_token_ids,
                    logprobs_lists=result.logprobs_lists,
                    req_ids=result.req_ids_output_copy,
                ),
            )
        # Sync path: routed_experts computed AFTER hook 7
        if not use_async and routed_experts_fn is not None:
            routed_experts_result = routed_experts_fn(result)
        return result, routed_experts_result

    def check_before_sample(
        self,
        *,
        scheduler_output: Any,
        logits: Any,
        positions: Any = None,
        total_scheduled_tokens: int = 0,
        logits_indices: Any = None,
        input_batch: Any = None,
    ) -> None:
        """Pre-sample hook: logits finite + position alignment."""
        self._last_input_batch = input_batch

        def _run() -> None:
            for alert in self.detectors.check_before_sample(
                scheduler_output=scheduler_output,
                logits=logits,
                positions=positions,
                total_scheduled_tokens=total_scheduled_tokens,
                logits_indices=logits_indices,
                input_batch=input_batch,
            ):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

        self._soft_fail("check_before_sample", _run)

    def check_after_sample(
        self,
        sampled_token_ids: Any,
        logprobs_lists: Any,
        req_ids: list[str] | None = None,
    ) -> None:
        """Sample step hook: run all post-sample detectors (token / substring / …).

        Detection gating (rank / dump / detector-on) lives in ``DetectorManager``.
        Arm wave prefers main-thread stamps from :meth:`record_sample_waves`.
        """

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
            for alert in self.detectors.check_after_sample(
                sampled_token_ids=sampled_token_ids,
                logprobs_lists=logprobs_lists,
                req_ids=req_ids,
            ):
                arm_wave = wave_by_req.get(alert.req_id) if alert.req_id else None
                self._handle_alert(
                    alert,
                    detector=self.detectors.get(alert.incident_type),
                    arm_wave=arm_wave,
                )
            # Last get_output / sync sample consumed stamps → reap finished reqs.
            self._reap_finished_requests()

        self._soft_fail("check_after_sample", _run)


    def ensure_logprobs_for_detection(self) -> None:
        """Bump per-request top-k logprobs when token-logprob detect needs them.

        Clients need not set ``logprobs`` on the request when
        ``detector.token_logprob.enabled`` is on. Independent of ``dump.enabled``.
        Safe no-op when the check is disabled or the batch has no sampling state.
        """
        topk = self.detectors.token_logprob_topk_if_enabled()
        if topk is None:
            return
        runner = self.runner
        # v1 InputBatch: dict[str, int] num_logprobs + SamplingMetadata
        input_batch = getattr(runner, "input_batch", None)
        num_logprobs = getattr(input_batch, "num_logprobs", None) if input_batch is not None else None
        if isinstance(num_logprobs, dict):
            req_ids = getattr(input_batch, "req_ids", None) or []
            changed = False
            for req_id in req_ids:
                cur = num_logprobs.get(req_id)
                # None / missing: no logprobs. Keep full-vocab requests as-is.
                if cur is None or (0 <= cur < topk):
                    num_logprobs[req_id] = topk
                    changed = True
            if changed and hasattr(input_batch, "_make_sampling_metadata"):
                input_batch.sampling_metadata = input_batch._make_sampling_metadata()
                logger.info_once(
                    "[Anomaly token_logprob] forcing request top-k logprobs=%d for detection",
                    topk,
                )
            return

        # v2 SamplingStates: num_logprobs[req_state_idx], -1 == none
        sampler = getattr(runner, "sampler", None)
        states = getattr(sampler, "sampling_states", None) if sampler is not None else None
        if states is None or input_batch is None:
            return
        idx_mapping_np = getattr(input_batch, "idx_mapping_np", None)
        num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
        if idx_mapping_np is None or num_reqs <= 0:
            return
        arr = states.num_logprobs
        changed = False
        for batch_i in range(num_reqs):
            req_state_idx = int(idx_mapping_np[batch_i])
            cur = int(arr[req_state_idx])
            # v2: -1 means no logprobs; positive N means top-N.
            if cur < 0 or cur < topk:
                arr[req_state_idx] = topk
                changed = True
        if changed:
            logger.info_once(
                "[Anomaly token_logprob] forcing request top-k logprobs=%d for detection (v2)",
                topk,
            )

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
            incident_type=alert.incident_type,
        )
        if self.runtime_config.log_print_sampling_meta():
            try:
                self.save_sample_param(alert.req_id)
            except Exception as exc:
                logger.warning("[runtime_guard] save_sample_param failed req_id=%s: %s", alert.req_id, exc)
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
        # consume (#5) and for manual reports/dumps. (Auto detection is
        # placement-spread across ranks; manual is an ops action, not a
        # detector.)
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
        # Prefer detector.manual_trigger.on_trigger when set; else default pair.
        det_cfg = self.runtime_config.detector_section("manual_trigger") or {}
        raw = det_cfg.get("on_trigger")
        override = None
        if raw is not None:
            override = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
        else:
            override = ["report", "dump_kv"]
        self.action_executor.handle(
            incident,
            detail=detail,
            tokenizer=self._get_report_tokenizer(),
            action_override=override,
            batch_rows=batch_rows,
            write_report=write_report,
        )

    def note_kv_block_writes(
        self,
        scheduler_output: Any | None = None,
        *,
        input_batch: Any = None,
    ) -> None:
        """Record per-block last-write wave/writer after a real forward wrote KV.

        No-op when both ``report.block_last_write_wave`` and
        ``report.block_last_writer`` are false, ``report.slot_last_write`` is
        false, and ``detector.block_kv.enabled`` is false. Uses scheduled token
        counts to mark only the blocks / slots touched this step.
        """
        need_meta = self.runtime_config.report_block_meta_enabled()
        need_slots = self.runtime_config.report_slot_last_write()
        need_slot_det = self.detectors.slot_consistency_enabled()
        need_block_det = bool(self.runtime_config.detector_get("block_kv", "enabled", False))
        if not need_meta and not need_slots and not need_slot_det and not need_block_det:
            return
        runner = self.runner
        so = scheduler_output if scheduler_output is not None else getattr(self, "_scheduler_output_for_step", None)
        if runner is None or so is None:
            return
        num_scheduled = getattr(so, "num_scheduled_tokens", None)
        if not isinstance(num_scheduled, dict) or not num_scheduled:
            return
        wave_tracker = getattr(self, "wave_tracker", None)
        wave = 0
        if wave_tracker is not None:
            try:
                wave = int(wave_tracker.current_wave())
            except (TypeError, ValueError):
                wave = 0
        block_size = int(getattr(runner, "block_size", 0) or 0)
        if block_size <= 0:
            cache_cfg = getattr(getattr(runner, "vllm_config", None), "cache_config", None)
            block_size = int(getattr(cache_cfg, "block_size", 0) or 0)
        if block_size <= 0:
            block_size = 16
        tracker = KvBlockMetaTracker.get()
        if input_batch is None:
            input_batch = getattr(runner, "input_batch", None)
        req_id_to_index = getattr(input_batch, "req_id_to_index", None) if input_batch else None
        req_ids = list(getattr(input_batch, "req_ids", None) or [])
        # S12 fix: build index once → O(1) lookup instead of O(n) .index() per req.
        req_id_to_idx_local = {rid: i for i, rid in enumerate(req_ids) if rid}
        for req_id, n_sched in num_scheduled.items():
            if not req_id:
                continue
            try:
                scheduled = int(n_sched)
            except (TypeError, ValueError):
                continue
            if scheduled <= 0:
                continue
            req_idx = None
            if isinstance(req_id_to_index, dict) and req_id in req_id_to_index:
                req_idx = int(req_id_to_index[req_id])
            elif req_id in req_id_to_idx_local:
                req_idx = req_id_to_idx_local[req_id]
            all_ids = block_ids_for_request(
                runner,
                str(req_id),
                req_idx,
                input_batch=input_batch,
            )
            if not all_ids:
                continue
            # Wave-before count: at note_kv_block_writes time (after forward,
            # before next scheduler update) this is still pre-step computed.
            computed_before = num_computed_before(
                runner,
                str(req_id),
                req_idx,
                scheduled,
                input_batch,
            )
            if computed_before is None:
                logger.debug(
                    "[Anomaly block_kv] skip req_id=%s: num_computed_before unknown (kv_cache_group=0 only)",
                    req_id,
                )
                continue
            touched = touched_block_ids(
                all_ids,
                block_size=block_size,
                num_computed_before=computed_before,
                num_scheduled=scheduled,
            )
            if not touched:
                continue
            for alert in self.detectors.check_kv_block_writes(str(req_id), touched, wave):
                self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))
            tracker.record_writes(str(req_id), touched, wave)
            # Slot record + consistency share one snapshot (prompt + cumulative
            # output at note time = every token whose KV exists for this req).
            seq: list[int] | None = None
            if need_slots or need_slot_det:
                snap = RequestIoSnapshotManager.get().snapshot(
                    runner,
                    str(req_id),
                    req_idx,
                    include_token_ids=True,
                )
                seq = list(snap.prompt_token_ids or []) + list(snap.output_token_ids or [])
            if need_slots:
                self._record_slot_writes(
                    str(req_id),
                    all_ids,
                    block_size=block_size,
                    computed_before=computed_before,
                    scheduled=scheduled,
                    seq=seq or [],
                )
            if need_slot_det:
                for alert in self.detectors.check_slot_consistency(
                    req_id=str(req_id),
                    req_idx=req_idx,
                    seq=seq or [],
                    block_ids=all_ids,
                    block_size=block_size,
                    computed_before=computed_before,
                    scheduled=scheduled,
                ):
                    self._handle_alert(alert, detector=self.detectors.get(alert.incident_type))

    def _record_slot_writes(
        self,
        req_id: str,
        all_ids: list[int],
        *,
        block_size: int,
        computed_before: int,
        scheduled: int,
        seq: list[int],
    ) -> None:
        """Stamp per-slot last write with the token whose KV lands at that slot.

        This step writes KV for sequence positions ``[computed_before,
        computed_before+scheduled)``; those tokens are inputs of this forward,
        i.e. all present in ``seq`` (prompt + cumulative output at note time;
        this step's samples are appended later by ``check_after_sample``).
        """
        entries: list[tuple[int, int | None]] = []
        for pos in range(max(0, computed_before), max(0, computed_before) + scheduled):
            bi = pos // block_size
            if bi >= len(all_ids):
                break
            slot = int(all_ids[bi]) * block_size + (pos % block_size)
            token = seq[pos] if pos < len(seq) else None
            entries.append((slot, token))
        if entries:
            KvBlockMetaTracker.get().record_slot_writes(req_id, entries)

    def _enrich_detail_with_block_meta(
        self,
        detail: dict[str, Any],
        req_id: str,
        req_idx: int | None = None,
        *,
        incident_type: str | None = None,
    ) -> dict[str, Any]:
        """Attach ``block_ids`` / ``blocks`` / ``slot_mapping`` per report.* flags.

        Only ``block_kv`` alerts force ``violated_blocks`` (last writer/wave) for
        **errored blocks only** (ids listed in ``detail.violations``), ignoring
        ``report.block_last_writer`` / ``report.block_last_write_wave``.
        Full-request ``blocks[]`` still follows those report flags.
        All other anomaly types / manual reports follow report flags strictly.
        """
        include_ids = self.runtime_config.report_include_block_ids()
        include_slots = self.runtime_config.report_include_slot_mapping()
        include_wave = self.runtime_config.report_block_last_write_wave()
        include_writer = self.runtime_config.report_block_last_writer()
        include_slot_meta = self.runtime_config.report_slot_last_write()
        force_kv_writer = incident_type == "block_kv"
        violated_ids = _block_kv_violated_block_ids(detail) if force_kv_writer else []
        if force_kv_writer and not violated_ids:
            force_kv_writer = False
        if (
            not include_ids
            and not include_slots
            and not include_wave
            and not include_writer
            and not include_slot_meta
            and not force_kv_writer
        ):
            return detail
        out = dict(detail)
        ids = block_ids_for_request(self.runner, req_id, req_idx)
        if include_ids or include_wave or include_writer:
            out["block_ids"] = ids
        if include_wave or include_writer:
            out["blocks"] = KvBlockMetaTracker.get().blocks_detail(
                ids,
                include_wave=include_wave,
                include_writer=include_writer,
                include_creation=True,
            )
        if include_slot_meta:
            # Sparse: only slots that were written while tracking was on.
            out["slots"] = KvBlockMetaTracker.get().slots_detail(
                slots_for_block_ids(ids, block_size=resolve_block_size(self.runner))
            )
        if force_kv_writer:
            # Errored blocks only — not the full request block table.
            out["violated_blocks"] = KvBlockMetaTracker.get().blocks_detail(
                violated_ids,
                include_wave=True,
                include_writer=True,
            )
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
            logger.warning("[DFX] tokenizer load failed error=%s", exc)
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

    def save_sample_param(self, target_req_id: str) -> None:
        """Log sampling metadata for ``target_req_id`` (TP0 + last PP only)."""
        runner = getattr(self, "runner", None)
        if runner is None or not target_req_id:
            return
        try:
            if int(getattr(runner, "tp_rank", 0)) != 0:
                return
            if not get_pp_group().is_last_rank:
                return
        except Exception:
            return

        input_batch = getattr(runner, "input_batch", None)
        if input_batch is None:
            return
        sampling_metadata = getattr(input_batch, "sampling_metadata", None)
        if sampling_metadata is None:
            return
        req_ids = input_batch.req_ids
        for req_idx, req_id in enumerate(req_ids):
            if req_id != target_req_id:
                continue

            temp = sampling_metadata.temperature[req_idx].item() if sampling_metadata.temperature is not None else None
            topk = sampling_metadata.top_k[req_idx].item() if sampling_metadata.top_k is not None else None
            topp = sampling_metadata.top_p[req_idx].item() if sampling_metadata.top_p is not None else None

            freq_pen = sampling_metadata.frequency_penalties[req_idx].item()
            pres_pen = sampling_metadata.presence_penalties[req_idx].item()
            rep_pen = sampling_metadata.repetition_penalties[req_idx].item()

            # S2 fix: bad_words_token_ids may be None (unlike output/spec
            # above which are truthiness-checked) — AttributeError would crash
            # the worker on the detect path.
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

            logger.info(
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
