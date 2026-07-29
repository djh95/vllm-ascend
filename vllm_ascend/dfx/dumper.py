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

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group, get_tp_group

from vllm_ascend.dfx.detector.manual_dump import MANUAL_DUMP_REQ_ID
from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)
if TYPE_CHECKING:
    from vllm_ascend.dfx.detector.alert import AnomalyAlert
    from vllm_ascend.dfx.detector.base import AnomalyDetector
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2


class Dumper:
    """Manages msprobe debugger lifecycle and dump arming/activation.

    Detectors live on the model runner; this class reacts to ``AnomalyAlert``
    via :meth:`handle_anomaly_alert`.
    """

    def __init__(
        self,
        runner: NPUModelRunner | NPUModelRunnerV2,
        dynamic_dump_config: Any | None = None,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
    ):
        self.runner = runner
        # Per-step flags for ModelRunnerOutput.debug_log_full (engine prints
        # input/output token ids). Cleared at each start_dump_data.
        self.full_log_requests_this_step: dict[str, bool] = {}
        # Survives the next start_dump_data clear until snapshotted once — needed
        # when async get_output overlaps the following execute_model.
        self._debug_log_full_carry: dict[str, bool] = {}
        self._debugger: Any | None = None

        if dfx_config is not None:
            self.dfx_config = dfx_config
        else:
            ascend = getattr(runner, "ascend_config", None)
            existing = getattr(ascend, "dfx_config", None) if ascend is not None else None
            if existing is not None:
                self.dfx_config = existing
            else:
                legacy = None
                if dynamic_dump_config is not None:
                    # Prefer explicit startup keys only; full ``.config`` carries
                    # defaults and must not overlay JSON.
                    legacy = getattr(dynamic_dump_config, "user_overrides", None)
                    if legacy is None:
                        legacy = getattr(dynamic_dump_config, "config", None)
                path = getattr(ascend, "dfx_config_path", None) if ascend is not None else None
                self.dfx_config = DfxRuntimeConfig(path, legacy_dynamic_dump=legacy)

        self._sync_dump_limits_from_config()

        self._msprobe_dump_total_count = 0
        self._msprobe_dumped_req_ids: set[str] = set()
        self._msprobe_last_dump_ts: float | None = None
        self._msprobe_dump_active = False
        # After enable: require one start→finalize(dump) pair before disable.
        # Avoids async check enabling dump mid-step then finalize turning it
        # off before any dump-capable forward runs.
        self._dump_needs_forward = False
        self._dump_forward_seen = False
        self._debugger_started = False
        # Async cross-rank alignment: check only arms pending; execute_model
        # entry ORs last-PP TP pending (early PP skipped; no PP broadcast).
        self._pending_dump = False
        self._pending_dump_req_id: str | None = None
        # Manual dump_once: activate without consuming max_times / cooldown.
        self._pending_dump_skip_quota = False
        # Real req_id to flag ``debug_log_full`` on the dump-forward step.
        # Engine output_processor prints input/output token ids from that flag.
        self._dump_full_log_req_id: str | None = None

        self._apply_observability_switches()

        logger.info_once(
            "DFX ready config=%s report_dir=%s dump.enabled=%s dump.max_times=%d "
            "ascend_log.level=%s ascend_log.debug=%s metrics=%s/%s trace=%s/%s "
            "spec_check=%s token_logprob_check=%s",
            str(self.dfx_config.config_path),
            str(self.dfx_config.report_dir),
            self.dfx_config.dump_enabled(),
            self._dump_max_times,
            self.dfx_config.ascend_log_level(),
            # info_once is lru_cached; args must be hashable (not list).
            tuple(self.dfx_config.ascend_log_debug_modules()),
            self.dfx_config.metrics_enabled(),
            self.dfx_config.metrics_level(),
            self.dfx_config.trace_enabled(),
            self.dfx_config.trace_level(),
            bool(self.dfx_config.detector_get("enable_spec_acceptance_check", True)),
            bool(self.dfx_config.detector_get("enable_token_logprob_check", False)),
        )

        # Keep debugger lifecycle fully encapsulated in Dumper.
        self._init_debugger(self.runner.compilation_config.cudagraph_mode)

    def _sync_dump_limits_from_config(self) -> None:
        self._dump_cooldown_seconds = self.dfx_config.dump_cooldown_seconds()
        self._dump_max_times = self.dfx_config.dump_max_times()

    def _apply_observability_switches(self) -> None:
        """Apply ``ascend_log`` from live config.

        ``metrics`` / ``trace`` accessors exist on ``DfxRuntimeConfig`` for later
        engine wiring; dump path respects ``dump.enabled`` separately.
        """
        self.dfx_config.apply_ascend_log_level()

    def apply_dfx_config(self) -> None:
        """Pull dump limits / ``ascend_log`` from already-synced ``dfx_config``.

        Runner owns :meth:`~DfxRuntimeConfig.sync_dfx_config`; call this only
        after a successful reload so Dumper never drives config I/O.
        """
        prev_max = self._dump_max_times
        prev_cd = self._dump_cooldown_seconds
        self._sync_dump_limits_from_config()
        self._apply_observability_switches()
        logger.info(
            "[DFX dumper] config applied dump.max_times=%d→%d cooldown=%d→%d %s "
            "(short/detect follows detector.*; max_times only gates dump arming)",
            prev_max,
            self._dump_max_times,
            prev_cd,
            self._dump_cooldown_seconds,
            self._dump_rank_tag(),
        )

    def handle_anomaly_alert(
        self,
        alert: AnomalyAlert,
        *,
        detector: AnomalyDetector | None = None,
    ) -> bool:
        """Arm / activate dump from a detector alert (report is runner-owned)."""
        if alert is None or not alert.is_ill or not alert.req_id:
            return False
        ok = self.enable_msprobe_dump_if_needed(
            alert.req_id,
            req_idx=alert.req_idx,
            skip_related_check=alert.skip_related_check,
            consume_quota=alert.consume_quota,
        )
        if not ok:
            return False
        if alert.mark_full_log:
            self.full_log_requests_this_step[alert.req_id] = True
            self._debug_log_full_carry[alert.req_id] = True
        if detector is not None:
            detector.on_alert_armed(alert)
        return True

    def take_debug_log_full(self) -> dict[str, bool]:
        """Snapshot ``debug_log_full`` for ``ModelRunnerOutput`` (consume carry)."""
        out = dict(self.full_log_requests_this_step)
        out.update(self._debug_log_full_carry)
        self._debug_log_full_carry.clear()
        return out

    def anomaly_check_skip_reason(self) -> str | None:
        """None if detectors may run; otherwise a short skip reason for logs.

        Detection / short logs are **not** gated on ``dump.max_times``.
        Quota only blocks arming dump in ``enable_msprobe_dump_if_needed``.
        """
        if not self.dfx_config.dump_enabled():
            return "dump.enabled=false"
        if not (
            self.dfx_config.detector_get("enable_spec_acceptance_check", False)
            or self.dfx_config.detector_get("enable_token_logprob_check", False)
        ):
            return "no detector enabled in live DFX config"
        if not self._should_run_anomaly_check():
            if not get_pp_group().is_last_rank:
                return "not last PP rank"
            if self._use_pending_dump_sync() and int(getattr(self.runner, "tp_rank", 0)) != 0:
                return "async: only output_rank TP0 runs anomaly check"
            return "rank not selected for anomaly check"
        if self._pending_dump:
            return "pending_dump already armed"
        if self._msprobe_dump_active:
            return "msprobe dump already active"
        return None

    def can_run_anomaly_detection(self) -> bool:
        """Whether this rank should invoke detectors this step."""
        return self.anomaly_check_skip_reason() is None

    def _init_debugger(self, cudagraph_mode: CUDAGraphMode):
        dump_cfg = self.runner.ascend_config.dump_config_path
        if dump_cfg is None:
            self._debugger = None
            return None
        if cudagraph_mode == CUDAGraphMode.NONE:
            from msprobe.pytorch import PrecisionDebugger

            self._debugger = PrecisionDebugger(dump_cfg)
            return self._debugger

        try:
            from msprobe.pytorch import AclGraphDumper
        except Exception as exc:
            raise RuntimeError(
                "Failed to import AclGraphDumper from msprobe. "
                "Please install/rebuild msprobe with aclgraph_dump enabled."
            ) from exc
        self._debugger = AclGraphDumper(dump_cfg)
        return self._debugger

    def _dump_rank_tag(self) -> str:
        tp = getattr(self.runner, "tp_rank", "?")
        dp = getattr(self.runner, "dp_rank", "?")
        try:
            pp = get_pp_group().rank_in_group
        except Exception:
            pp = "?"
        return f"dp={dp} tp={tp} pp={pp}"

    def _dump_state_tag(self) -> str:
        return (
            f"active={self._msprobe_dump_active} "
            f"needs_fwd={self._dump_needs_forward} "
            f"fwd_seen={self._dump_forward_seen} "
            f"dbg_started={self._debugger_started} "
            f"pending={self._pending_dump} "
        )

    def start_dump_data(self) -> None:
        # Always clear per-step flags, even when debugger is inactive.
        self.full_log_requests_this_step.clear()
        if self._debugger is None:
            return

        # Mark dump-forward even when debugger was pre-started (e.g. load_model
        # graph hook): early-return must not skip _dump_forward_seen or disable
        # stays deferred forever.
        will_mark_forward_seen = bool(self._msprobe_dump_active and self._dump_needs_forward)
        if not self._debugger_started:
            self._debugger.start(self.runner.model)
            self._debugger_started = True
        if will_mark_forward_seen:
            self._dump_forward_seen = True
            # Lightweight only: re-arm debug_log_full after clear so this step's
            # ModelRunnerOutput carries the flag. Engine prints token ids later.
            # Never log full token lists here — that stalls TP0 before forward
            # collectives and hangs the request.
            self._arm_debug_log_full_for_dump_step()
            logger.info(
                "[Anomaly msprobe] start dump-forward %s %s debug_log_full=%s",
                self._dump_rank_tag(),
                self._dump_state_tag(),
                sorted(self.full_log_requests_this_step),
            )

    def _arm_debug_log_full_for_dump_step(self) -> None:
        """Flag dump target req for engine-side input/output token id logging."""
        req_id = self._dump_full_log_req_id
        if not req_id:
            return
        self.full_log_requests_this_step[req_id] = True
        self._debug_log_full_carry[req_id] = True

    def finalize_dump_data(self, *, dump: bool = True) -> None:
        if self._debugger is None or not self._debugger_started:
            return
        dumping = bool(self._msprobe_dump_active)
        if hasattr(self._debugger, "stop"):
            self._debugger.stop()
            self._debugger_started = False

        if dump:
            self._debugger.step()
        else:
            self._debugger.step(dump=False)
        # capture/dummy (dump=False): must not consume the pending dump-forward window.
        if not dump:
            if self._dump_needs_forward:
                self._dump_forward_seen = False
            return
        self.disable_msprobe_dump_if_needed()
        if dumping:
            logger.debug(
                "[Anomaly msprobe] finalize after dump-forward %s %s",
                self._dump_rank_tag(),
                self._dump_state_tag(),
            )

    @contextmanager
    def lock_msprobe_config(self, config_path: Path):
        lock_path = Path(f"{config_path}.lock")
        os.makedirs(lock_path.parent, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def disable_msprobe_dump_if_needed(self) -> None:
        if not self._msprobe_dump_active:
            return
        if self._debugger is None:
            return
        # Async check may enable dump after this step's start (or even after
        # forward). Keep dump_enable until a later start→finalize pair runs.
        if self._dump_needs_forward and not self._dump_forward_seen:
            logger.debug(
                "[Anomaly msprobe] disable deferred (needs forward) %s %s",
                self._dump_rank_tag(),
                self._dump_state_tag(),
            )
            return
        if not self.set_msprobe_dump_state(False):
            return
        self._msprobe_dump_active = False
        self._dump_needs_forward = False
        self._dump_forward_seen = False
        self._dump_full_log_req_id = None
        logger.info(
            "[Anomaly msprobe] disable succeeded %s",
            self._dump_rank_tag(),
        )

    def set_msprobe_dump_state(self, dump_state: bool) -> bool:
        """Write dump_enable and reload debugger config under the same lock.

        Reload must stay next to the write: any work between them (logging,
        sample-param dump, another thread's start/finalize, or another TP
        flipping the shared JSON) can make start() see a stale in-memory flag.
        """
        dump_cfg = self.runner.ascend_config.dump_config_path
        if not dump_cfg:
            logger.error("[Anomaly msprobe] set msprobe dump state failed, because dump_config_path is empty")
            return False

        config_path = Path(dump_cfg)
        if not config_path.exists():
            logger.error(
                "[Anomaly msprobe] set msprobe dump state failed, because config file not found. path=%s",
                str(config_path),
            )
            return False

        try:
            with self.lock_msprobe_config(config_path):
                with config_path.open("r", encoding="utf-8") as f:
                    config_obj = json.load(f)

                if not isinstance(config_obj, dict):
                    logger.error(
                        "[Anomaly msprobe] set msprobe dump state failed, because json root is not object. type=%s",
                        type(config_obj).__name__,
                    )
                    return False

                ori_value = config_obj.get("dump_enable")
                if ori_value != dump_state:
                    config_obj["dump_enable"] = dump_state
                    with config_path.open("w", encoding="utf-8") as f:
                        json.dump(config_obj, f, ensure_ascii=False, indent=2)
                        f.write("\n")
                # Reload while still holding the lock so this process picks up
                # the value we just wrote before another rank can change it.
                if self._debugger is not None:
                    self._debugger._maybe_reload_config(force=True)
            return True
        except Exception as e:
            logger.error(
                "[Anomaly msprobe] set msprobe dump state failed, path=%s error=%s",
                str(config_path),
                e,
            )
            return False

    def is_related_local_request(self, req_id: str, req_idx: int | None = None) -> bool:
        input_batch = getattr(self.runner, "input_batch", None)
        req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None

        # v2 (and batch-local) path: req_idx is the position in input_batch.req_ids.
        if req_ids is not None and req_idx is not None:
            if req_idx < 0 or req_idx >= len(req_ids) or req_ids[req_idx] != req_id:
                return False
            requests = getattr(self.runner, "requests", None)
            if requests is not None and req_id not in requests:
                return False
            req_states = getattr(self.runner, "req_states", None)
            req_id_to_index = getattr(req_states, "req_id_to_index", None)
            if req_id_to_index is not None and req_id not in req_id_to_index:
                return False
            discard_request_mask = getattr(self.runner, "discard_request_mask", None)
            if discard_request_mask is not None and hasattr(discard_request_mask, "np"):
                if req_idx < len(discard_request_mask.np) and discard_request_mask.np[req_idx]:
                    return False
            return True

        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if req_id_to_index is None:
            req_states = getattr(self.runner, "req_states", None)
            req_id_to_index = getattr(req_states, "req_id_to_index", None)
        if req_id_to_index is None:
            return False

        mapped_idx = req_id_to_index.get(req_id)
        if mapped_idx is None:
            return False

        if req_idx is not None and mapped_idx != req_idx:
            if self.runner.tp_rank == 0:
                logger.warning(
                    "[Anomaly msprobe] req_id=%s skip dump: req_idx mismatch input=%d mapped=%d",
                    req_id,
                    req_idx,
                    mapped_idx,
                )
            return False

        num_reqs = getattr(input_batch, "num_reqs", None)
        if num_reqs is None:
            req_states = getattr(self.runner, "req_states", None)
            num_reqs_np = getattr(req_states, "num_reqs_np", None)
            if num_reqs_np is not None:
                num_reqs = int(num_reqs_np[0])
        if num_reqs is None:
            return False

        if mapped_idx < 0 or mapped_idx >= num_reqs:
            return False

        if req_ids is not None and mapped_idx < len(req_ids) and req_ids[mapped_idx] != req_id:
            return False

        discard_request_mask = getattr(self.runner, "discard_request_mask", None)
        if discard_request_mask is not None and hasattr(discard_request_mask, "np"):
            if mapped_idx < len(discard_request_mask.np) and discard_request_mask.np[mapped_idx]:
                return False

        requests = getattr(self.runner, "requests", None)
        if requests is not None:
            return req_id in requests
        return True

    def _anomaly_dump_feature_enabled(self) -> bool:
        """Whether auto dump arming / OR-sync is allowed (needs quota)."""
        if not self.dfx_config.dump_enabled():
            return False
        max_times = self.dfx_config.dump_max_times()
        if max_times <= 0 and self._dump_max_times <= 0:
            return False
        return bool(
            self.dfx_config.detector_get("enable_spec_acceptance_check", False)
            or self.dfx_config.detector_get("enable_token_logprob_check", False)
        )

    def _dump_max_times_live(self) -> int:
        """Prefer live JSON ``dump.max_times``; fall back to last applied cache."""
        live = self.dfx_config.dump_max_times()
        if live > 0:
            return live
        return int(self._dump_max_times)

    def _use_pending_dump_sync(self) -> bool:
        """Defer dump_enable until last-PP TP OR at execute_model entry.

        Required for async (only TP0 checks). Also used whenever TP>1 so sync
        path never activates debugger mid-sample on a subset of ranks (that
        desyncs the next forward collective and hangs).
        """
        if bool(getattr(self.runner, "use_async_scheduling", False)):
            return True
        try:
            return get_tp_group().world_size > 1
        except Exception:
            return False

    def _should_run_anomaly_check(self) -> bool:
        """Whether this rank should evaluate anomaly detectors.

        Last PP only. When pending-OR is used (async, or sync with TP>1), only
        TP0 checks and arms ``pending_dump``; peers join OR at execute entry.
        Sync TP=1: the single rank checks and may activate immediately.
        """
        if not get_pp_group().is_last_rank:
            return False
        if self._use_pending_dump_sync():
            return int(getattr(self.runner, "tp_rank", 0)) == 0
        return True

    def _clear_pending_dump(self) -> None:
        self._pending_dump = False
        self._pending_dump_req_id = None
        self._pending_dump_skip_quota = False

    def _activate_msprobe_dump(self, req_id: str | None, *, consume_quota: bool = True) -> bool:
        """Turn on dump_enable + reload on this rank (called after sync decide).

        ``consume_quota=False``: manual ``dump_once`` — do not bump count / cooldown.
        """
        if self._debugger is None:
            logger.error(
                "[Anomaly msprobe] skip dump activate req_id=%s: debugger is None",
                req_id,
            )
            return False
        if self._msprobe_dump_active:
            return True
        if not self.set_msprobe_dump_state(True):
            logger.error(
                "[Anomaly msprobe] set dump state failed req_id=%s",
                req_id,
            )
            return False
        self._msprobe_dump_active = True
        self._dump_needs_forward = True
        self._dump_forward_seen = False
        if req_id and req_id != MANUAL_DUMP_REQ_ID:
            self._dump_full_log_req_id = req_id
        else:
            # dump_once / unknown: no engine debug_log_full target.
            self._dump_full_log_req_id = None
        if consume_quota:
            if req_id is not None:
                self._msprobe_dumped_req_ids.add(req_id)
            self._msprobe_dump_total_count += 1
            self._msprobe_last_dump_ts = time.time()

        logger.info(
            "[Anomaly msprobe] activate ok req_id=%s count=%d/%d consume_quota=%s %s",
            req_id,
            self._msprobe_dump_total_count,
            self._dump_max_times,
            consume_quota,
            self._dump_rank_tag(),
        )
        return True

    def sync_dump_pending_or(self, *, allow_arm: bool = True) -> bool:
        """Align dump among last-PP TP ranks (dump OR only; no config sync).

        Call **after** runner ``dfx.refresh_config()`` / ``sync_dfx_config``.
        Config sync is a world collective and must run on every rank; this
        method is last-PP TP only — do not fold config reload into it.

        Only **last PP** dumps (precision compare usually needs the final stage).
        Early PP skip entirely — no PP / world collective here.

        When pending-OR is enabled (async, or sync with TP>1): OR ``pending_dump``
        across TP; if any rank armed, all last-PP TPs activate together.

        ``allow_arm``: False on dummy/capture — last-PP TPs still join the
        all_reduce (avoid deadlock) but do not activate or clear pending.
        """
        tag = self._dump_rank_tag()
        if not self._use_pending_dump_sync():
            if not self._anomaly_dump_feature_enabled() and not self._pending_dump:
                return False
            return self._msprobe_dump_active

        pp_group = get_pp_group()
        if not pp_group.is_last_rank:
            return False

        tp_group = get_tp_group()
        # Always join OR on last-PP (even if local pending is false /
        # anomaly detectors are off). A peer with dump_once pending must not
        # hang alone in all_reduce.
        local = 1 if self._pending_dump else 0
        logger.debug(
            "[DFX sync] enter stage=dump_pending_or local=%d allow_arm=%s tp_world=%s %s",
            local,
            allow_arm,
            tp_group.world_size,
            tag,
        )
        # CPU int32 SUM: OR = sum > 0. tp_group.world_size is TP size only
        # (e.g. DP2/PP2/TP2 → 2, not 8).
        pending_t = torch.tensor([local], dtype=torch.int32)
        if tp_group.world_size > 1:
            torch.distributed.all_reduce(pending_t, group=tp_group.cpu_group)
        pending_sum = int(pending_t.item())
        logger.debug(
            "[DFX sync] leave stage=dump_pending_or sum=%d %s",
            pending_sum,
            tag,
        )
        if pending_sum <= 0:
            return False

        if not allow_arm:
            return False

        req_id = self._pending_dump_req_id
        consume_quota = not self._pending_dump_skip_quota
        logger.debug(
            "[DFX sync] enter stage=dump_activate req_id=%s consume_quota=%s %s",
            req_id,
            consume_quota,
            tag,
        )
        if not self._activate_msprobe_dump(req_id, consume_quota=consume_quota):
            if self._pending_dump:
                logger.error("[Anomaly msprobe] dump activate failed after OR; keep pending")
            logger.debug("[DFX sync] leave stage=dump_activate ok=False %s", tag)
            return False
        self._clear_pending_dump()
        logger.debug("[DFX sync] leave stage=dump_activate ok=True %s", tag)
        return True

    def enable_msprobe_dump_if_needed(
        self,
        req_id: str,
        req_idx: int | None = None,
        *,
        skip_related_check: bool = False,
        consume_quota: bool = True,
    ) -> bool:
        if self._debugger is None:
            logger.error(
                "[Anomaly msprobe] skip dump req_id=%s: debugger is None",
                req_id,
            )
            return False
        if not self.dfx_config.dump_enabled():
            logger.warning(
                "[Anomaly msprobe] skip dump req_id=%s: dump.enabled=false",
                req_id,
            )
            return False
        if not get_pp_group().is_last_rank:
            return False
        if not skip_related_check and not self.is_related_local_request(req_id, req_idx):
            return False
        if self._pending_dump or self._msprobe_dump_active:
            # Already armed / dumping this cycle.
            return True
        if consume_quota:
            if req_id in self._msprobe_dumped_req_ids:
                return False
            max_times = self._dump_max_times_live()
            if max_times <= 0 or self._msprobe_dump_total_count >= max_times:
                logger.info_once(
                    "[Anomaly msprobe] skip dump req_id=%s: dump.max_times=%d count=%d "
                    "(detectors still run; only dump is quota-gated)",
                    req_id,
                    max_times,
                    self._msprobe_dump_total_count,
                )
                return False
            now_ts = time.time()
            elapsed = None if self._msprobe_last_dump_ts is None else now_ts - self._msprobe_last_dump_ts
            if elapsed is not None and elapsed < self._dump_cooldown_seconds:
                return False

        # Async: only arm pending; dump_enable + reload, dumped_req_ids, and
        # cooldown timestamp happen in _activate_msprobe_dump after OR sync so
        # a failed activate does not permanently blacklist the request.
        if self._use_pending_dump_sync():
            self._pending_dump = True
            self._pending_dump_req_id = req_id if consume_quota else None
            self._pending_dump_skip_quota = not consume_quota
            logger.info(
                "[Anomaly msprobe] req_id=%s armed pending_dump (await OR sync). "
                "next_activation_count=%d/%d consume_quota=%s",
                req_id,
                self._msprobe_dump_total_count + (1 if consume_quota else 0),
                self._dump_max_times_live(),
                consume_quota,
            )
            return True

        return self._activate_msprobe_dump(req_id if consume_quota else None, consume_quota=consume_quota)
