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

"""Resolve and run configured incident actions."""

from __future__ import annotations

from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_guard.action.actions import Action, get_action
from vllm_ascend.runtime_guard.action.context import ActionContext
from vllm_ascend.runtime_guard.action.queue import ActionQueue
from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader
from vllm_ascend.runtime_guard.quota import DumpQuota
from vllm_ascend.runtime_guard.rank_gate import dump_rank_tag, should_run_anomaly_check_on_rank
from vllm_ascend.runtime_guard.report import ReportWriter
from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)

_DEFAULT_ACTIONS = ["report"]


class ActionExecutor:
    def __init__(
        self,
        runner: Any,
        *,
        runtime_config: RuntimeConfig,
        report_writer: ReportWriter,
        quota: DumpQuota,
        action_queue: ActionQueue | None = None,
    ) -> None:
        self._runner = runner
        self._runtime_config = runtime_config
        self._report_writer = report_writer
        self._quota = quota
        self._kv_reader = KvCacheReader(runner)
        self._queue = action_queue if action_queue is not None else ActionQueue()

    @property
    def action_queue(self) -> ActionQueue:
        return self._queue

    def rebind_runner(self, runner: Any, *, runtime_config: RuntimeConfig | None = None) -> None:
        self._runner = runner
        if runtime_config is not None:
            self._runtime_config = runtime_config
        self._kv_reader = KvCacheReader(runner)

    def start(self) -> None:
        self._queue.start()

    def stop(self) -> None:
        self._queue.stop()

    def can_run_detection(self) -> bool:
        return should_run_anomaly_check_on_rank(self._runner)

    def anomaly_check_skip_reason(self) -> str | None:
        from vllm_ascend.runtime_guard.rank_gate import anomaly_check_rank_skip_reason

        return anomaly_check_rank_skip_reason(self._runner)

    def _submit_heavy(self, job: Any) -> None:
        self._queue.submit(job, heavy=True)

    def resolve_actions(
        self,
        incident_type: str,
        *,
        override: list[str] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        if override is not None:
            names = list(override)
            overrides = self._runtime_config.detector_section(incident_type) or {}
            return names, dict(overrides) if isinstance(overrides, dict) else {}

        defaults = self._runtime_config.actions_default_on_trigger()
        det = self._runtime_config.detector_section(incident_type) or {}
        raw = det.get("on_trigger")
        if raw is None:
            names = list(defaults or _DEFAULT_ACTIONS)
        elif isinstance(raw, str):
            names = [raw]
        elif isinstance(raw, list):
            names = [str(x) for x in raw]
        else:
            names = list(defaults or _DEFAULT_ACTIONS)
        return names, dict(det)

    def handle(
        self,
        incident: Incident,
        *,
        detail: dict[str, Any],
        tokenizer: Any | None = None,
        action_override: list[str] | None = None,
        batch_rows: list[tuple[str, int | None]] | None = None,
        write_report: bool = True,
    ) -> None:
        if not self.can_run_detection():
            return
        names, det_cfg = self.resolve_actions(incident.incident_type, override=action_override)
        if not write_report:
            names = [n for n in names if n != "report"]
        overrides = dict(det_cfg)
        overrides["_actions"] = names
        ctx = ActionContext(
            incident=incident,
            runner=self._runner,
            runtime_config=self._runtime_config,
            report_writer=self._report_writer,
            kv_reader=self._kv_reader,
            quota=self._quota,
            rank_tag=dump_rank_tag(self._runner),
            tokenizer=tokenizer,
            detail=detail,
            action_overrides=overrides,
            batch_rows=batch_rows,
            submit_async=self._submit_heavy,
        )

        # Pipeline (inference thread + action worker):
        # 1) sync_only actions (e.g. set_log_level)
        # 2) sync prepare report (CPU detail already snapshotted) → async write/metrics
        # 3) sync KV D2H for this request's blocks only → async torch.save
        # Report is enqueued before D2H so file/metrics overlap with device copy.
        ordered: list[str] = []
        for name in names:
            if name not in ordered:
                ordered.append(name)
        # Stable preference: report before dump_kv when both are present.
        if "report" in ordered and "dump_kv" in ordered:
            ordered = [n for n in ordered if n not in ("report", "dump_kv")]
            # Keep relative order of other actions; inject report then dump_kv up front
            # after any leading sync_only names already listed.
            head = [n for n in ordered if (get_action(n) and get_action(n).sync_only)]
            tail = [n for n in ordered if not (get_action(n) and get_action(n).sync_only)]
            ordered = head + ["report", "dump_kv"] + [n for n in tail if n not in head]

        for name in ordered:
            action = get_action(name)
            if action is None:
                logger.warning("[runtime_guard action] unknown action=%s incident=%s", name, incident.incident_type)
                continue
            try:
                if action.sync_only:
                    action.run(ctx)
                    continue
                prepared = action.prepare(ctx)
                if prepared is None:
                    continue

                def _commit(action: Action = action, prepared: Any = prepared) -> None:
                    try:
                        action.commit(prepared)
                    except Exception:
                        logger.exception(
                            "[runtime_guard action] %s commit failed incident=%s req_id=%s",
                            action.name,
                            incident.incident_type,
                            incident.req_id,
                        )

                # Submit immediately so report IO can overlap with subsequent KV D2H.
                self._queue.submit(_commit, heavy=action.heavy)
            except Exception as exc:
                logger.exception(
                    "[runtime_guard action] %s prepare failed incident=%s req_id=%s: %s",
                    name,
                    incident.incident_type,
                    incident.req_id,
                    exc,
                )

    def apply_runtime_config(self) -> None:
        self._quota.sync_from_config()
