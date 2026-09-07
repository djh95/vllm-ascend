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

"""Configurable incident actions.

Hot path: :meth:`Action.prepare` sync-snapshots anything that needs the live
runner / device tensors. Cold path: :meth:`Action.commit` writes files / metrics
on the action worker thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.runtime_guard.action.context import ActionContext
from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader, KvDumpSnapshot
from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE
from vllm_ascend.runtime_guard.rank_gate import is_action_leader_rank

logger = init_logger_ascend(__name__)


class Action(ABC):
    name: str
    # True: run entirely on the inference thread (no queue).
    sync_only: bool = False
    # True: commit is heavy I/O (torch.save-scale). On a full action queue
    # heavy commits are dropped, never run inline on the inference thread.
    heavy: bool = False

    def prepare(self, ctx: ActionContext) -> Any | None:
        """Sync phase: capture CPU-side payloads. Return None to skip commit."""
        return None

    def commit(self, prepared: Any) -> None:
        """Async phase: persist / emit metrics from prepared payloads."""
        return None

    def run(self, ctx: ActionContext) -> None:
        """Sync-only actions implement this; default is prepare+commit inline."""
        prepared = self.prepare(ctx)
        if prepared is not None:
            self.commit(prepared)


@dataclass
class ReportPrepared:
    report_writer: Any
    kwargs: dict[str, Any]


class ReportAction(Action):
    name = "report"

    def prepare(self, ctx: ActionContext) -> ReportPrepared | None:
        # Any detection-eligible rank reports: the processor builds each
        # rank's ReportWriter with a rank-tagged dir when not the leader, so
        # per-rank reports never interleave with the leader's (plan A routing).
        dump_count, dump_max_times = ctx.quota.snapshot()
        return ReportPrepared(
            report_writer=ctx.report_writer,
            kwargs={
                "incident_type": ctx.incident.incident_type,
                "req_id": ctx.incident.req_id,
                "detail": dict(ctx.detail),
                "rank_tag": ctx.rank_tag,
                "tokenizer": ctx.tokenizer,
                "dump_attempted": "dump_kv" in ctx.action_overrides.get("_actions", []),
                "dump_armed": True,
                "dump_count": dump_count,
                "dump_max_times": dump_max_times,
                "dump_arm_wave": ctx.incident.wave,
            },
        )

    def commit(self, prepared: ReportPrepared) -> None:
        prepared.report_writer.write(**prepared.kwargs)


@dataclass
class DumpKvPrepared:
    snapshots: list[KvDumpSnapshot]


class DumpKvAction(Action):
    """Dump this request's paged KV blocks (not the full cache), then async-write.

    Default ``scope=request`` uses ``incident.block_ids`` only — saves D2H time
    and disk vs dumping every block.
    """

    name = "dump_kv"
    heavy = True

    def prepare(self, ctx: ActionContext) -> DumpKvPrepared | None:
        # Any detection-eligible rank dumps — KvCacheReader reads this rank's
        # own KV slice (payload carries tp_rank / num_kv_heads). Full TP
        # coverage requires the detector on multiple ranks (ALL scope) or a
        # manual dump per rank. Quota is rank-local on non-leader ranks.
        # B'4: atomic try_consume up front; refunded below if D2H yields nothing.
        if not ctx.quota.try_consume(consume_quota=ctx.incident.consume_quota):
            logger.warning(
                "[runtime_guard dump_kv] quota/cooldown blocked req_id=%s type=%s",
                ctx.incident.req_id,
                ctx.incident.incident_type,
            )
            return None
        cfg = ctx.action_overrides.get("dump_kv") or {}
        if isinstance(cfg, bool):
            cfg = {}
        dump_all_blocks = bool(cfg.get("dump_all_blocks", ctx.runtime_config.dump_get("dump_all_blocks", False)))
        scope = str(cfg.get("scope", "request"))
        # dump.dump_dir (hot-reloadable) or derived <report_dir>/kv_cache.
        base = Path(ctx.runtime_config.dump_root()) / ctx.incident.incident_type

        def _targets():
            if scope == "all_requests" and ctx.batch_rows:
                shared_dir = base / ("shared_all_blocks" if dump_all_blocks else "shared")
                for req_id, _ in ctx.batch_rows:
                    block_ids = (
                        ctx.incident.block_ids
                        if req_id == ctx.incident.req_id
                        else list(ctx.detail.get("block_ids") or [])
                    )
                    yield str(req_id), block_ids or [], (shared_dir if dump_all_blocks else base / str(req_id))
            else:
                req_id = str(ctx.incident.req_id or "unknown")
                yield req_id, list(ctx.incident.block_ids or []), base / req_id

        # B'3: submit each layer snapshot for async save as soon as its D2H
        # completes — a full-cache dump must not stack every layer in host RAM
        # before the first write. Fallback (no submit_async) keeps old callers.
        produced = 0
        fallback: list[KvDumpSnapshot] = []
        for req_id, block_ids, out_dir in _targets():
            for snap in ctx.kv_reader.iter_request_snapshots(
                req_id=req_id,
                block_ids=block_ids,
                out_dir=out_dir,
                dump_all_blocks=dump_all_blocks,
            ):
                produced += 1
                if ctx.submit_async is not None:
                    ctx.submit_async(lambda s=snap: KvCacheReader.write_snapshots([s]))
                else:
                    fallback.append(snap)
        if produced == 0:
            ctx.quota.refund(consume_quota=ctx.incident.consume_quota)
            return None
        if ctx.incident.incident_type == MANUAL_TRIGGER_TYPE:
            # #5: manual_dump=N must drain — arm succeeded on the leader rank,
            # so consume one count now (continuous=true never drains inside).
            # Manual triggers are leader-only (processor gates them), and the
            # config consume must stay single-writer.
            if is_action_leader_rank(ctx.runner) and ctx.runtime_config.consume_manual_trigger():
                logger.info(
                    "[runtime_guard dump_kv] manual_dump consumed, remaining=%d",
                    ctx.runtime_config.manual_trigger_count(),
                )
        if fallback:
            return DumpKvPrepared(snapshots=fallback)
        return None

    def commit(self, prepared: DumpKvPrepared) -> None:
        KvCacheReader.write_snapshots(prepared.snapshots)


class SetLogLevelAction(Action):
    name = "set_log_level"
    sync_only = True

    def run(self, ctx: ActionContext) -> None:
        cfg = ctx.action_overrides.get("set_log_level") or {}
        if not isinstance(cfg, dict):
            return
        level = cfg.get("level")
        modules = cfg.get("modules")
        if level is None and not modules:
            return
        from vllm_ascend.logger import apply_ascend_log_level

        apply_ascend_log_level(
            str(level or ctx.runtime_config.ascend_log_level()),
            module_levels=dict(modules) if isinstance(modules, dict) else None,
        )
        logger.info(
            "[runtime_guard set_log_level] incident=%s level=%s modules=%s",
            ctx.incident.incident_type,
            level,
            modules,
        )


_ACTIONS: dict[str, Action] = {
    ReportAction.name: ReportAction(),
    DumpKvAction.name: DumpKvAction(),
    SetLogLevelAction.name: SetLogLevelAction(),
}


def get_action(name: str) -> Action | None:
    return _ACTIONS.get(name)
