#!/usr/bin/env python3
"""Apply per-wave absolute-timestamp probe to tip15 worktree (idempotent).

v1: split merged-bus AR into all_reduce-call vs .item() wait, record gate
backend, and log per-wave wall-clock timestamps for rank alignment.

v2 (misalignment forensics — "rank 是第几次，为什么对不齐"):
  - [RG-BUS-WV2] live per-wave line at drain with idx (bus AR counter),
    sync (sync_for_step counter), tp, head/submit/ar_in/ar_out/fwd_in/fwd_out
  - [RG-BUS-AR-IN]/[RG-BUS-AR-OUT] worker-thread live logs at AR start/end
  - hooks.py wrapper records forward start/end around execute_model
  - MergedBusRequest carries wave_idx/tp_rank for worker-side logs
"""
import sys
from pathlib import Path

ROOT = Path("/data0/test-mrv2-cann91/rg-tip15-8f5e3/vllm_ascend/observability")
BUS = ROOT / "runtime_config" / "_task_bus.py"
WORKER = ROOT / "runtime_guard" / "bus_worker.py"
PROC = ROOT / "runtime_guard" / "processor_bus.py"
PROC_CORE = ROOT / "runtime_guard" / "processor.py"
HOOKS = ROOT / "runtime_guard" / "hooks.py"


def sub_once(text: str, old: str, new: str, path: Path) -> tuple[str, bool]:
    if new in text:
        return text, True
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR COUNT {n} (want 1) in {path}: {old[:90]!r}")
    return text.replace(old, new, 1), False


report: list[str] = []

# ---------- _task_bus.py ----------
t = BUS.read_text()
t, done = sub_once(
    t,
    "from __future__ import annotations\n\nfrom collections.abc import Callable, Sequence",
    "from __future__ import annotations\n\nimport time\nfrom collections.abc import Callable, Sequence",
    BUS,
)
report.append(f"_task_bus import time: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    "def sync_due_bits(group: Any, due_locals: Sequence[bool]) -> list[bool]:",
    "def sync_due_bits(\n    group: Any, due_locals: Sequence[bool], *, timings: dict[str, Any] | None = None\n) -> list[bool]:",
    BUS,
)
report.append(f"_task_bus signature: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    """    torch.distributed.all_reduce(
        due_t,
        op=torch.distributed.ReduceOp.MAX,
        group=gate,
    )
    return [float(due_t[i].item()) >= 0.5 for i in range(len(bits))]""",
    """    t_ar0 = time.perf_counter_ns()
    if timings is not None:
        timings["gate"] = device
        timings["ar_enter_wall"] = time.time()
    torch.distributed.all_reduce(
        due_t,
        op=torch.distributed.ReduceOp.MAX,
        group=gate,
    )
    t_ar1 = time.perf_counter_ns()
    out = [float(due_t[i].item()) >= 0.5 for i in range(len(bits))]
    if timings is not None:
        timings["ar_call_ns"] = t_ar1 - t_ar0
        timings["ar_item_ns"] = time.perf_counter_ns() - t_ar1
        timings["ar_exit_wall"] = time.time()
    return out""",
    BUS,
)
report.append(f"_task_bus ar split: {'already' if done else 'applied'}")
BUS.write_text(t)

# ---------- bus_worker.py ----------
t = WORKER.read_text()
t, done = sub_once(
    t,
    "    request: MergedBusRequest\n    enqueued_ns: int = 0",
    "    request: MergedBusRequest\n    enqueued_ns: int = 0\n    enqueued_wall: float = 0.0",
    WORKER,
)
report.append(f"worker _Slot wall: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    "        slot = _Slot(request=request, enqueued_ns=time.perf_counter_ns())",
    "        slot = _Slot(\n            request=request,\n            enqueued_ns=time.perf_counter_ns(),\n            enqueued_wall=time.time(),\n        )",
    WORKER,
)
report.append(f"worker submit wall: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    '                tm["sched"] = t_start - item.enqueued_ns\n                tm["total"] = time.perf_counter_ns() - item.enqueued_ns',
    '                tm["sched"] = t_start - item.enqueued_ns\n                tm["total"] = time.perf_counter_ns() - item.enqueued_ns\n                tm["submit_wall"] = item.enqueued_wall',
    WORKER,
)
report.append(f"worker loop submit_wall: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    """        config_due, dump_due = sync_due_bits(
            req.sync_group,
            [req.config_due_local, req.dump_due_local],
        )""",
    """        config_due, dump_due = sync_due_bits(
            req.sync_group,
            [req.config_due_local, req.dump_due_local],
            timings=timings,
        )""",
    WORKER,
)
report.append(f"worker pass timings: {'already' if done else 'applied'}")

# v2: request carries per-wave idx/rank for worker-side AR start/end logs
t, done = sub_once(
    t,
    "    # Leader-only: build config JSON after AR says config_due.\n    build_config_payload: Callable[[], tuple[Any, bool]] | None = None",
    "    # Leader-only: build config JSON after AR says config_due.\n    build_config_payload: Callable[[], tuple[Any, bool]] | None = None\n    # Probe-only: rank-local wave counters for misalignment forensics.\n    wave_idx: int = -1\n    tp_rank: int = -1",
    WORKER,
)
report.append(f"worker request idx fields: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    """        timings: dict[str, int] = {}
        t_ar = time.perf_counter_ns()
        config_due, dump_due = sync_due_bits(
            req.sync_group,
            [req.config_due_local, req.dump_due_local],
            timings=timings,
        )
        timings["ar"] = time.perf_counter_ns() - t_ar""",
    """        timings: dict[str, int] = {}
        t_ar = time.perf_counter_ns()
        logger.info(
            "[RG-BUS-AR-IN] idx=%d tp=%d wall=%.3f", req.wave_idx, req.tp_rank, time.time()
        )
        config_due, dump_due = sync_due_bits(
            req.sync_group,
            [req.config_due_local, req.dump_due_local],
            timings=timings,
        )
        logger.info(
            "[RG-BUS-AR-OUT] idx=%d tp=%d wall=%.3f enter=%.3f exit=%.3f call_us=%.1f item_us=%.1f",
            req.wave_idx,
            req.tp_rank,
            time.time(),
            float(timings.get("ar_enter_wall", 0.0)),
            float(timings.get("ar_exit_wall", 0.0)),
            float(timings.get("ar_call_ns", 0)) / 1000.0,
            float(timings.get("ar_item_ns", 0)) / 1000.0,
        )
        timings["ar"] = time.perf_counter_ns() - t_ar""",
    WORKER,
)
report.append(f"worker AR start/end logs: {'already' if done else 'applied'}")
WORKER.write_text(t)

# ---------- processor_bus.py ----------
t = PROC.read_text()
t, done = sub_once(
    t,
    '                "bcast_ns": 0,\n            }\n            processor._bus_stats_st = st',
    '                "bcast_ns": 0,\n                "recent": [],\n                "last_head_wall": 0.0,\n            }\n            processor._bus_stats_st = st',
    PROC,
)
report.append(f"proc stats keys: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    """        config_due_local, dump_due_local, dump_jobs, can_dump, is_first = (
            RuntimeGuardBusMixin._prepare_merged_bus_locals(self, sync_group)
        )""",
    """        st_head = RuntimeGuardBusMixin._bus_stats(self)
        st_head["last_head_wall"] = time.time()
        config_due_local, dump_due_local, dump_jobs, can_dump, is_first = (
            RuntimeGuardBusMixin._prepare_merged_bus_locals(self, sync_group)
        )""",
    PROC,
)
report.append(f"proc head wall: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    """            if tm.get("bcast", 0):
                st["bcast_n"] += 1
                st["bcast_ns"] += tm["bcast"]
        changed = RuntimeGuardBusMixin._apply_merged_bus_result(""",
    """            if tm.get("bcast", 0):
                st["bcast_n"] += 1
                st["bcast_ns"] += tm["bcast"]
        st["recent"].append(
            {
                "head": st.get("last_head_wall", 0.0),
                "submit": float(tm.get("submit_wall", 0.0)) if tm else 0.0,
                "ar_in": float(tm.get("ar_enter_wall", 0.0)) if tm else 0.0,
                "ar_out": float(tm.get("ar_exit_wall", 0.0)) if tm else 0.0,
                "ar_call_us": round(tm.get("ar_call_ns", 0) / 1000.0, 1) if tm else 0.0,
                "ar_item_us": round(tm.get("ar_item_ns", 0) / 1000.0, 1) if tm else 0.0,
                "gate": str(tm.get("gate", "?")) if tm else "?",
                "wait_us": round(dt_wait / 1000.0, 1),
                "drain_end": time.time(),
            }
        )
        if len(st["recent"]) > 8:
            st["recent"].pop(0)
        changed = RuntimeGuardBusMixin._apply_merged_bus_result(""",
    PROC,
)
report.append(f"proc drain recent: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    """        for k in st:
            st[k] = 0""",
    """        for r in st.get("recent", []):
            logger.info(
                "[RG-BUS-WAVE] head=%.3f submit=+%.1fms ar_in=+%.1fms ar_out=+%.1fms "
                "ar_call_us=%.1f ar_item_us=%.1f gate=%s wait_us=%.1f drain_end=%.3f",
                r.get("head", 0.0),
                (r.get("submit", 0.0) - r.get("head", 0.0)) * 1000.0,
                (r.get("ar_in", 0.0) - r.get("head", 0.0)) * 1000.0,
                (r.get("ar_out", 0.0) - r.get("head", 0.0)) * 1000.0,
                r.get("ar_call_us", 0.0),
                r.get("ar_item_us", 0.0),
                r.get("gate", "?"),
                r.get("wait_us", 0.0),
                r.get("drain_end", 0.0),
            )
        for k in st:
            st[k] = 0
        st["recent"] = []
        st["last_head_wall"] = 0.0""",
    PROC,
)
report.append(f"proc report recent: {'already' if done else 'applied'}")

# v2: per-wave idx stash at head + pass wave_idx/tp_rank into request.
# Multi-state tolerant: works on fresh worktree, on the legacy-runner-tp
# intermediate, and no-ops when the final form is already present.
STASH_FINAL = """        st = RuntimeGuardBusMixin._bus_stats(self)
        t_sub = time.perf_counter_ns()
        probe_idx = int(getattr(self, "_probe_bus_idx", 0)) + 1
        self._probe_bus_idx = probe_idx
        self._probe_head_idx = probe_idx
        self._probe_head_wall = float(st.get("last_head_wall", 0.0) or time.time())
        self._probe_sync_at_head = int(getattr(self, "_probe_sync_idx", 0))
        try:
            self._probe_tp = int(getattr(sync_group, "rank_in_group", -1))
        except Exception:
            self._probe_tp = int(getattr(getattr(self, "runner", None), "tp_rank", -1))
        try:
            worker.submit(
                MergedBusRequest(
                    sync_group=sync_group,
                    config_due_local=config_due_local,
                    dump_due_local=dump_due_local,
                    dump_jobs=dump_jobs,
                    hot_reload_enabled=bool(cfg.hot_reload_enabled),
                    is_first_rank=is_first,
                    build_config_payload=build,
                    wave_idx=probe_idx,
                    tp_rank=self._probe_tp,
                )
            )"""
STASH_FRESH = """        st = RuntimeGuardBusMixin._bus_stats(self)
        t_sub = time.perf_counter_ns()
        try:
            worker.submit(
                MergedBusRequest(
                    sync_group=sync_group,
                    config_due_local=config_due_local,
                    dump_due_local=dump_due_local,
                    dump_jobs=dump_jobs,
                    hot_reload_enabled=bool(cfg.hot_reload_enabled),
                    is_first_rank=is_first,
                    build_config_payload=build,
                )
            )"""
STASH_LEGACY_TP = """        st = RuntimeGuardBusMixin._bus_stats(self)
        t_sub = time.perf_counter_ns()
        probe_idx = int(getattr(self, "_probe_bus_idx", 0)) + 1
        self._probe_bus_idx = probe_idx
        self._probe_head_idx = probe_idx
        self._probe_head_wall = float(st.get("last_head_wall", 0.0) or time.time())
        self._probe_sync_at_head = int(getattr(self, "_probe_sync_idx", 0))
        self._probe_tp = int(getattr(getattr(self, "runner", None), "tp_rank", -1))
        try:
            worker.submit(
                MergedBusRequest(
                    sync_group=sync_group,
                    config_due_local=config_due_local,
                    dump_due_local=dump_due_local,
                    dump_jobs=dump_jobs,
                    hot_reload_enabled=bool(cfg.hot_reload_enabled),
                    is_first_rank=is_first,
                    build_config_payload=build,
                    wave_idx=probe_idx,
                    tp_rank=self._probe_tp,
                )
            )"""
if STASH_FINAL in t:
    report.append("proc head idx stash: already")
else:
    for variant, tag in ((STASH_FRESH, "fresh"), (STASH_LEGACY_TP, "legacy-tp")):
        if t.count(variant) == 1:
            t = t.replace(variant, STASH_FINAL, 1)
            report.append(f"proc head idx stash: applied (from {tag})")
            break
    else:
        raise SystemExit("ANCHOR: proc head idx stash — no single fresh/legacy match")

# v2: live per-wave [RG-BUS-WV2] line at drain (head/AR/forward timeline)
t, done = sub_once(
    t,
    """        if len(st["recent"]) > 8:
            st["recent"].pop(0)
        changed = RuntimeGuardBusMixin._apply_merged_bus_result(""",
    """        if len(st["recent"]) > 8:
            st["recent"].pop(0)
        probe_head = float(getattr(self, "_probe_head_wall", 0.0))
        probe_idx = int(getattr(self, "_probe_head_idx", -1))
        probe_sync = int(getattr(self, "_probe_sync_at_head", -1))
        probe_tp = int(getattr(self, "_probe_tp", -1))
        fwd_in = float(getattr(self, "_probe_fwd_in", 0.0))
        fwd_out = float(getattr(self, "_probe_fwd_out", 0.0))
        sub_w = float(tm.get("submit_wall", 0.0)) if tm else 0.0
        ar_in_w = float(tm.get("ar_enter_wall", 0.0)) if tm else 0.0
        ar_out_w = float(tm.get("ar_exit_wall", 0.0)) if tm else 0.0
        logger.info(
            "[RG-BUS-WV2] idx=%d sync=%d tp=%d head=%.3f submit=+%.1fms "
            "ar_in=+%.1fms ar_out=+%.1fms fwd_in=+%.1fms fwd_out=+%.1fms "
            "ar_call_us=%.1f ar_item_us=%.1f sched_us=%.1f wait_us=%.1f "
            "drain_end=%.3f gate=%s",
            probe_idx,
            probe_sync,
            probe_tp,
            probe_head,
            (sub_w - probe_head) * 1000.0,
            (ar_in_w - probe_head) * 1000.0,
            (ar_out_w - probe_head) * 1000.0,
            (fwd_in - probe_head) * 1000.0 if fwd_in else 0.0,
            (fwd_out - probe_head) * 1000.0 if fwd_out else 0.0,
            float(tm.get("ar_call_ns", 0)) / 1000.0 if tm else 0.0,
            float(tm.get("ar_item_ns", 0)) / 1000.0 if tm else 0.0,
            float(tm.get("sched", 0)) / 1000.0 if tm else 0.0,
            dt_wait / 1000.0,
            time.time(),
            str(tm.get("gate", "?")) if tm else "?",
        )
        self._probe_fwd_in = 0.0
        self._probe_fwd_out = 0.0
        changed = RuntimeGuardBusMixin._apply_merged_bus_result(""",
    PROC,
)
report.append(f"proc WV2 live line: {'already' if done else 'applied'}")
PROC.write_text(t)

# ---------- processor.py (v2) ----------
t = PROC_CORE.read_text()
t, done = sub_once(
    t,
    """        self._scheduler_output_for_step = scheduler_output
        try:
            self.wave_tracker.advance(allow_arm=allow_arm)""",
    """        self._scheduler_output_for_step = scheduler_output
        try:
            self._probe_sync_idx = int(getattr(self, "_probe_sync_idx", 0)) + 1
            self.wave_tracker.advance(allow_arm=allow_arm)""",
    PROC_CORE,
)
report.append(f"proc sync idx counter: {'already' if done else 'applied'}")
PROC_CORE.write_text(t)

# ---------- hooks.py (v2: forward start/end) ----------
t = HOOKS.read_text()
t, done = sub_once(
    t,
    "import functools\nfrom contextlib import nullcontext",
    "import functools\nimport time\nfrom contextlib import nullcontext",
    HOOKS,
)
report.append(f"hooks import time: {'already' if done else 'applied'}")

t, done = sub_once(
    t,
    """        guard.sync_for_step(scheduler_output=scheduler_output, allow_arm=allow_arm)
        try:
            return execute_model_fn(self, scheduler_output, *args, **kwargs)
        finally:
            # Collectives must stay lockstep — do not soft-fail this gate.
            if dummy_run or self.execute_model_state is None:
                guard.end_of_wave_sync(allow_arm=False)""",
    """        guard.sync_for_step(scheduler_output=scheduler_output, allow_arm=allow_arm)
        guard._probe_fwd_in = time.time()
        try:
            return execute_model_fn(self, scheduler_output, *args, **kwargs)
        finally:
            guard._probe_fwd_out = time.time()
            # Collectives must stay lockstep — do not soft-fail this gate.
            if dummy_run or self.execute_model_state is None:
                guard.end_of_wave_sync(allow_arm=False)""",
    HOOKS,
)
report.append(f"hooks fwd start/end: {'already' if done else 'applied'}")
HOOKS.write_text(t)

print("\n".join(report))
print("PROBE_PATCH_OK")
