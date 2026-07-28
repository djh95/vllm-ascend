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

"""DFX runtime config: one JSON leader + world broadcast (or per-rank file poll).

Preferred multi-node mode (``sync_mode=broadcast``, default):
  - Only global rank0 reads / writes the JSON file.
  - All ranks periodically join a collective; rank0 broadcasts the payload.
  - Other machines do **not** need a shared filesystem.

Fallback (``sync_mode=file``): every process polls the same path (shared FS);
writes (ensure_persisted / save / dump_once clear) are still leader / single-process only.

Production: ``AscendConfig`` builds this with ``ensure_file=False`` (API/EngineCore
safe). ``DfxProcessor`` on workers calls :meth:`DfxRuntimeConfig.ensure_persisted`
so only the worker leader materializes the JSON once.

Note: ``get_world_group()`` covers one engine replica (TP×PP×DP inside that
process group). Separate external-online-DP engines are not in one world —
use shared FS, or put the JSON on each engine's rank0.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

DEFAULT_CONFIG_FILENAME = "dfx_config.json"

# sync_mode values
SYNC_BROADCAST = "broadcast"
SYNC_FILE = "file"

# Logger names updated by ``apply_log_switches`` / ascend logging setup.
DFX_LOGGER_NAMES: tuple[str, ...] = (
    "vllm.vllm_ascend.dfx.dumper",
    "vllm.vllm_ascend.dfx.detector.spec_acceptance",
    "vllm.vllm_ascend.dfx.detector.token_logprob",
    "vllm.vllm_ascend.dfx.detector.manual_dump",
    "vllm.vllm_ascend.dfx.runtime_config",
    "vllm.vllm_ascend.dfx.report",
    "vllm.vllm_ascend.dfx.processor",
)


def default_dfx_root() -> Path:
    """Execution-directory DFX root: ``<cwd>/dfx``."""
    return Path(os.getcwd()) / "dfx"


def default_config_dir() -> Path:
    return default_dfx_root() / "config"


def default_report_dir() -> Path:
    return default_dfx_root() / "report"


_DEFAULTS: dict[str, Any] = {
    # broadcast: rank0 reads JSON, world-broadcasts; file: each rank polls path
    "sync_mode": SYNC_BROADCAST,
    # Kept in JSON for visibility; effective hot-reload interval is set at
    # process start via additional_config.dfx_config_reload_interval (default 5).
    # Set 0 at startup to disable. JSON field alone cannot re-enable after start.
    "reload_interval_seconds": 5,
    "dump": {
        "enabled": True,
        "max_times": 0,
        "cooldown_seconds": 5 * 60,
        # Manual one-shot: set true in JSON → next hot-reload arms dump, then cleared to false.
        # Requires additional_config.dfx_config_reload_interval > 0. Does not consume max_times / cooldown.
        "dump_once": False,
    },
    "log": {
        "enabled": True,
        "level": "INFO",
    },
    "metrics": {
        "enabled": True,
        "level": "INFO",
    },
    "trace": {
        "enabled": False,
        "level": "INFO",
        "otlp_endpoint": None,
    },
    "detector": {
        "enable_spec_acceptance_check": True,
        "enable_token_logprob_check": False,
        "spec_acceptance_window": 10,
        "spec_acceptance_low_threshold": 0.3,
        "spec_acceptance_len_low_threshold": 1.4,
        "spec_acceptance_high_threshold": 0.96,
        "spec_acceptance_len_high_threshold": 2.8,
        "token_logprob_window": 64,
        "token_logprob_stride": 32,
        "token_logprob_topk": 20,
        "ill_nan_window_thresh": 1,
        "ill_rare_window_thresh": 1,
        "ill_garbled_window_thresh": 1,
        "ill_repet_window_thresh": 2,
    },
}


def resolve_dfx_config_path(configured_path: str | None = None) -> Path:
    """Resolve config file path.

    Priority:
    1. Explicit ``dfx_config_path`` / ``dfx-config`` from additional_config
    2. Default ``<cwd>/dfx/config/dfx_config.json``
    """
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (default_config_dir() / DEFAULT_CONFIG_FILENAME).resolve()


def resolve_dfx_report_dir(config_path: Path, configured_report_dir: str | None = None) -> Path:
    if configured_report_dir:
        return Path(configured_report_dir).expanduser().resolve()
    dfx_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return (dfx_root / "report").resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _legacy_dynamic_dump_to_sections(legacy: dict[str, Any] | None) -> dict[str, Any]:
    """Map old ``dynamic_dump_config`` flat keys into dump/detector sections."""
    if not legacy:
        return {}
    dump_keys = {"dynamic_dump_max_times", "dynamic_dump_cooldown_seconds"}
    dump: dict[str, Any] = {}
    detector: dict[str, Any] = {}
    for key, value in legacy.items():
        if key == "dynamic_dump_max_times":
            dump["max_times"] = value
        elif key == "dynamic_dump_cooldown_seconds":
            dump["cooldown_seconds"] = value
        elif key not in dump_keys:
            detector[key] = value
    out: dict[str, Any] = {}
    if dump:
        out["dump"] = dump
    if detector:
        out["detector"] = detector
    return out


def _world_group_or_none():
    try:
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group()
    except Exception:
        return None


def _is_json_writer() -> bool:
    """True if this process may write the DFX JSON (leader or single process).

    Prefer world group when distributed is up; otherwise ``RANK`` from the
    launcher (torchrun / vLLM). Unset ``RANK`` → treat as single-process writer.
    """
    world = _world_group_or_none()
    if world is not None and world.world_size > 1:
        return bool(world.is_first_rank)
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            pass
    return True


class DfxRuntimeConfig:
    """Runtime DFX switches loaded from JSON, optionally world-broadcast.

    Prefer this name over a bare ``config`` module: it is a live control plane,
    not static build/packaging config.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        legacy_dynamic_dump: dict[str, Any] | None = None,
        report_dir: str | Path | None = None,
        ensure_file: bool = False,
        sync_mode: str | None = None,
        reload_interval_seconds: float | int | None = None,
    ) -> None:
        # None → default ``<cwd>/dfx/config/dfx_config.json`` (not an "explicit" path).
        self._explicit_config_path = config_path is not None
        self.config_path = resolve_dfx_config_path(str(config_path) if config_path is not None else None)
        self.report_dir = resolve_dfx_report_dir(
            self.config_path,
            str(report_dir) if report_dir is not None else None,
        )
        # Startup override: None → default 5s hot reload; 0 → off; >0 → every N seconds.
        # This is authoritative and is not re-enabled by JSON after load.
        if reload_interval_seconds is None:
            self._reload_interval = 5.0
        else:
            self._reload_interval = float(reload_interval_seconds)
        if self._reload_interval < 0:
            raise ValueError(f"dfx_config_reload_interval must be >= 0, got {self._reload_interval}")
        self._mtime: float | None = None
        self._version: float = 0.0
        self._last_reload_ts = 0.0
        # Explicit startup keys only (mapped to dump/detector sections).
        self._startup_overlay = _legacy_dynamic_dump_to_sections(legacy_dynamic_dump)
        self._ctor_sync_mode = sync_mode
        self._initial_broadcast_done = False
        self._data = deepcopy(_DEFAULTS)
        self._bootstrap_persisted = False
        self._bg_reloader_started = False
        self._bg_thread: threading.Thread | None = None

        # In-memory merge always. ``ensure_file=True`` persists immediately (tests /
        # rare callers). Production AscendConfig uses False; worker leader calls
        # :meth:`ensure_persisted` once from ``DfxProcessor``.
        self._bootstrap(persist=ensure_file)
        logger.info(
            "[DFX runtime_config] path=%s explicit_path=%s report_dir=%s hot_reload=%s persisted=%s",
            self.config_path,
            self._explicit_config_path,
            self.report_dir,
            self.hot_reload_enabled,
            self._bootstrap_persisted,
        )
        if self.hot_reload_enabled:
            logger.info_once(
                "[DFX runtime_config] hot-reload enabled interval=%.3fs sync_mode=%s path=%s",
                self.reload_interval_seconds,
                self.sync_mode,
                str(self.config_path),
            )
        else:
            logger.info_once(
                "[DFX runtime_config] hot-reload disabled "
                "(set additional_config.dfx_config_reload_interval > 0 to enable; "
                "default is 5; dump.dump_once also requires interval > 0)"
            )

    def _read_json_object(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                logger.error(
                    "[DFX runtime_config] root must be object, got %s; ignoring file",
                    type(loaded).__name__,
                )
                return {}
            return loaded
        except Exception as exc:
            logger.warning(
                "[DFX runtime_config] failed to read path=%s error=%s; using defaults",
                self.config_path,
                exc,
            )
            return {}

    def _merge_bootstrap(self, loaded: dict[str, Any]) -> dict[str, Any]:
        """Build effective config for process start.

        - No explicit ``dfx_config_path`` + startup overlay: **overwrite** JSON
          basis (defaults ← startup only; ignore prior file content).
        - Explicit path (or no startup overlay): defaults ← JSON ← startup.
        Missing keys always come from ``_DEFAULTS``.
        """
        if not self._explicit_config_path and self._startup_overlay:
            # No path configured: startup dynamic_dump_config covers the JSON.
            merged = _deep_merge(_DEFAULTS, self._startup_overlay)
        else:
            merged = _deep_merge(_DEFAULTS, loaded)
            if self._startup_overlay:
                merged = _deep_merge(merged, self._startup_overlay)
        if self._ctor_sync_mode is not None:
            merged["sync_mode"] = self._ctor_sync_mode
        # Persist startup hot-reload interval for visibility (runtime gate is still
        # ``self._reload_interval`` only).
        merged["reload_interval_seconds"] = self._reload_interval
        return merged

    def _write_data_unlocked(self, data: dict[str, Any]) -> None:
        """Atomic write; caller must hold config lock / own the path."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, self.config_path)

    def _bootstrap(self, *, persist: bool) -> None:
        """Load / merge / optionally save complete effective config at startup.

        Disk write is leader-only (or single-process); other ranks keep in-memory merge.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        overwrite_default = bool(not self._explicit_config_path and self._startup_overlay)
        # Explicit path (or no startup overlay): read JSON first.
        # No path + startup overlay: ignore prior file ("覆盖").
        if self._explicit_config_path or not self._startup_overlay:
            loaded = self._read_json_object()
        else:
            loaded = {}
        merged = self._merge_bootstrap(loaded)
        self._validate(merged)

        can_write = persist and _is_json_writer()
        if overwrite_default:
            logger.info(
                "[DFX runtime_config] overwrite default json path=%s "
                "reason=no_dfx_config_path+startup_overlay will_persist=%s",
                self.config_path,
                can_write,
            )

        if can_write:
            try:
                with self._lock_config():
                    self._write_data_unlocked(merged)
                    mtime = self.config_path.stat().st_mtime
                self._apply_loaded(merged, version=mtime)
                self._bootstrap_persisted = True
                logger.info(
                    "[DFX runtime_config] bootstrap saved path=%s explicit_path=%s "
                    "startup_overlay=%s overwrite_default=%s",
                    self.config_path,
                    self._explicit_config_path,
                    bool(self._startup_overlay),
                    overwrite_default,
                )
            except Exception as exc:
                logger.warning(
                    "[DFX runtime_config] bootstrap save failed path=%s error=%s; using in-memory",
                    self.config_path,
                    exc,
                )
                self._data = merged
                self._version = 0.0
        else:
            self._data = merged
            if overwrite_default and self.config_path.exists():
                # Deferred persist: drop stale default JSON so API/EngineCore
                # background reload cannot re-apply old disk content before the
                # worker leader writes defaults←startup.
                try:
                    self.config_path.unlink()
                    logger.info(
                        "[DFX runtime_config] removed stale default json path=%s "
                        "(overwrite+deferred persist; awaiting worker leader)",
                        self.config_path,
                    )
                except OSError as exc:
                    logger.warning(
                        "[DFX runtime_config] failed to remove stale default json path=%s error=%s",
                        self.config_path,
                        exc,
                    )
                self._mtime = None
                self._version = 0.0
            elif self.config_path.exists():
                try:
                    self._mtime = self.config_path.stat().st_mtime
                    self._version = float(self._mtime)
                except OSError:
                    self._version = 0.0
            else:
                self._version = 0.0
            if persist and not _is_json_writer():
                logger.info(
                    "[DFX runtime_config] bootstrap skip persist (non-leader) path=%s",
                    self.config_path,
                )
        self._last_reload_ts = time.time()

    def ensure_persisted(self) -> bool:
        """Write bootstrap merge to disk once (worker leader / single-process only).

        Safe to call from every worker: non-leaders no-op; leaders write at most once
        per process. Call from ``DfxProcessor`` so API/EngineCore never persist.
        """
        if self._bootstrap_persisted:
            return True
        if not _is_json_writer():
            logger.info(
                "[DFX runtime_config] ensure_persisted skip (non-leader) path=%s",
                self.config_path,
            )
            return False
        try:
            with self._lock_config():
                self._write_data_unlocked(self._data)
                mtime = self.config_path.stat().st_mtime
            self._mtime = mtime
            self._version = float(mtime)
            self._bootstrap_persisted = True
            logger.info(
                "[DFX runtime_config] worker leader persisted path=%s explicit_path=%s startup_overlay=%s",
                self.config_path,
                self._explicit_config_path,
                bool(self._startup_overlay),
            )
            return True
        except Exception as exc:
            logger.warning(
                "[DFX runtime_config] ensure_persisted failed path=%s error=%s",
                self.config_path,
                exc,
            )
            return False

    # ---- section accessors -------------------------------------------------

    @property
    def hot_reload_enabled(self) -> bool:
        """True when startup ``dfx_config_reload_interval`` > 0."""
        return self._reload_interval > 0

    @property
    def sync_mode(self) -> str:
        mode = str(self._data.get("sync_mode", SYNC_BROADCAST)).lower()
        return mode if mode in (SYNC_BROADCAST, SYNC_FILE) else SYNC_BROADCAST

    @property
    def reload_interval_seconds(self) -> float:
        """Effective hot-reload period from startup; 0 means disabled."""
        return self._reload_interval

    @property
    def dump(self) -> dict[str, Any]:
        return self._data["dump"]

    @property
    def log(self) -> dict[str, Any]:
        return self._data["log"]

    @property
    def metrics(self) -> dict[str, Any]:
        return self._data["metrics"]

    @property
    def trace(self) -> dict[str, Any]:
        return self._data["trace"]

    @property
    def detector(self) -> dict[str, Any]:
        return self._data["detector"]

    def dump_enabled(self) -> bool:
        return bool(self.dump.get("enabled", True))

    def dump_max_times(self) -> int:
        return int(self.dump.get("max_times", 0))

    def dump_cooldown_seconds(self) -> int:
        return int(self.dump.get("cooldown_seconds", 300))

    def dump_once(self) -> bool:
        """Manual one-shot dump request from JSON (``dump.dump_once``).

        Only observed after a successful hot-reload; requires
        ``dfx_config_reload_interval > 0``.
        """
        return bool(self.dump.get("dump_once", False))

    def consume_dump_once(self) -> bool:
        """If ``dump_once`` is true, clear it and return True.

        All ranks clear in-memory; only the JSON writer (leader / single
        process) persists ``false``.
        """
        if not self.dump_once():
            return False
        self.dump["dump_once"] = False
        if _is_json_writer():
            if self.save({"dump": {"dump_once": False}}):
                logger.info(
                    "[DFX runtime_config] dump_once consumed → false path=%s",
                    self.config_path,
                )
            else:
                logger.warning(
                    "[DFX runtime_config] dump_once cleared in-memory but failed to persist path=%s",
                    self.config_path,
                )
        return True

    def log_enabled(self) -> bool:
        return bool(self.log.get("enabled", True))

    def log_level(self) -> str:
        return str(self.log.get("level", "INFO")).upper()

    def metrics_enabled(self) -> bool:
        return bool(self.metrics.get("enabled", True))

    def metrics_level(self) -> str:
        return str(self.metrics.get("level", "INFO")).upper()

    def trace_enabled(self) -> bool:
        return bool(self.trace.get("enabled", False))

    def trace_level(self) -> str:
        return str(self.trace.get("level", "INFO")).upper()

    def trace_otlp_endpoint(self) -> str | None:
        endpoint = self.trace.get("otlp_endpoint")
        return str(endpoint) if endpoint else None

    def detector_get(self, key: str, default: Any = None) -> Any:
        return self.detector.get(key, default)

    def apply_log_switches(self) -> None:
        """Apply ``log.enabled`` / ``log.level`` to DFX loggers (this process only)."""
        import logging

        if not self.log_enabled():
            return
        level = getattr(logging, self.log_level(), logging.INFO)
        for name in DFX_LOGGER_NAMES:
            logging.getLogger(name).setLevel(level)

    def start_non_worker_background_reload(self) -> bool:
        """Daemon thread: file-poll JSON and apply log switches (API / EngineCore).

        - No-op when hot-reload is off, or ``RANK`` is set (distributed worker).
        - Uses **local file reload only** — never joins worker world broadcast.
        - Does not persist JSON.
        Workers keep step-driven :meth:`sync_dfx_config` and must not call this.
        """
        if not self.hot_reload_enabled:
            return False
        if os.environ.get("RANK") is not None:
            logger.info(
                "[DFX runtime_config] skip non-worker reloader (RANK set) path=%s",
                self.config_path,
            )
            return False
        if self._bg_reloader_started:
            return False
        self._bg_reloader_started = True
        interval = self.reload_interval_seconds

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    # Wait for worker leader to materialize the file after
                    # overwrite+delete; avoid no-op thrashing when missing.
                    if not self.config_path.exists():
                        continue
                    # Force file poll path even if JSON says broadcast — this
                    # process is outside the worker world group.
                    if self._maybe_reload_local():
                        self.apply_log_switches()
                        logger.info(
                            "[DFX runtime_config] non-worker reload applied path=%s log.level=%s version=%.6f",
                            self.config_path,
                            self.log_level(),
                            self._version,
                        )
                except Exception as exc:
                    logger.warning(
                        "[DFX runtime_config] non-worker reload error path=%s error=%s",
                        self.config_path,
                        exc,
                    )

        self._bg_thread = threading.Thread(
            target=_loop,
            name="dfx-non-worker-reload",
            daemon=True,
        )
        self._bg_thread.start()
        logger.info(
            "[DFX runtime_config] non-worker background reload started interval=%.3fs path=%s",
            interval,
            self.config_path,
        )
        return True

    def sync_dfx_config(self) -> bool:
        """All-rank config sync entry. Hot-reload off → immediate ``False``.

        Must be called on **every** rank each step when broadcast mode is used;
        skipping early-PP would deadlock ``broadcast_object``.
        """
        return self.maybe_reload()

    def maybe_reload(self) -> bool:
        """Interval-gated sync. No-op when hot-reload is disabled (interval<=0).

        Broadcast mode is a collective — call on all ranks when enabled.
        Prefer :meth:`sync_dfx_config` at runner step entry.
        """
        if not self.hot_reload_enabled:
            return False
        world = _world_group_or_none()
        if self.sync_mode == SYNC_BROADCAST and world is not None and world.world_size > 1:
            return self._maybe_reload_broadcast(world)
        return self._maybe_reload_local()

    def _maybe_reload_local(self) -> bool:
        now = time.time()
        if now - self._last_reload_ts < self.reload_interval_seconds:
            return False
        return self.reload(force=False)

    def _maybe_reload_broadcast(self, world) -> bool:
        """Rank0 reads JSON; all ranks all_reduce(due) then broadcast_object."""
        import torch

        now = time.time()
        interval = self.reload_interval_seconds
        due_local = 1.0 if ((not self._initial_broadcast_done) or (now - self._last_reload_ts >= interval)) else 0.0
        due_t = torch.tensor([due_local], dtype=torch.float32)
        torch.distributed.all_reduce(
            due_t,
            op=torch.distributed.ReduceOp.MAX,
            group=world.cpu_group,
        )
        if float(due_t.item()) < 0.5:
            return False

        self._last_reload_ts = now
        changed = False
        payload: dict[str, Any] | None = None
        if world.is_first_rank:
            # Always re-stat; reload() no-ops when mtime unchanged.
            changed = self.reload(force=False)
            payload = {
                "version": float(self._version),
                "data": deepcopy(self._data),
            }
        payload = world.broadcast_object(payload, src=0)
        first_sync = not self._initial_broadcast_done
        self._initial_broadcast_done = True
        if payload is None or not isinstance(payload, dict):
            return False
        version = float(payload.get("version", 0.0))
        data = payload.get("data")
        if not isinstance(data, dict):
            return False
        if not world.is_first_rank:
            if version != self._version:
                return self._apply_loaded(data, version=version)
            return False
        # Leader: true if file changed, or first world sync (refresh callers once).
        return changed or first_sync

    def reload(self, *, force: bool = False) -> bool:
        """Local file reload (leader / file mode / pre-dist bootstrap).

        Hot-reload follows JSON only: ``defaults ← JSON``.
        Startup overlay is applied once at bootstrap (and persisted), not on reload.
        """
        self._last_reload_ts = time.time()
        if not self.config_path.exists():
            if force:
                self._data = deepcopy(_DEFAULTS)
                self._version = 0.0
            return False

        try:
            mtime = self.config_path.stat().st_mtime
        except OSError as exc:
            logger.warning("[DFX runtime_config] stat failed path=%s error=%s", self.config_path, exc)
            return False

        if not force and self._mtime is not None and mtime <= self._mtime:
            return False

        try:
            with self._lock_config(), self.config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                logger.error("[DFX runtime_config] root must be object, got %s", type(loaded).__name__)
                return False
            merged = _deep_merge(_DEFAULTS, loaded)
            return self._apply_loaded(merged, version=mtime)
        except Exception as exc:
            logger.error("[DFX runtime_config] reload failed path=%s error=%s", self.config_path, exc)
            return False

    def _apply_loaded(self, merged: dict[str, Any], *, version: float) -> bool:
        self._validate(merged)
        self._data = merged
        self._mtime = version
        self._version = version
        # Log every apply so hot-reload / broadcast updates are visible (not info_once).
        logger.info(
            "[DFX runtime_config] applied path=%s sync_mode=%s version=%.6f "
            "dump.enabled=%s dump.max_times=%s dump.cooldown=%s log.level=%s "
            "spec_check=%s token_logprob_check=%s",
            str(self.config_path),
            self.sync_mode,
            self._version,
            self.dump_enabled(),
            self.dump_max_times(),
            self.dump_cooldown_seconds(),
            self.log_level(),
            self.detector_get("enable_spec_acceptance_check"),
            self.detector_get("enable_token_logprob_check"),
        )
        return True

    def save(self, updates: dict[str, Any] | None = None) -> bool:
        """Merge ``updates`` and write JSON. Leader (or single-process) only."""
        if not _is_json_writer():
            logger.debug(
                "[DFX runtime_config] save ignored on non-leader path=%s",
                self.config_path,
            )
            return False
        try:
            with self._lock_config():
                data = deepcopy(self._data)
                if updates:
                    data = _deep_merge(data, updates)
                self._validate(data)
                self._write_data_unlocked(data)
                self._data = data
                self._mtime = self.config_path.stat().st_mtime
                self._version = float(self._mtime)
            logger.info("[DFX runtime_config] saved path=%s", self.config_path)
            return True
        except Exception as exc:
            logger.error("[DFX runtime_config] save failed path=%s error=%s", self.config_path, exc)
            return False

    def _lock_config(self):
        lock_path = Path(f"{self.config_path}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        class _LockCtx:
            def __enter__(self_inner):
                self_inner._fd = lock_path.open("w", encoding="utf-8")
                fcntl.flock(self_inner._fd, fcntl.LOCK_EX)
                return self_inner._fd

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    fcntl.flock(self_inner._fd, fcntl.LOCK_UN)
                finally:
                    self_inner._fd.close()

        return _LockCtx()

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        for section in ("dump", "log", "metrics", "trace", "detector"):
            if section not in data or not isinstance(data[section], dict):
                raise ValueError(f"dfx config missing object section '{section}'")
        interval = data.get("reload_interval_seconds", 0)
        if not isinstance(interval, (int, float)) or interval < 0:
            raise ValueError(f"reload_interval_seconds must be >= 0, got {interval}")
        sync_mode = str(data.get("sync_mode", SYNC_BROADCAST)).lower()
        if sync_mode not in (SYNC_BROADCAST, SYNC_FILE):
            raise ValueError(f"sync_mode must be '{SYNC_BROADCAST}' or '{SYNC_FILE}'")
        for flag_section in ("dump", "log", "metrics", "trace"):
            enabled = data[flag_section].get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                raise ValueError(f"{flag_section}.enabled must be bool")
        dump_once = data["dump"].get("dump_once")
        if dump_once is not None and not isinstance(dump_once, bool):
            raise ValueError("dump.dump_once must be bool")
        for level_section in ("log", "metrics", "trace"):
            level = data[level_section].get("level", "INFO")
            if not isinstance(level, str):
                raise ValueError(f"{level_section}.level must be str")
        detector = data["detector"]
        window = detector.get("token_logprob_window", 64)
        stride = detector.get("token_logprob_stride", 32)
        if int(window) < int(stride):
            raise ValueError("detector.token_logprob_window must be >= token_logprob_stride")
