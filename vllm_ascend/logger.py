# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging configuration for vLLM-Ascend.

Provides two logging mechanisms:
1. Console: A dedicated handler on the vllm_ascend logger with
   [vllm-ascend] [module] prefix. No modification to vLLM's global
   logging state — safe for upstream tests and multiprocessing.
2. File: A rotating file handler on both vllm and vllm_ascend loggers,
   capturing all logs with Ascend formatting.
"""

import logging
import os
import sys
import time
from contextlib import suppress
from datetime import datetime

from vllm import envs
from vllm.logger import init_logger
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
# Second-level strftime pattern; milliseconds are appended in formatTime.
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _format_time_ms(formatter: logging.Formatter, record: logging.LogRecord, datefmt: str | None = None) -> str:
    """Format record time with millisecond precision (strftime has no %f here)."""
    ct = formatter.converter(record.created)
    s = time.strftime(datefmt or _DATE_FORMAT, ct)
    return f"{s}.{int(record.msecs):03d}"


_LOG_DIR = os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend")
_LOG_MAX_BYTES = 20 * 1024 * 1024


def init_logger_ascend(name: str) -> logging.Logger:
    """Logger under ``vllm_ascend.*`` (Ascend handler), not ``vllm.*``.

    Nesting under ``vllm.`` makes records hit the root ``vllm`` StreamHandler,
    whose level tracks ``VLLM_LOGGING_LEVEL`` (often INFO). That filters out
    DEBUG even when package / module logger levels are DEBUG.
    Ascend's own handler stays at DEBUG; levels are gated by
    :func:`apply_ascend_log_level`.
    """
    return init_logger(name)


def _resolve_log_level(level: str) -> int:
    return getattr(logging, str(level).upper(), logging.INFO)


def _normalize_debug_module(entry: str) -> str:
    """Map config entry to a ``vllm_ascend.*`` logger name."""
    name = str(entry).strip().strip(".")
    if not name:
        return ""
    if name == "vllm_ascend" or name.startswith("vllm_ascend."):
        return name
    if name.startswith("vllm.vllm_ascend"):
        # Legacy tree → primary Ascend namespace.
        return name[len("vllm.") :] if name.startswith("vllm.") else name
    return f"vllm_ascend.{name}"


def _iter_ascend_logger_names() -> list[str]:
    names: list[str] = []
    for name in list(logging.Logger.manager.loggerDict):
        if not isinstance(name, str):
            continue
        if name.startswith("vllm_ascend.") or name.startswith("vllm.vllm_ascend."):
            names.append(name)
    return names


def _set_logger_tree_level(prefix: str, level: int) -> None:
    """Set ``prefix`` and any already-created ``prefix.*`` loggers."""
    logging.getLogger(prefix).setLevel(level)
    legacy = f"vllm.{prefix}" if prefix.startswith("vllm_ascend") else None
    if legacy:
        logging.getLogger(legacy).setLevel(level)
    dotted = prefix + "."
    legacy_dotted = (legacy + ".") if legacy else None
    for name in _iter_ascend_logger_names():
        if (
            name == prefix
            or name.startswith(dotted)
            or legacy
            and (name == legacy or (legacy_dotted and name.startswith(legacy_dotted)))
        ):
            logging.getLogger(name).setLevel(level)


_last_applied_ascend_log: tuple[str, tuple[str, ...]] | None = None
_log_chain_probe_done: set[str] = set()

# Loggers compared when diagnosing UC vs Ascend formatting.
_LOG_CHAIN_PROBE_NAMES: tuple[str, ...] = (
    "vllm_ascend",
    "vllm_ascend.logger",
    "vllm_ascend.dfx",
    "vllm_ascend.dfx.detector.spec_acceptance",
    "vllm_ascend.dfx.dumper",
    "vllm_ascend.dfx.runtime_config",
    "vllm.logger",
    "vllm",
    "root",
    "UC",
)


def _handler_brief(handler: logging.Handler) -> str:
    fmt = handler.formatter
    fmt_name = type(fmt).__name__ if fmt is not None else "None"
    stream = getattr(handler, "stream", None)
    stream_name = getattr(stream, "name", type(stream).__name__) if stream is not None else "-"
    return f"{type(handler).__name__}(level={logging.getLevelName(handler.level)},fmt={fmt_name},stream={stream_name})"


def format_logger_chain(name: str) -> str:
    """Describe logger→parent handler chain (safe with PlaceHolder parents)."""
    if name == "root":
        cur: logging.Logger | logging.PlaceHolder | None = logging.root
    else:
        cur = logging.getLogger(name)
    parts: list[str] = []
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if not isinstance(cur, logging.Logger):
            parts.append(f"{getattr(cur, 'name', '?')}({type(cur).__name__})")
            break
        handlers = [_handler_brief(h) for h in cur.handlers]
        parts.append(
            f"{cur.name}[level={logging.getLevelName(cur.level)}/"
            f"eff={logging.getLevelName(cur.getEffectiveLevel())},"
            f"prop={cur.propagate},handlers={handlers or '[]'}]"
        )
        if not cur.propagate or cur is logging.root:
            break
        cur = cur.parent
    return " -> ".join(parts)


def _qualname(obj: object) -> str:
    mod = getattr(obj, "__module__", "?")
    name = getattr(obj, "__qualname__", getattr(obj, "__name__", type(obj).__name__))
    return f"{mod}.{name}"


def _describe_io_hooks() -> list[str]:
    """Detect stdout/stderr/logging emit patches that can rewrite lines as ``[UC]``."""
    lines: list[str] = []
    for label, stream, orig in (
        ("stdout", sys.stdout, sys.__stdout__),
        ("stderr", sys.stderr, sys.__stderr__),
    ):
        write = getattr(stream, "write", None)
        orig_write = getattr(orig, "write", None) if orig is not None else None
        same_as_orig = write is orig_write if orig_write is not None else None
        lines.append(
            f"[ascend_log_chain] {label}: type={type(stream).__module__}.{type(stream).__name__} "
            f"write={_qualname(write) if write is not None else None} "
            f"same_as_dunder={same_as_orig}"
        )
    # Compare live StreamHandler.emit to the stdlib unbound function.
    std_emit = logging.StreamHandler.emit
    lines.append(f"[ascend_log_chain] StreamHandler.emit={_qualname(std_emit)}")
    for logger_name in ("vllm_ascend", "vllm"):
        lg = logging.getLogger(logger_name)
        for idx, handler in enumerate(lg.handlers):
            emit = getattr(handler, "emit", None)
            emit_func = getattr(emit, "__func__", emit)
            patched = emit_func is not std_emit
            lines.append(
                f"[ascend_log_chain] {logger_name}.handlers[{idx}] "
                f"type={type(handler).__name__} emit={_qualname(emit) if emit else None} "
                f"emit_patched={patched} "
                f"formatter={type(handler.formatter).__name__ if handler.formatter else None}"
            )
    lines.append(f"[ascend_log_chain] Logger.manager.class={_qualname(logging.getLoggerClass())}")
    factory = logging.getLogRecordFactory()
    lines.append(f"[ascend_log_chain] LogRecordFactory={_qualname(factory)}")
    return lines


def log_logger_chain_probe(reason: str, *, force: bool = False) -> None:
    """Once-per-reason dump of key logger chains (stderr + ascend logger).

    Used to see which Handler formats DFX lines as ``[UC]`` vs ``[vllm-ascend]``.
    Set ``VLLM_ASCEND_LOG_CHAIN_PROBE=0`` to disable.
    """
    if os.environ.get("VLLM_ASCEND_LOG_CHAIN_PROBE", "1").strip() in ("0", "false", "False"):
        return
    if not force and reason in _log_chain_probe_done:
        return
    _log_chain_probe_done.add(reason)

    lines: list[str] = [f"[ascend_log_chain] reason={reason} pid={os.getpid()}"]
    # Any logger whose name looks like UC / slog collector.
    uc_names = sorted(
        n
        for n in list(logging.Logger.manager.loggerDict)
        if isinstance(n, str) and (n.upper() == "UC" or "UC" in n.split(".") or n.endswith(".UC"))
    )
    lines.append(f"[ascend_log_chain] uc_logger_names={uc_names or []}")
    lines.append(f"[ascend_log_chain] root_handlers={[_handler_brief(h) for h in logging.root.handlers] or []}")
    lines.extend(_describe_io_hooks())
    for probe_name in _LOG_CHAIN_PROBE_NAMES:
        try:
            lines.append(f"[ascend_log_chain] {format_logger_chain(probe_name)}")
        except Exception as exc:  # noqa: BLE001 — diagnostics must not break serving
            lines.append(f"[ascend_log_chain] {probe_name} ERROR {type(exc).__name__}: {exc}")

    # A/B markers: same process, different loggers — see which lines become [UC] downstream.
    marker = f"ascend_log_chain_marker reason={reason} pid={os.getpid()}"
    lines.append(f"[ascend_log_chain] emitting A/B markers: {marker}")

    text = "\n".join(lines)
    # stderr bypasses logging Handlers (so UC Formatter cannot rewrite this dump).
    print(text, file=sys.stderr, flush=True)
    with suppress(Exception):
        logging.getLogger("vllm_ascend.logger").info("[A] %s via=vllm_ascend.logger", marker)
        logging.getLogger("vllm_ascend.dfx.runtime_config").info("[B] %s via=vllm_ascend.dfx.runtime_config", marker)
        logging.getLogger("vllm.logger").info("[C] %s via=vllm.logger", marker)
        logging.getLogger("vllm_ascend.logger").info("%s", text)


def apply_ascend_log_level(
    level: str = "INFO",
    debug_modules: list[str] | None = None,
) -> None:
    """Apply package root level and per-module DEBUG whitelist.

    Args:
        level: Level for the ``vllm_ascend`` root logger (e.g. ``INFO``).
        debug_modules: Relative module paths forced to DEBUG, such as
            ``\"dfx\"`` or ``\"dfx.runtime_config\"`` (mapped to
            ``vllm_ascend.<entry>``). Full ``vllm_ascend.*`` names are also
            accepted.
    """
    global _last_applied_ascend_log

    configure_ascend_logging()
    root_level = _resolve_log_level(level)
    debug_list = list(debug_modules or ())
    level_key = str(level).upper()
    debug_key = tuple(debug_list)
    needs_debug = root_level <= logging.DEBUG or bool(debug_list)
    announce = _last_applied_ascend_log != (level_key, debug_key)

    ascend = logging.getLogger("vllm_ascend")
    ascend.setLevel(root_level)
    ascend.propagate = False
    # Keep handlers able to emit DEBUG; logger.level is the gate.
    for h in ascend.handlers:
        if h.level > logging.DEBUG:
            h.setLevel(logging.DEBUG)

    legacy = logging.getLogger("vllm.vllm_ascend")
    legacy.setLevel(root_level)
    # Avoid INFO-filtered ``vllm`` StreamHandler swallowing DEBUG.
    legacy.propagate = False
    if ascend.handlers:
        for h in ascend.handlers:
            if h not in legacy.handlers:
                legacy.addHandler(h)

    # Reset known children to the root level (clears a prior debug whitelist).
    for name in _iter_ascend_logger_names():
        logging.getLogger(name).setLevel(root_level)

    for entry in debug_list:
        prefix = _normalize_debug_module(entry)
        if not prefix or prefix == "vllm_ascend":
            # Root already set; ignore empty / redundant root entries.
            if prefix == "vllm_ascend":
                ascend.setLevel(logging.DEBUG)
                legacy.setLevel(logging.DEBUG)
            continue
        _set_logger_tree_level(prefix, logging.DEBUG)

    if needs_debug:
        # Outer collectors (e.g. UC) often attach INFO-level handlers on root /
        # ``vllm``. Lower those handlers so DEBUG records are not dropped after
        # our loggers allow them. Logger levels elsewhere stay unchanged.
        for h in logging.root.handlers:
            if h.level > logging.DEBUG:
                h.setLevel(logging.DEBUG)
        vllm_logger = logging.getLogger("vllm")
        for h in vllm_logger.handlers:
            if h.level > logging.DEBUG:
                h.setLevel(logging.DEBUG)
        for name in list(logging.Logger.manager.loggerDict):
            if not isinstance(name, str):
                continue
            # Huawei UC / slog-style module loggers seen in the wild.
            if name.upper() == "UC" or name.endswith(".UC"):
                lg = logging.getLogger(name)
                if lg.level > logging.DEBUG and lg.level != logging.NOTSET:
                    lg.setLevel(logging.DEBUG)
                for h in lg.handlers:
                    if h.level > logging.DEBUG:
                        h.setLevel(logging.DEBUG)

    _last_applied_ascend_log = (level_key, debug_key)
    probe = logging.getLogger("vllm_ascend.dfx")
    if announce:
        # INFO so operators can confirm apply even when DEBUG is still filtered.
        logging.getLogger("vllm_ascend.logger").info(
            "[ascend_log] applied level=%s debug=%s root_effective=%s "
            "dfx_effective=%s dfx_debug_enabled=%s handlers_ascend=%d handlers_root=%d",
            level_key,
            debug_list,
            logging.getLevelName(ascend.getEffectiveLevel()),
            logging.getLevelName(probe.getEffectiveLevel()),
            probe.isEnabledFor(logging.DEBUG),
            len(ascend.handlers),
            len(logging.root.handlers),
        )
        if needs_debug and probe.isEnabledFor(logging.DEBUG):
            # Canary: if this never appears, an outer INFO filter is still dropping DEBUG.
            probe.debug("[ascend_log] debug canary from vllm_ascend.dfx")
        # Compare handler chains right after Ascend configure (vs later DFX emit).
        log_logger_chain_probe(f"apply_ascend_log level={level_key}")


def _use_color() -> bool:
    """Determine if colored output should be used."""
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


def _is_ascend_module(pathname: str) -> bool:
    if not pathname:
        return False
    return "vllm_ascend" in pathname.replace("\\", "/")


def _infer_module_name(pathname: str) -> str:
    """Infer module name from the file path of the log caller."""
    if not pathname:
        return "core"
    parts = pathname.replace("\\", "/").split("/")
    try:
        idx = parts.index("vllm_ascend")
        if idx + 1 >= len(parts):
            return "core"
        item = parts[idx + 1]
        if idx + 2 >= len(parts):
            return item[:-3] if item.endswith(".py") else item
        return item
    except ValueError:
        return "core"


def _format_with_ascend_prefix(self, record, super_format):
    if not _is_ascend_module(record.pathname):
        return super_format(record)
    module = _infer_module_name(record.pathname)
    if record.filename == module + ".py":
        prefix = "[vllm-ascend]"
    else:
        prefix = f"[vllm-ascend] [{module}]"
    orig_msg = record.msg
    orig_args = record.args
    try:
        record.msg = f"{prefix} - {record.getMessage()}"
        record.args = ()
        return super_format(record)
    finally:
        record.msg = orig_msg
        record.args = orig_args


class AscendFormatter(NewLineFormatter):
    """Extends NewLineFormatter with [vllm-ascend] prefix and module name."""

    def formatTime(self, record, datefmt=None):
        return _format_time_ms(self, record, datefmt)

    def format(self, record):
        return _format_with_ascend_prefix(self, record, super().format)


class AscendColoredFormatter(ColoredFormatter):
    """Extends ColoredFormatter with [vllm-ascend] prefix and module name."""

    def formatTime(self, record, datefmt=None):
        return _format_time_ms(self, record, datefmt)

    def format(self, record):
        return _format_with_ascend_prefix(self, record, super().format)


class RotatingAscendFileHandler(logging.FileHandler):
    """FileHandler that rotates log files when they exceed a size limit.

    Naming convention:
        vllm_ascend_{timestamp}_{pid}.log          <- first file
        vllm_ascend_{timestamp}_{pid}_002.log       <- second file
        vllm_ascend_{timestamp}_{pid}_003.log       <- third file
    """

    def __init__(self, log_dir: str, max_bytes: int = _LOG_MAX_BYTES) -> None:
        self._log_dir = log_dir
        self._max_bytes = max_bytes
        self._sequence = 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._base_name = f"vllm_ascend_{timestamp}_{os.getpid()}"
        log_file = os.path.join(log_dir, f"{self._base_name}.log")
        super().__init__(log_file, encoding="utf-8")

    def emit(self, record) -> None:
        try:
            if self.stream is not None and os.path.isfile(self.baseFilename):
                if os.path.getsize(self.baseFilename) >= self._max_bytes:
                    self._rotate()
        except OSError:
            pass
        super().emit(record)

    def _rotate(self) -> None:
        self.stream.close()
        self.stream = None  # type: ignore[assignment]
        self._sequence += 1
        new_file = os.path.join(self._log_dir, f"{self._base_name}_{self._sequence:03d}.log")
        self.baseFilename = new_file
        self.stream = self._open()


_file_logging_configured = False
_file_handler: logging.Handler | None = None


def _setup_file_logging(log_dir: str | None = None) -> None:
    global _file_logging_configured, _file_handler
    if _file_logging_configured:
        return
    target_dir = log_dir or _LOG_DIR
    os.makedirs(target_dir, exist_ok=True)
    file_handler = RotatingAscendFileHandler(target_dir)
    vllm_logger = logging.getLogger("vllm")
    ascend_logger = logging.getLogger("vllm_ascend")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    vllm_logger.addHandler(file_handler)
    ascend_logger.addHandler(file_handler)
    _file_handler = file_handler
    _file_logging_configured = True


def configure_ascend_file_logging() -> None:
    global _file_logging_configured, _file_handler
    log_dir = _LOG_DIR
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        ascend_config = get_ascend_config()
        log_dir = ascend_config.ascend_log_path
    except Exception:
        pass
    if log_dir != _LOG_DIR:
        vllm_logger = logging.getLogger("vllm")
        ascend_logger = logging.getLogger("vllm_ascend")
        if _file_handler is not None:
            vllm_logger.removeHandler(_file_handler)
            ascend_logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        _file_logging_configured = False
    _setup_file_logging(log_dir)


def configure_ascend_logging() -> None:
    """Configure vllm_ascend logger with Ascend formatters.

    Creates a dedicated handler for the vllm_ascend logger namespace,
    avoiding any modification to vLLM's global logging state.
    This approach is safe for upstream tests and multiprocessing.
    """
    ascend_logger = logging.getLogger("vllm_ascend")
    if ascend_logger.handlers:
        return

    # Parse stream parameter
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
        stream = sys.stdout
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
        stream = sys.stderr
    else:
        stream = sys.stderr

    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)

    if _use_color():
        handler.setFormatter(AscendColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    else:
        handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))

    ascend_logger.addHandler(handler)
    ascend_logger.setLevel(envs.VLLM_LOGGING_LEVEL)
    ascend_logger.propagate = False

    # Package logger levels are owned by ``apply_ascend_log_level``
    # (driven by DFX ``ascend_log``); do not force DEBUG here.
