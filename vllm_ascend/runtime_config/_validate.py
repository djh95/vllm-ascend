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

"""Validate / coerce runtime_config JSON payloads in place."""

from __future__ import annotations

from typing import Any

from vllm_ascend.runtime_config._defaults import (
    DETECTOR_KEYS,
    DETECTOR_SECTIONS,
    DUMP_KEYS,
    LOG_KEYS,
    REPORT_KEYS,
    _DEFAULTS,
)
from vllm_ascend.runtime_config._dist import SYNC_BROADCAST, SYNC_FILE
from vllm_ascend.runtime_config._merge import _normalize_config_sections_into


def int_field(value: Any, field: str, *, min_value: int | None = None) -> int:
    # C3: reject None/str/NaN and silently-truncated floats (2.7 → 2) with
    # an error that names the offending field.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"{field} must be a number, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    iv = int(value)
    if min_value is not None and iv < min_value:
        raise ValueError(f"{field} must be >= {min_value}, got {iv}")
    return iv


def float_field(
    value: Any,
    field: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"{field} must be a number, got {value!r}")
    fv = float(value)
    if min_value is not None and fv < min_value:
        raise ValueError(f"{field} must be >= {min_value}, got {fv}")
    if max_value is not None and fv > max_value:
        raise ValueError(f"{field} must be <= {max_value}, got {fv}")
    return fv


def validate_dump_mutual_exclusive(dump: dict[str, Any]) -> None:
    try:
        auto_on = int(dump.get("auto_max_times", 0)) > 0
    except (TypeError, ValueError):
        auto_on = False
    manual_raw = dump.get("manual_dump", False)
    manual_on = manual_raw not in (False, 0)
    if auto_on and manual_on:
        raise ValueError(
            "dump.auto_max_times>0 and dump.manual_dump active are mutually exclusive"
        )


def validate_runtime_config(data: dict[str, Any]) -> None:
    """Validate / normalize ``data`` in place.

    Detect and dump are orthogonal: dump-only / detect-only / both are valid.
    Soft warnings for easy-to-misread combos live on ``RuntimeConfig``.

    S10 fix: defensively re-run ``_normalize_config_sections`` at entry so
    validation is safe regardless of whether the caller normalized first.
    """
    _normalize_config_sections_into(data)
    for section in (
        "dump",
        "ascend_log",
        "log",
        "detector",
        "input_filter",
        "report",
    ):
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"dfx config missing object section '{section}'")
    interval = data.get("reload_interval_seconds", 0)
    if not isinstance(interval, (int, float)) or interval < 0:
        raise ValueError(f"reload_interval_seconds must be >= 0, got {interval}")
    sync_mode = str(data.get("sync_mode", SYNC_BROADCAST)).lower()
    if sync_mode not in (SYNC_BROADCAST, SYNC_FILE):
        raise ValueError(f"sync_mode must be '{SYNC_BROADCAST}' or '{SYNC_FILE}'")
    unknown_dump = sorted(set(data["dump"]) - DUMP_KEYS)
    # Retired msprobe-dump bridge keys; ignore in older JSON (native dump_kv only).
    for retired in ("reload_msprobe", "msprobe_config_path"):
        if retired in data["dump"]:
            data["dump"].pop(retired, None)
            unknown_dump = [k for k in unknown_dump if k != retired]
    if unknown_dump:
        raise ValueError(f"dump has unknown key(s) {unknown_dump}; allowed={sorted(DUMP_KEYS)}")
    auto_max_times = data["dump"].get("auto_max_times", 0)
    data["dump"]["auto_max_times"] = int_field(
        auto_max_times, "dump.auto_max_times", min_value=0
    )
    auto_cd = data["dump"].get("auto_cooldown_seconds", 300)
    data["dump"]["auto_cooldown_seconds"] = int_field(
        auto_cd, "dump.auto_cooldown_seconds", min_value=0
    )
    manual_dump = data["dump"].get("manual_dump")
    if manual_dump is not None and not isinstance(manual_dump, bool):
        if isinstance(manual_dump, int) and not isinstance(manual_dump, bool):
            if manual_dump < 0:
                raise ValueError("dump.manual_dump must be >= 0")
            if manual_dump == 0:
                data["dump"]["manual_dump"] = False
        else:
            raise ValueError("dump.manual_dump must be bool or non-negative int")
    dump_dir_raw = data["dump"].get("dump_dir")
    if dump_dir_raw is not None and not isinstance(dump_dir_raw, str):
        raise ValueError("dump.dump_dir must be a string path or null")
    validate_dump_mutual_exclusive(data["dump"])
    unknown_log = sorted(set(data["log"]) - LOG_KEYS)
    if unknown_log:
        raise ValueError(f"log has unknown key(s) {unknown_log}; allowed={sorted(LOG_KEYS)}")
    unknown_report = sorted(set(data["report"]) - REPORT_KEYS)
    if unknown_report:
        raise ValueError(
            f"report has unknown key(s) {unknown_report}; allowed={sorted(REPORT_KEYS)}"
        )
    save_sensitive = data["report"].get("save_sensitive_info")
    if save_sensitive is not None and not isinstance(save_sensitive, bool):
        if save_sensitive in (0, 1):
            data["report"]["save_sensitive_info"] = bool(save_sensitive)
        else:
            raise ValueError("report.save_sensitive_info must be bool")
    for log_key in ("print_sampling_meta", "print_output_on_finish"):
        log_val = data["log"].get(log_key)
        if log_val is not None and not isinstance(log_val, bool):
            if log_val in (0, 1):
                data["log"][log_key] = bool(log_val)
            else:
                raise ValueError(f"log.{log_key} must be bool")
    decode_ids = data["report"].get("decode_token_ids")
    if decode_ids is not None and not isinstance(decode_ids, bool):
        if decode_ids in (0, 1):
            data["report"]["decode_token_ids"] = bool(decode_ids)
        else:
            raise ValueError("report.decode_token_ids must be bool")
    for max_key in ("max_prompt_token_ids", "max_output_token_ids"):
        max_val = data["report"].get(max_key)
        if max_val is None:
            continue
        if isinstance(max_val, bool) or not isinstance(max_val, (int, float)):
            raise ValueError(f"report.{max_key} must be an int >= 0")
        if int(max_val) < 0:
            raise ValueError(f"report.{max_key} must be >= 0")
        data["report"][max_key] = int(max_val)
    for block_key in (
        "include_block_ids",
        "include_slot_mapping",
        "block_last_write_wave",
        "block_last_writer",
        "slot_last_write",
    ):
        block_val = data["report"].get(block_key)
        if block_val is not None and not isinstance(block_val, bool):
            if block_val in (0, 1):
                data["report"][block_key] = bool(block_val)
            else:
                raise ValueError(f"report.{block_key} must be bool")
    print_once = data["input_filter"].get("print_input_token_ids_once")
    if print_once is not None and not isinstance(print_once, bool):
        if print_once in (0, 1):
            data["input_filter"]["print_input_token_ids_once"] = bool(print_once)
        else:
            raise ValueError("input_filter.print_input_token_ids_once must be bool")
    from vllm_ascend.runtime_guard.input_filters import normalize_input_filter_configs

    data["input_filter"]["filters"] = normalize_input_filter_configs(data["input_filter"].get("filters", []))
    level = data["ascend_log"].get("level", "INFO")
    if not isinstance(level, str):
        raise ValueError("ascend_log.level must be str")
    debug = data["ascend_log"].get("debug", [])
    if not isinstance(debug, list):
        raise ValueError("ascend_log.debug must be a list of module name strings")
    for item in debug:
        if not isinstance(item, (str, int, float)):
            raise ValueError("ascend_log.debug entries must be strings")
    modules = data["ascend_log"].get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("ascend_log.modules must be an object")
    for key, val in modules.items():
        if not isinstance(key, str):
            raise ValueError("ascend_log.modules keys must be strings")
        if not isinstance(val, str):
            raise ValueError("ascend_log.modules values must be strings")
    detector = data["detector"]
    known = set(DETECTOR_SECTIONS)
    for key, value in detector.items():
        if key == "stop_after_alert":
            if not isinstance(value, bool):
                if value in (0, 1):
                    detector["stop_after_alert"] = bool(value)
                else:
                    raise ValueError("detector.stop_after_alert must be bool")
            continue
        if key not in known:
            raise ValueError(
                f"detector.{key} is not a known detector section; "
                f"expected nested objects among {sorted(known)} "
                f"(e.g. detector.spec_acceptance.enabled)"
            )
        if not isinstance(value, dict):
            raise ValueError(f"detector.{key} must be an object")
        unknown_sub = sorted(set(value) - DETECTOR_KEYS[key])
        if unknown_sub:
            raise ValueError(
                f"detector.{key} has unknown key(s) {unknown_sub}; "
                f"allowed={sorted(DETECTOR_KEYS[key])}"
            )
        scope = value.get("exec_scope", "auto")
        if scope not in ("auto", "leader", "any", "all", "external"):
            raise ValueError(
                f"detector.{key}.exec_scope must be one of "
                f"auto/leader/any/all/external; got {scope!r}"
            )
    for name in DETECTOR_SECTIONS:
        sec = detector.setdefault(name, {})
        if not isinstance(sec, dict):
            raise ValueError(f"detector.{name} must be an object")
        enabled = sec.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            if enabled in (0, 1):
                sec["enabled"] = bool(enabled)
            else:
                raise ValueError(f"detector.{name}.enabled must be bool")

    token = detector["token_logprob"]
    token["window"] = int_field(
        token.get("window", 64), "detector.token_logprob.window", min_value=1
    )
    token["stride"] = int_field(
        token.get("stride", 32), "detector.token_logprob.stride", min_value=1
    )
    if token["window"] < token["stride"]:
        raise ValueError("detector.token_logprob.window must be >= detector.token_logprob.stride")

    from vllm_ascend.runtime_guard.detector.output_substring import normalize_raw_patterns

    out_sub = detector["output_substring"]
    out_sub["patterns"] = normalize_raw_patterns(out_sub.get("patterns", []))
    add_special = out_sub.get("add_special_tokens")
    if add_special is not None and not isinstance(add_special, bool):
        if add_special in (0, 1):
            out_sub["add_special_tokens"] = bool(add_special)
        else:
            raise ValueError("detector.output_substring.add_special_tokens must be bool")
    match_prefix = out_sub.get("match_prefix")
    if match_prefix is not None and not isinstance(match_prefix, bool):
        if match_prefix in (0, 1):
            out_sub["match_prefix"] = bool(match_prefix)
        else:
            raise ValueError("detector.output_substring.match_prefix must be bool")

    from vllm_ascend.runtime_guard.detector.token_repeat import normalize_ignore_token_ids

    token_repeat = detector["token_repeat"]
    token_repeat["window"] = int_field(
        token_repeat.get("window", 32), "detector.token_repeat.window", min_value=1
    )
    token_repeat["repeat_sum_threshold"] = int_field(
        token_repeat.get("repeat_sum_threshold", 64),
        "detector.token_repeat.repeat_sum_threshold",
        min_value=0,
    )
    token_repeat["min_tokens"] = int_field(
        token_repeat.get("min_tokens", token_repeat["window"]),
        "detector.token_repeat.min_tokens",
        min_value=0,
    )
    token_repeat["consecutive_hits"] = int_field(
        token_repeat.get("consecutive_hits", 1),
        "detector.token_repeat.consecutive_hits",
        min_value=1,
    )
    token_repeat["ignore_token_ids"] = normalize_ignore_token_ids(token_repeat.get("ignore_token_ids", []))

    spec = detector["spec_acceptance"]
    spec["window"] = int_field(
        spec.get("window", 10), "detector.spec_acceptance.window", min_value=1
    )
    for rate_key in ("low_threshold", "high_threshold"):
        spec[rate_key] = float_field(
            spec.get(rate_key, _DEFAULTS["detector"]["spec_acceptance"][rate_key]),
            f"detector.spec_acceptance.{rate_key}",
            min_value=0.0,
            max_value=1.0,
        )
    for len_key in ("len_low_threshold", "len_high_threshold"):
        spec[len_key] = float_field(
            spec.get(len_key, _DEFAULTS["detector"]["spec_acceptance"][len_key]),
            f"detector.spec_acceptance.{len_key}",
            min_value=0.0,
        )

    placement = data.setdefault("detector_placement", dict(_DEFAULTS["detector_placement"]))
    if not isinstance(placement, dict):
        raise ValueError("detector_placement must be an object")
    mode = placement.get("mode", "auto")
    if mode not in ("auto", "manual"):
        raise ValueError(f"detector_placement.mode must be 'auto' or 'manual'; got {mode!r}")
    manual_map = placement.get("manual", {})
    if not isinstance(manual_map, dict):
        raise ValueError("detector_placement.manual must be an object")
    known_types = set(DETECTOR_SECTIONS)
    for mkey, mval in manual_map.items():
        if not isinstance(mkey, str):
            raise ValueError("detector_placement.manual keys must be strings")
        if mkey not in known_types:
            raise ValueError(
                f"detector_placement.manual has unknown detector {mkey!r}; "
                f"expected {sorted(known_types)}"
            )
        if isinstance(mval, bool) or not isinstance(mval, int) or mval < 0:
            raise ValueError(
                f"detector_placement.manual.{mkey} must be a non-negative int (tp_rank)"
            )
    if not isinstance(placement.get("pin", False), bool):
        raise ValueError("detector_placement.pin must be bool")
