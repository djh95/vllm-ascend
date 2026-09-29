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

"""Default ``runtime_config.json`` schema (hot-reload control plane)."""

from __future__ import annotations

from typing import Any

# Startup-only / internal knobs (not exposed in runtime_config.json).
HOT_RELOAD_INTERVAL_SECONDS: float = 3.0
ACTION_QUEUE_MAX_SIZE: int = 64
DUMP_FREE_HEADROOM_BYTES: int = 5 * 1024 * 1024 * 1024
SPEC_SHORT_LOG_INTERVAL_SECONDS: float = 2.0
LOGITS_FINITE_DEFERRED_QUEUE_MAX: int = 256

# Retired JSON keys: silently dropped on validate so old on-disk configs still load.
_RETIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"reload_interval_seconds"})
_RETIRED_DUMP_KEYS: frozenset[str] = frozenset({"free_headroom_bytes"})
_RETIRED_REPORT_KEYS: frozenset[str] = frozenset({"decode_token_ids", "include_block_ids"})
_RETIRED_ACTIONS_KEYS: frozenset[str] = frozenset({"queue_max_size"})
_RETIRED_DETECTOR_SECTIONS: frozenset[str] = frozenset({"output_substring"})
_RETIRED_DETECTOR_KEYS: dict[str, frozenset[str]] = {
    "spec_acceptance": frozenset({"short_log_interval_seconds"}),
    "logits_finite": frozenset({"deferred_queue_max"}),
}

_DEFAULTS: dict[str, Any] = {
    # Hot-reload on/off is startup-only (additional_config.runtime_config_hot_reload);
    # poll period is an internal constant — not exposed in this JSON.
    "dump": {
        # Auto dump (detector anomaly arm): quota >0 enables; mutually exclusive
        # with manual_dump. dump.enabled is derived at runtime (auto || manual).
        "auto_max_times": 0,
        "auto_cooldown_seconds": 5 * 60,
        # Manual dump: false/0=off; positive int N = next N armed waves
        # (prefer N=1 — one shot is enough). true=continuous every wave until
        # hot-reload false (not recommended: little debug value, floods disk /
        # ActionQueue). Needs runtime_config_hot_reload=true. Skips auto
        # quota/cooldown/filters. Count is decremented in-memory each armed
        # wave; JSON is rewritten only when the count reaches 0 (false). While
        # the in-memory count is still >0, a hand-edit to this file still
        # hot-reloads into memory as usual. Multi-DP sharing one
        # runtime_config.json: each DP replica may dump up to N times (file
        # stays at N until some replica persists 0) — worst case about
        # num_DP × N dumps across the cluster.
        "manual_dump": False,
        # KV dump landing root (default derived: <report_dir>/kv_cache).
        # ``<incident_type>/<req_id>/`` is created under it per incident.
        # Settable at startup (additional_config.runtime_dump_dir) and via
        # this JSON key (hot-reload); startup value seeds JSON when unset.
        "dump_dir": None,
    },
    "ascend_log": {
        "level": "INFO",
        # Relative module paths under vllm_ascend forced to DEBUG, e.g. ["runtime_guard"].
        "debug": [],
        # Per-logger overrides, e.g. {"vllm.worker": "WARNING", "runtime_guard": "DEBUG"}.
        "modules": {},
    },
    "report": {
        # Default False: anomaly reports store lengths only.
        # Set true to persist prompt_token_ids + cumulative output_token_ids
        # (always decoded to text when sensitive is on).
        "save_sensitive_info": False,
        # Cap persisted token-id list lengths (0 = unlimited). Counts stay full.
        "max_prompt_token_ids": 1000,
        "max_output_token_ids": 1000,
        # Same (incident_type, req_id): max report files; at cap, stop detecting
        # that req (all detectors). Default 1 = one report then stop.
        # GPU block_ids are always included in report detail.
        "max_per_req": 1,
    },
    # Nested detector sections under ``detector`` (each has ``enabled``).
    "actions": {
        "defaults": {
            # Always includes report so max_per_req can stop-detect after writes.
            "on_trigger": ["report"],
        },
    },
    "detector": {
        # Stop-detect after report.max_per_req successful writes (see report).
        "spec_acceptance": {
            "enabled": False,
            "window": 10,
            "low_threshold": 0.3,
            "len_low_threshold": 1.4,
            "high_threshold": 0.96,
            "len_high_threshold": 2.8,
        },
        # Sliding-window token re-read detector (no logprobs). Per new token:
        # score = count of that id in the previous ``window`` content tokens;
        # alert when sum of the last ``window`` scores exceeds threshold.
        "token_repeat": {
            "enabled": False,
            "window": 32,
            "repeat_sum_threshold": 64,
            # Require this many content tokens before alerting (0 = no warmup).
            "min_tokens": 32,
            # Require this many consecutive over-threshold steps.
            "consecutive_hits": 1,
            # Token ids skipped for the content window (e.g. punctuation fillers).
            "ignore_token_ids": [],
        },
        # Pre-sample logits NaN/Inf on sampling rows (no msprobe; ill_type=nan).
        # Every step: device isfinite + one gate scalar. Default async .item()
        # (wait at after-sample); set item_sync=true to block at pre-sample.
        "logits_finite": {
            "enabled": False,
            # false: non-blocking D2H of the all-finite gate; wait in check_deferred.
            #   Logits may be mutated afterward (grammar bitmask → -inf); gate still
            #   uses precomputed row_finite, but hit-time finite_kind may see post-mutation
            #   values. true: blocking .item() (+ hit resolve) at pre-sample — correct
            #   pre-grammar logits / kind.
            "item_sync": False,
        },
    },
}


# Known nested detector sections under ``detector``.
DETECTOR_SECTIONS: tuple[str, ...] = (
    "spec_acceptance",
    "token_repeat",
    "logits_finite",
)
# Allowed top-level keys (typos like ``windw`` must fail validation loudly).
TOP_LEVEL_KEYS: frozenset[str] = frozenset(_DEFAULTS)
# Allowed keys under ``dump`` / ``report``.
DUMP_KEYS: frozenset[str] = frozenset(_DEFAULTS["dump"])
REPORT_KEYS: frozenset[str] = frozenset(_DEFAULTS["report"])
ASCEND_LOG_KEYS: frozenset[str] = frozenset(_DEFAULTS["ascend_log"])
ACTIONS_KEYS: frozenset[str] = frozenset(_DEFAULTS["actions"])
# Per-detector action overrides (not in each detector's default dict).
_DETECTOR_ACTION_KEYS: frozenset[str] = frozenset(
    {
        "on_trigger",
        "dump_kv",
        "report",
    }
)
# Allowed keys per detector section (params ∪ action overrides).
DETECTOR_KEYS: dict[str, frozenset[str]] = {
    name: frozenset(sec) | _DETECTOR_ACTION_KEYS for name, sec in _DEFAULTS["detector"].items() if isinstance(sec, dict)
}
# Control-plane section for incident_type=manual_trigger (not a detector).
MANUAL_TRIGGER_SECTION_KEYS: frozenset[str] = frozenset(_DETECTOR_ACTION_KEYS)
