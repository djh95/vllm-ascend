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

from vllm_ascend.runtime_config._dist import SYNC_BROADCAST


_DEFAULTS: dict[str, Any] = {
    # broadcast: EngineCore leader reads JSON, in-DP broadcast (or file poll);
    # file: each rank polls the path (shared FS / per-node copy).
    "sync_mode": SYNC_BROADCAST,
    # Kept in JSON for visibility; effective hot-reload interval is set at
    # process start via additional_config.runtime_config_reload_interval (default 0).
    # Set >0 at startup to enable. JSON field alone cannot re-enable after start.
    "reload_interval_seconds": 0,
    "dump": {
        # Auto dump (detector anomaly arm): quota >0 enables; mutually exclusive
        # with manual_dump. dump.enabled is derived at runtime (auto || manual).
        "auto_max_times": 0,
        "auto_cooldown_seconds": 5 * 60,
        # Manual dump: false/0=off; true=continuous until hot-reload false;
        # positive int N = next N execute_model waves with scheduled_tokens>0.
        # Needs runtime_config_reload_interval>0. Skips auto quota/cooldown/filters.
        "manual_dump": False,
        "dump_all_blocks": False,
        # KV dump landing root (default derived: <report_dir>/kv_cache).
        # ``<incident_type>/<req_id>/`` is created under it per incident.
        # Settable at startup (additional_config.runtime_dump_dir) and via
        # this JSON key (hot-reload); startup value seeds JSON when unset.
        "dump_dir": None,
    },
    "ascend_log": {
        "level": "INFO",
        # Relative module paths under vllm_ascend forced to DEBUG, e.g. ["dfx"].
        "debug": [],
        # Per-logger overrides, e.g. {"vllm.worker": "WARNING", "dfx": "DEBUG"}.
        "modules": {},
    },
    # Ops logging switches (not persisted into anomaly report JSON files).
    "log": {
        # Log [SamplingMeta] for the anomalous req (TP0 + last PP only).
        "print_sampling_meta": False,
        # When a request finishes: log output_token_ids + decoded text (TP0 only).
        # Applies to every finished request. Accumulate only while true
        # (no backfill); mid-request enable may be partial or empty.
        "print_output_on_finish": False,
    },
    "report": {
        # Default False: anomaly reports store lengths only.
        # Set true to persist prompt_token_ids + cumulative output_token_ids.
        "save_sensitive_info": False,
        # When save_sensitive_info, decode prompt/output ids to text (lazy tokenizer).
        "decode_token_ids": True,
        # Cap persisted token-id list lengths (0 = unlimited). Counts stay full.
        "max_prompt_token_ids": 1000,
        "max_output_token_ids": 1000,
        # Persist each request's current GPU block_ids in report detail.
        "include_block_ids": True,
        # D2H this wave's real paged-attention slot_mapping slice (default off).
        "include_slot_mapping": False,
        # Track/report per-block slot meta state (see blocks[]). Does not
        # store per-slot token ids — only state / next_offset / source.
        "block_state": False,
        # Edge audit: hook reshape_and_cache / zero / offload H2D into
        # KvBlockMetaTracker. Also auto-on when slot_consistency /
        # kv_slot_order or report.block_state is enabled.
        "kv_audit": False,
        # When kv_audit (explicit or auto): log error if dummy/pad slot_mapping
        # still has >=0 ids.
        "kv_audit_pad_check": True,
    },
    # Per-detector nested sections. Each has ``enabled`` (default false).
    "actions": {
        "defaults": {
            "on_trigger": ["report"],
        },
    },
    "detector": {
        # Shared detect behavior (not a detector section): keep detecting a
        # request on every step, but once an anomaly is found for it, stop
        # detecting that request (prevents endless reports for the same req).
        "stop_after_alert": True,
        "spec_acceptance": {
            "enabled": False,
            # Where this detector runs: auto/leader/any/all/external
            # (see detector/placement.py; "auto" = planner decides).
            "exec_scope": "auto",
            "window": 10,
            "low_threshold": 0.3,
            "len_low_threshold": 1.4,
            "high_threshold": 0.96,
            "len_high_threshold": 2.8,
        },
        "token_logprob": {
            "enabled": False,
            "exec_scope": "auto",
            "window": 64,
            "stride": 32,
            "topk": 20,
            "ill_nan_window_thresh": 1,
            "ill_rare_window_thresh": 1,
            "ill_garbled_window_thresh": 1,
            "ill_repet_window_thresh": 2,
        },
        "output_substring": {
            "enabled": False,
            "exec_scope": "auto",
            "patterns": [],
            "add_special_tokens": False,
            # true: patterns match only at the start (prefix) of cumulative output;
            # false (default): match anywhere as a contiguous token-id subsequence.
            "match_prefix": False,
        },
        # Sliding-window token re-read detector (no logprobs). Per new token:
        # score = count of that id in the previous ``window`` content tokens;
        # alert when sum of the last ``window`` scores exceeds threshold.
        "token_repeat": {
            "enabled": False,
            "exec_scope": "auto",
            "window": 32,
            "repeat_sum_threshold": 64,
            # Require this many content tokens before alerting (0 = no warmup).
            "min_tokens": 32,
            # Require this many consecutive over-threshold steps.
            "consecutive_hits": 1,
            # Token ids skipped for the content window (e.g. punctuation fillers).
            "ignore_token_ids": [],
        },
        # Lifecycle: write one report when a request is reaped (finished +
        # sample-wave drained). Non-ill; does not consume dump quota.
        "finish": {
            "enabled": False,
            "exec_scope": "auto",
            "on_trigger": ["report"],
        },
    },
    # Soft-assert invariants (not LPT detectors). See runtime_guard.invariant.
    # check_scope: auto|leader|all — who runs check (and thus who writes report;
    # no cross-rank Incident shipping). auto → leader, except slot_consistency /
    # kv_slot_order under CP/DP → all.
    "invariant": {
        # Slot meta token vs inference sequence (incident_type=kv_slot_token).
        "slot_consistency": {
            "enabled": False,
            "check_scope": "auto",
            "on_trigger": ["report"],
        },
        # Sequential slot-offset order within a block (incident_type=kv_slot_order).
        "kv_slot_order": {
            "enabled": False,
            "check_scope": "auto",
            "on_trigger": ["report"],
        },
        # BLOCK_SEALED + non-zero-offset slot rewrite (incident_type=kv_state).
        "kv_state": {
            "enabled": False,
            "check_scope": "auto",
            "on_trigger": ["report"],
        },
        "logits_finite": {
            "enabled": False,
            "check_scope": "auto",
            "on_trigger": ["report"],
        },
    },
    # Detector→rank placement (ExecScope scheduling; detector/placement.py).
    # Detection no longer pins to TP0: logits are all-gathered per rank,
    # sampling is redundant, and scheduler metadata is TP-replicated, so
    # detectors spread across TP ranks to avoid a rank-0 hot-path bottleneck.
    "detector_placement": {
        # "auto": LPT load-balance enabled detectors across TP ranks.
        # "manual": detector_placement.manual entries win for ANY detectors.
        "mode": "auto",
        # incident_type -> tp_rank (honored in manual mode; out-of-range
        # entries are ignored by the planner).
        "manual": {},
        # Keep current assignments on re-plan; place only newly enabled ones.
        "pin": False,
    },
    # Detect-time InputFilterManager (+ one-shot prompt print for authoring).
    "input_filter": {
        # [] = no filter. Use type input_token_id_prefix for prefix matching.
        "filters": [],
        # One-shot: next real execute_model with requests logs prompt token ids
        # and length, then cleared to false. Needs reload_interval > 0.
        "print_input_token_ids_once": False,
    },
}


# Known nested detector sections under ``detector`` (anomaly detectors only).
DETECTOR_SECTIONS: tuple[str, ...] = (
    "spec_acceptance",
    "token_logprob",
    "output_substring",
    "token_repeat",
    "finish",
)
# Soft-assert sections under ``invariant`` (single source of truth).
INVARIANT_SECTIONS: tuple[str, ...] = (
    "slot_consistency",
    "kv_slot_order",
    "kv_state",
    "logits_finite",
)
# Allowed keys under ``dump`` / ``log`` / ``report``.
DUMP_KEYS: frozenset[str] = frozenset(_DEFAULTS["dump"])
LOG_KEYS: frozenset[str] = frozenset(_DEFAULTS["log"])
REPORT_KEYS: frozenset[str] = frozenset(_DEFAULTS["report"])
# Allowed keys per detector / invariant section.
DETECTOR_KEYS: dict[str, frozenset[str]] = {
    name: frozenset(sec) for name, sec in _DEFAULTS["detector"].items() if isinstance(sec, dict)
}
INVARIANT_KEYS: dict[str, frozenset[str]] = {
    name: frozenset(sec) for name, sec in _DEFAULTS["invariant"].items() if isinstance(sec, dict)
}

