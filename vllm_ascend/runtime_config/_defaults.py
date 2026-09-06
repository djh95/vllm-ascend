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
        # Track/report last write wave per physical block (see blocks[]).
        "block_last_write_wave": False,
        # Track/report last writer req_id per physical block (see blocks[]).
        "block_last_writer": False,
        # Track/report per-slot last write: writer req_id, wave, ts and the
        # token id whose KV was written at that slot (see slots[]).
        "slot_last_write": False,
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
        # KV block write integrity (uses KvBlockMetaTracker; no msprobe).
        "block_kv": {
            "enabled": False,
            "exec_scope": "auto",
            "check_wave_regression": True,
            "check_same_wave_writer": True,
        },
        # Slot-token consistency: the token id recorded at each of the
        # request's block slots (write-time meta) must equal the token at the
        # same position in the request's inference sequence (prompt+output).
        # Catches wrong-block / stale-reuse / cross-request KV contamination
        # (metadata level; actual tensor verification stays offline in
        # verify_request_kv.py).
        "slot_consistency": {
            "enabled": False,
            "exec_scope": "auto",
            # "first": full prefix check once per request at its first note
            #          step (covers prefix-cache imported slots).
            # "step":  recheck the whole prefix every step — strongest, but
            #          O(prefix_len) per request per step; debug only.
            "mode": "first",
        },
        # 1-D position_ids alignment for newly scheduled tokens (no msprobe).
        "position_alignment": {
            "enabled": False,
            "exec_scope": "auto",
        },
        # Pre-sample logits NaN/Inf on sampling rows (no msprobe; ill_type=nan).
        "logits_finite": {
            "enabled": False,
            "exec_scope": "auto",
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


# Known nested detector sections under ``detector``.
DETECTOR_SECTIONS: tuple[str, ...] = (
    "spec_acceptance",
    "token_logprob",
    "output_substring",
    "token_repeat",
    "block_kv",
    "slot_consistency",
    "position_alignment",
    "logits_finite",
)
# Allowed keys under ``dump`` / ``log`` / ``report``.
DUMP_KEYS: frozenset[str] = frozenset(_DEFAULTS["dump"])
LOG_KEYS: frozenset[str] = frozenset(_DEFAULTS["log"])
REPORT_KEYS: frozenset[str] = frozenset(_DEFAULTS["report"])
# Allowed keys per detector section.
DETECTOR_KEYS: dict[str, frozenset[str]] = {
    name: frozenset(sec) for name, sec in _DEFAULTS["detector"].items() if isinstance(sec, dict)
}

