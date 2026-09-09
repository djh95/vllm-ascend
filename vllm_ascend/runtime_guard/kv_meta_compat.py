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

"""KV meta (L1/L2) compatibility gate.

Unsupported layouts (sparse / Mamba·hybrid / sliding window) force-disable
block-slot meta consumers and log once. Prefix / PD / offload stay allowed for
L1; when L2 is also requested we log that load paths may leave slots
unverified. Further adaptations are tracked in ``KV_META_TODO.md``.
"""

from __future__ import annotations

from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# Soft-assert sections that share the PA block-slot ledger (not logits_finite).
_KV_META_INVARIANT_SECTIONS: tuple[str, ...] = (
    "slot_consistency",
    "kv_slot_order",
    "kv_state",
)

_blocked_reasons: tuple[str, ...] = ()
_last_block_key: str | None = None
_last_load_advisory_key: str | None = None


def reset_for_tests() -> None:
    """Clear process-local gate state (unit tests)."""
    global _blocked_reasons, _last_block_key, _last_load_advisory_key
    _blocked_reasons = ()
    _last_block_key = None
    _last_load_advisory_key = None


def is_kv_meta_blocked() -> bool:
    """True when KV block/slot meta must stay off for this process."""
    return bool(_blocked_reasons)


def blocked_reasons() -> tuple[str, ...]:
    return _blocked_reasons


def probe_incompatible_features(runner: Any) -> list[str]:
    """Return human-readable reasons KV meta cannot run with this runner."""
    reasons: list[str] = []
    if runner is None:
        return reasons

    ascend = getattr(runner, "ascend_config", None)
    sparse_cfg = getattr(ascend, "sparse_kv_offload_config", None) if ascend is not None else None
    if sparse_cfg is not None and bool(getattr(sparse_cfg, "enabled", False)):
        reasons.append("sparse_kv_offload")

    vllm_config = getattr(runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
    cache_config = getattr(vllm_config, "cache_config", None) if vllm_config is not None else None

    if model_config is not None and bool(getattr(model_config, "is_hybrid", False)):
        reasons.append("hybrid_mamba_model")

    mamba_mode = str(getattr(cache_config, "mamba_cache_mode", "none") or "none").lower()
    if mamba_mode not in ("", "none"):
        reasons.append(f"mamba_cache_mode={mamba_mode}")

    if _model_has_sliding_window(model_config):
        reasons.append("sliding_window")

    # Dedupe while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def probe_external_load_features(runner: Any) -> list[str]:
    """Features that inject whole blocks (L1 OK; L2 often unverified)."""
    tags: list[str] = []
    if runner is None:
        return tags
    vllm_config = getattr(runner, "vllm_config", None)
    if vllm_config is None:
        return tags
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and bool(getattr(cache_config, "enable_prefix_caching", False)):
        tags.append("prefix_caching")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        tags.append("kv_transfer")
    # Native / simple CPU offload often rides kv_transfer_config; also check
    # Ascend additional knobs when present.
    ascend = getattr(runner, "ascend_config", None)
    for attr in ("kv_offload_config", "recompute_cpu_offload_config"):
        cfg = getattr(ascend, attr, None) if ascend is not None else None
        if cfg is not None and bool(getattr(cfg, "enabled", False)):
            tags.append(attr.replace("_config", ""))
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _model_has_sliding_window(model_config: Any) -> bool:
    if model_config is None:
        return False
    getter = getattr(model_config, "get_sliding_window", None)
    if callable(getter):
        try:
            sw = getter()
            if sw:
                return True
        except Exception:
            pass
    if getattr(model_config, "sliding_window", None):
        return True
    hf = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
    if hf is not None and getattr(hf, "sliding_window", None):
        return True
    return False


def _force_disable_kv_meta_sections(cfg: Any) -> list[str]:
    """Flip in-memory JSON so KV meta consumers read as disabled. Return flipped keys."""
    flipped: list[str] = []
    data = getattr(cfg, "_data", None)
    if not isinstance(data, dict):
        return flipped
    inv = data.setdefault("invariant", {})
    if not isinstance(inv, dict):
        return flipped
    for name in _KV_META_INVARIANT_SECTIONS:
        sec = inv.get(name)
        if isinstance(sec, dict) and bool(sec.get("enabled", False)):
            sec["enabled"] = False
            flipped.append(f"invariant.{name}")
    report = data.setdefault("report", {})
    if isinstance(report, dict):
        if bool(report.get("kv_audit", False)):
            report["kv_audit"] = False
            flipped.append("report.kv_audit")
        if bool(report.get("block_state", False)):
            report["block_state"] = False
            flipped.append("report.block_state")
    return flipped


def _wants_l2_slot_check(cfg: Any) -> bool:
    if cfg is None:
        return False
    for name in ("slot_consistency", "kv_slot_order"):
        if bool(cfg.invariant_get(name, "enabled", False)):
            return True
    return False


def apply_kv_meta_compat(runner: Any, cfg: Any) -> bool:
    """Apply refuse gate + load-path advisory. Return True if meta is blocked.

    Call from ``_sync_kv_audit`` / bind so hot-reload cannot re-enable against
    an incompatible runner.
    """
    global _blocked_reasons, _last_block_key, _last_load_advisory_key

    reasons = probe_incompatible_features(runner)
    _blocked_reasons = tuple(reasons)

    if reasons:
        # Always force-disable so a later JSON enable cannot stick while blocked.
        _force_disable_kv_meta_sections(cfg)
        key = ",".join(reasons)
        if key != _last_block_key:
            _last_block_key = key
            logger.warning(
                "[runtime_guard] KV meta (kv_audit / slot_consistency / "
                "kv_slot_order / kv_state / block_state) disabled — incompatible "
                "with %s. logits_finite and non-KV detectors are unchanged. "
                "See runtime_guard/KV_META_TODO.md for adaptation backlog.",
                ", ".join(reasons),
            )
        return True

    _last_block_key = None
    load_tags = probe_external_load_features(runner)
    if load_tags and _wants_l2_slot_check(cfg):
        key = ",".join(load_tags)
        if key != _last_load_advisory_key:
            _last_load_advisory_key = key
            logger.info(
                "[runtime_guard] KV meta L1 (block state) is supported with %s; "
                "L2 slot-token checks may leave load-filled slots unverified "
                "until content-level adaptation lands (KV_META_TODO P0-4).",
                ", ".join(load_tags),
            )
    else:
        _last_load_advisory_key = None
    return False
