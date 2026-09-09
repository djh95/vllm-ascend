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

Any layout that still needs adaptation (see ``KV_META_TODO.md``) force-disables
block-slot meta consumers and logs a warning. Dense local paged-attention
reshape_and_cache paths remain the only supported shape for now.
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


def reset_for_tests() -> None:
    """Clear process-local gate state (unit tests)."""
    global _blocked_reasons, _last_block_key
    _blocked_reasons = ()
    _last_block_key = None


def is_kv_meta_blocked() -> bool:
    """True when KV block/slot meta must stay off for this process."""
    return bool(_blocked_reasons)


def blocked_reasons() -> tuple[str, ...]:
    return _blocked_reasons


def probe_incompatible_features(runner: Any) -> list[str]:
    """Return reasons KV meta cannot run until adaptation lands."""
    reasons: list[str] = []
    if runner is None:
        return reasons

    ascend = getattr(runner, "ascend_config", None)
    sparse_cfg = getattr(ascend, "sparse_kv_offload_config", None) if ascend is not None else None
    if sparse_cfg is not None and bool(getattr(sparse_cfg, "enabled", False)):
        reasons.append("sparse_kv_offload")

    if ascend is not None:
        if bool(getattr(ascend, "enable_sparse_sfa_c8", False)):
            reasons.append("enable_sparse_sfa_c8")
        if bool(getattr(ascend, "enable_sparse_li_c8", False)):
            reasons.append("enable_sparse_li_c8")
        if bool(getattr(ascend, "enable_dsa_cp", False)):
            reasons.append("enable_dsa_cp")
        for attr in ("kv_offload_config", "recompute_cpu_offload_config"):
            cfg = getattr(ascend, attr, None)
            if cfg is not None and bool(getattr(cfg, "enabled", False)):
                reasons.append(attr.replace("_config", ""))

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

    if cache_config is not None and bool(getattr(cache_config, "enable_prefix_caching", False)):
        reasons.append("prefix_caching")

    if vllm_config is not None and getattr(vllm_config, "kv_transfer_config", None) is not None:
        reasons.append("kv_transfer")
        connector = _kv_connector_name(getattr(vllm_config, "kv_transfer_config", None))
        if connector:
            reasons.append(f"kv_connector={connector}")

    # Dedupe while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def probe_external_load_features(runner: Any) -> list[str]:
    """Deprecated alias: external-load features are refuse-gated now."""
    return [
        r
        for r in probe_incompatible_features(runner)
        if r
        in (
            "prefix_caching",
            "kv_transfer",
            "kv_offload",
            "recompute_cpu_offload",
        )
        or r.startswith("kv_connector=")
    ]


def _kv_connector_name(kv_transfer_config: Any) -> str | None:
    if kv_transfer_config is None:
        return None
    for attr in ("kv_connector", "connector", "engine_id"):
        raw = getattr(kv_transfer_config, attr, None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    # Class name fallback (MooncakeConnector / AscendStore…).
    cls = type(kv_transfer_config).__name__
    if cls and cls != "object":
        return cls
    return None


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


def apply_kv_meta_compat(runner: Any, cfg: Any) -> bool:
    """Refuse-gate KV meta when any unadapted feature is on. Return True if blocked.

    Call from ``_sync_kv_audit`` / bind so hot-reload cannot re-enable against
    an incompatible runner.
    """
    global _blocked_reasons, _last_block_key

    reasons = probe_incompatible_features(runner)
    _blocked_reasons = tuple(reasons)

    if reasons:
        _force_disable_kv_meta_sections(cfg)
        key = ",".join(reasons)
        if key != _last_block_key:
            _last_block_key = key
            logger.warning(
                "[runtime_guard] KV meta (kv_audit / slot_consistency / "
                "kv_slot_order / kv_state / block_state) disabled — not yet "
                "adapted for %s. logits_finite and non-KV detectors are "
                "unchanged. See runtime_guard/KV_META_TODO.md.",
                ", ".join(reasons),
            )
        return True

    _last_block_key = None
    return False
