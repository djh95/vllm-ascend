# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashComm per-forward state and hookable metric APIs."""

from __future__ import annotations

from dataclasses import dataclass

DECISION_ENABLED = "enabled"
DECISION_CONFIG_OFF = "config_off"
DECISION_DENSE_BELOW_THRESHOLD = "dense_below_threshold"
DECISION_DENSE_DRAFT = "dense_draft"
DECISION_NO_NUM_TOKENS = "no_num_tokens"

FLASHCOMM_DENSE_TOKEN_THRESHOLD = 1000


@dataclass
class FlashCommForwardStats:
    decision: str | None = None
    flash_comm_v1_enabled: bool = False
    num_tokens: int = 0
    pad_size: int = 0


_current: FlashCommForwardStats | None = None


def classify_decision(
    *,
    enable_sp: bool,
    is_moe_model: bool,
    is_draft_model: bool,
    num_tokens: int | None,
    flash_comm_v1_enabled: bool,
) -> str:
    if not enable_sp:
        return DECISION_CONFIG_OFF
    if num_tokens is None:
        return DECISION_NO_NUM_TOKENS
    if is_draft_model and not is_moe_model:
        return DECISION_DENSE_DRAFT
    if not is_moe_model and num_tokens <= FLASHCOMM_DENSE_TOKEN_THRESHOLD:
        return DECISION_DENSE_BELOW_THRESHOLD
    if flash_comm_v1_enabled:
        return DECISION_ENABLED
    return DECISION_CONFIG_OFF


def publish_forward_gate(
    *,
    decision: str,
    flash_comm_v1_enabled: bool,
    num_tokens: int | None,
    pad_size: int,
) -> FlashCommForwardStats:
    global _current
    stats = FlashCommForwardStats(
        decision=decision,
        flash_comm_v1_enabled=bool(flash_comm_v1_enabled),
        num_tokens=int(num_tokens or 0),
        pad_size=int(pad_size or 0),
    )
    _current = stats
    return stats


class FlashCommMetricHooks:
    """YAML-hookable entry points."""

    @staticmethod
    def publish_forward_gate(
        *,
        decision: str,
        flash_comm_v1_enabled: bool,
        num_tokens: int | None,
        pad_size: int,
    ) -> FlashCommForwardStats:
        return publish_forward_gate(
            decision=decision,
            flash_comm_v1_enabled=flash_comm_v1_enabled,
            num_tokens=num_tokens,
            pad_size=pad_size,
        )
