# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async-scheduling accumulators and YAML-hookable note APIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _AsyncSchedulingAccum:
    stale_discard: int = 0
    placeholder_underflow: int = 0


_ACCUM = _AsyncSchedulingAccum()


def snapshot_and_reset() -> _AsyncSchedulingAccum | None:
    if _ACCUM.stale_discard == 0 and _ACCUM.placeholder_underflow == 0:
        return None
    snap = _AsyncSchedulingAccum(
        stale_discard=_ACCUM.stale_discard,
        placeholder_underflow=_ACCUM.placeholder_underflow,
    )
    _ACCUM.stale_discard = 0
    _ACCUM.placeholder_underflow = 0
    return snap


class AsyncSchedulingMetricHooks:
    """Hook points for ms-service-metric YAML and runtime patches."""

    @staticmethod
    def note_stale_discard() -> None:
        _ACCUM.stale_discard += 1

    @staticmethod
    def note_placeholder_underflow() -> None:
        _ACCUM.placeholder_underflow += 1
