# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async-scheduling YAML-hookable note APIs."""

from __future__ import annotations


class AsyncSchedulingMetricHooks:
    """Hook points for ms-service-metric YAML and runtime patches.

    Runtime code calls these as signals; the YAML symbol hook intercepts the
    call and records the corresponding counter.
    """

    @staticmethod
    def note_stale_discard() -> None:
        pass

    @staticmethod
    def note_placeholder_underflow() -> None:
        pass
