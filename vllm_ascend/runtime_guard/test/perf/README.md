# SPDX-License-Identifier: Apache-2.0
"""Perf / live NPU benches moved off this branch.

Authoritative location (analysis toolbox branch)::

    feat/runtime-guard-analysis
    vllm_ascend/runtime_guard/test/perf/     # C1–C6, scripts, README
    vllm_ascend/runtime_guard/test/live/    # functional checklist + launch scripts

This product branch keeps CPU UTs under ``vllm_ascend/runtime_guard/test/``
(see ``TEST_MATRIX.md``). Do **not** re-add NPU throughput scripts here.
"""
