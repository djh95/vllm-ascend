# SPDX-License-Identifier: Apache-2.0
"""Merge notes: analysis toolbox vs product config branch.

When landing both onto ``main``:

1. **Prefer product** ``vllm_ascend/runtime_guard/__init__.py`` from
   ``feat/runtime-guard-config`` (real package exports). The analysis stub is
   only for importability of offline scripts on a main-based worktree.
2. Keep analysis-only trees as additive:
   ``runtime_guard/analysis/**``, ``runtime_guard/test/{analysis,live,perf}/**``.
3. Do **not** merge analysis ``test/perf`` pointer from config — config only
   keeps a short README pointing here.
4. Resolve conflicts by taking config product modules + analysis tool paths.
"""
