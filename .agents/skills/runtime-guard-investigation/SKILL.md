---
name: runtime-guard-investigation
description: >-
  End-to-end investigation of intermittent NPU/vllm-ascend bugs on a
  runtime_guard-instrumented deployment using native dump_kv only (no msprobe):
  triage → repro → dump_kv sanity → stress + capture → ref dump_kv →
  locate_first_divergence / compare_per_layer. Use for token_repeat, garbled
  output, NaN, KV corruption, or similar live issues.
---

# runtime_guard investigation

正文与完整 8-step 流程：`vllm_ascend/runtime_guard/analysis/skill/investigation/SKILL.md`  
分析脚本：`python -m vllm_ascend.runtime_guard.analysis.scripts.<module>`  
抓现场后离线分析：`.agents/skills/runtime-guard-analysis/SKILL.md`  
抓 ref + 两表对比：`.agents/skills/runtime-guard-ref-kv-dump/SKILL.md`

**仅 native `dump_kv`。** 按 `skill/investigation/SKILL.md` 逐步执行。
