---
name: runtime-guard-ref-kv-dump
description: >-
  Capture reference KV via dump_kv (force-feed token IDs) and compare to buggy
  kv_cache with locate_first_divergence / compare_per_layer. No msprobe.
---

# runtime_guard ref KV

正文：`vllm_ascend/runtime_guard/analysis/skill/ref-kv-dump/SKILL.md`  
脚本：`prepare_ref_inputs` / `locate_first_divergence` / `compare_per_layer`

**按正文执行。**
