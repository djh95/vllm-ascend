# SPDX-License-Identifier: Apache-2.0
"""runtime_guard analysis

仓内调查技能与脚本（**仅 native ``dump_kv``**，无 msprobe）。

## Skills（5）

| Skill | 正文 | Cursor 入口 | 何时用 |
|---|---|---|---|
| `runtime-guard-investigation` | `skill/investigation/` | `.agents/skills/runtime-guard-investigation/` | **主入口**：偶发 bug 端到端 |
| `runtime-guard-detector-sweep` | `skill/detector-sweep/` | `.agents/skills/runtime-guard-detector-sweep/` | 全开 detector 压测 |
| `runtime-guard-config-recommender` | `skill/config-recommender/` | `.agents/skills/runtime-guard-config-recommender/` | 推荐 `runtime_config.json` |
| `runtime-guard-ref-kv-dump` | `skill/ref-kv-dump/` | `.agents/skills/runtime-guard-ref-kv-dump/` | 抓 ref（同 token ids）+ 两表对比 |
| `runtime-guard-analysis` | `skill/analysis/` | `.agents/skills/runtime-guard-analysis/` | 现场汇总 / 关联 / 对账 / 抽查 |

## 落盘

```
runtime/report/
  <incident_type>/report_*.json
  kv_cache/<incident_type>/<req_id>/*.pt
```

`.pt`：`req_id`, `block_ids`, `dump_all_blocks`, `layer`, `source`, `tensor`

## 脚本

`python -m vllm_ascend.runtime_guard.analysis.scripts.<name> ...`

| 脚本 | 作用 |
|---|---|
| `summarize_reports` | 报告表 + type 计数 |
| `correlate_incident` | `req_id` ↔ report + kv 目录 |
| `verify_request_kv` | report ↔ 请求 `.pt` 对账 |
| `inspect_kv_dump` | 单 `.pt` 统计 |
| `prepare_ref_inputs` | 从 report 写出 force-feed token JSON |
| `request_from_report` | 读 report 拼接 token ids，POST `/v1/completions`（force-feed） |
| `compare_kv_similarity` | **两 dump + 两 report**：逐 token min-cos + 第一坏点逐层 |
| `locate_first_divergence` | 两 dump + 单 report（同上，简化入口） |
| `compare_per_layer` | 按层聚合 cos |

Force-feed 到在线服务（从 report 拼 token ids，走 `/v1/completions`；默认 `--feed history`）：

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.request_from_report \
  --report ./runtime/report/token_repeat/report_xxx.json \
  --url http://127.0.0.1:8000/v1/completions \
  --model <served-model-name>
```

对比示例：

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.prepare_ref_inputs \
  --report ./runtime/report/token_repeat/report_xxx.json --out ref_inputs.json

# 干净服务上 dump_kv + force-feed 后（推荐双 report）：
python -m vllm_ascend.runtime_guard.analysis.scripts.compare_kv_similarity \
  --buggy-dir ./runtime/report/kv_cache/token_repeat/<bad_req>/ \
  --ref-dir   ./runtime/report/kv_cache/manual_trigger/<ref_req>/ \
  --buggy-report ./runtime/report/token_repeat/report_xxx.json \
  --ref-report   ./runtime/report/manual_trigger/report_ref.json

# 单 report 也可用 locate_first_divergence：
python -m vllm_ascend.runtime_guard.analysis.scripts.locate_first_divergence \
  --buggy-dir ./runtime/report/kv_cache/token_repeat/<bad_req>/ \
  --ref-dir   ./runtime/report/kv_cache/manual_trigger/<ref_req>/ \
  --report    ./runtime/report/token_repeat/report_xxx.json
```

`--head` 默认 `0`：张量为 `[N,H,D]` 时只比 head 0。
