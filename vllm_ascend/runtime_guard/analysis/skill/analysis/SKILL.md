---
name: runtime-guard-analysis
description: >-
  Analyze runtime_guard anomaly reports and per-request KV .pt dumps. Use when
  the user mentions runtime_guard report JSON, kv_cache dumps, dump_kv,
  correlate req_id, nan/inf in KV, or post-mortem of token_repeat / block_kv /
  manual_trigger captures.
---

# runtime_guard analysis（抓现场后）

前提：线上已写出 `report_*.json`，可选写出 `dump_kv` 的 `.pt`。  
本 skill 只做离线读盘分析，不改 `runtime_config`、不重跑服务。

默认 `--report-dir`：`./runtime/report`（可问用户确认）。

## 分析流程

### Step 0 — 确认输入根目录

| | |
|---|---|
| **做什么** | 确定 report 根目录 |
| **脚本** | 无 |
| **输入** | 用户路径，或默认 `./runtime/report` |
| **输出** | 后续命令统一使用的 `--report-dir` |

### Step 1 — 汇总报告

| | |
|---|---|
| **做什么** | 扫最近报告，挑目标 `incident_type` / `req_id` |
| **脚本** | `summarize_reports` |
| **输入** | `--report-dir`；可选 `--limit`、`--incident-type`、`--detail` |
| **输出** | 表：file / type / req_id / prompt / output / blocks / dump；末尾 type 计数 |
| **判据** | 有行则可继续；空目录 → 停，告知无报告 |

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.summarize_reports \
  --report-dir ./runtime/report --limit 20
```

### Step 2 — 按 req 关联

| | |
|---|---|
| **做什么** | 同一 `req_id` 下报告与 KV 目录对齐 |
| **脚本** | `correlate_incident` |
| **输入** | `--report-dir`、`--req-id`；可选 `--incident-type` |
| **输出** | 匹配的 report 路径、`dump_armed` / `block_ids`、`kv_cache/.../<req_id>/` 与 `shared/` 文件列表；并打印下一步 `verify_request_kv` 命令 |
| **判据** | `reports>=1`；若 `dump_armed` 但 kv 目录缺失 → 记为配额/路径问题 |

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.correlate_incident \
  --report-dir ./runtime/report --req-id <req_id>
```

### Step 3 — report ↔ KV 对账

| | |
|---|---|
| **做什么** | 校验 token/block 信息与 `.pt` 是否可读、有限、层文件存在 |
| **脚本** | `verify_request_kv` |
| **输入** | `--report <report_*.json>`；可选 `--report-dir`、`--kv-dir`、`--block-size`、`--min-files` |
| **输出** | 分段检查日志；结论 `PASS` 或 `FAIL`（exit 0/1） |
| **判据** | `PASS` 再深入看 tensor；`FAIL` 先修路径/配额/`on_trigger`，不臆测数值根因 |

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.verify_request_kv \
  --report ./runtime/report/<type>/report_....json \
  --report-dir ./runtime/report
```

### Step 4 — 抽查单个 `.pt`（按需）

| | |
|---|---|
| **做什么** | 看某一层 shape / nan / inf / min-max / head |
| **脚本** | `inspect_kv_dump` |
| **输入** | `--path <file>.pt`；可选 `--max-print` |
| **输出** | `req_id` / `layer` / `block_ids` / `tp_rank` / `num_kv_heads` / 统计量 |
| **判据** | nan/inf 记为发现；不等于 dump 损坏 |

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.inspect_kv_dump \
  --path ./runtime/report/kv_cache/<type>/<req_id>/<file>_req.pt
```

payload 还带 `tp_rank` / `num_kv_heads`（本 TP shard 切过的 KV head 数）— 跨 rank / 跨 TP 对比前先用它们核对
两份 dump 的 head 切片一致，否则 cos 差可能只是 TP 切分不同而不是数值问题。层名排序是 natural sort
（`layer_2` < `layer_10`），首个发散层的序号可信。

### Step 5 — 结论

用中文（除非用户要求其他语言）汇总：

1. `incident_type`、`req_id`、`dump_armed` / wave（若有）
2. `block_ids` 数量与 verify 结论
3. 关键要层的 nan/inf / 数值范围（若跑了 Step 4）
4. 未覆盖项（例如无 dump、仅 shared、token 列表缺失）

## 落盘约定

```
runtime/report/
  <incident_type>/report_*.json
  kv_cache/<incident_type>/<req_id>/*.pt
  kv_cache/<incident_type>/shared/*.pt
```

默认是 **本请求 `block_ids` 快照**，不是全 cache、不是 op 级 dump。

## 需要定界 / 标杆对比时

本 skill 只做现场对账。端到端调查从 `runtime-guard-investigation` 开始；抓 ref + **两表对比**用 `runtime-guard-ref-kv-dump`（脚本：`prepare_ref_inputs` / `locate_first_divergence` / `compare_per_layer`）。

## Do not

- 不改 `runtime_config.json`、不重启服务（除非用户明确要求）
- 不假设全量 KV dump（native `dump_kv` 默认是本请求 `block_ids`）
- 不在 NPU 上 load `.pt`（脚本用 CPU `map_location`）
- 不用外部 / msprobe 布局工具去对 native `kv_cache/*.pt` — 只用本 skill 的 `verify_request_kv`
