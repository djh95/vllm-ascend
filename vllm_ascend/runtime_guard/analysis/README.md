# SPDX-License-Identifier: Apache-2.0
"""runtime_guard analysis（临时旁支）

## 定位

本分支 **基于 `main`**，只**新增**离线工具 / 旁支测试文件，**不包含**
`feat/runtime-guard-config` 的产品改动。

| 分支 | 内容 |
|------|------|
| `main` | 上游基线 |
| `feat/runtime-guard-config` | 产品：detect / report / dump_kv / 产品 UT |
| `feat/runtime-guard-analysis`（本分支） | `main` + 下表白名单文件（临时工具箱） |

**本分支只添加/修改：**

```
vllm_ascend/runtime_guard/analysis/**
vllm_ascend/runtime_guard/test/analysis/**
vllm_ascend/runtime_config/analysis/**
vllm_ascend/runtime_config/test/**
```

（另含使 import 成立的最小 `__init__.py`：`runtime_guard/`、`runtime_guard/test/`、`runtime_config/`。）

产品 UT / `TEST_MATRIX` / 实卡 perf **不在本分支**；请到 config。  
dump 布局以 **config 运行时写盘**为准，本仓库脚本按约定读盘。

---

## 模块

| 模块 | 路径 | 说明 |
|------|------|------|
| **分析后处理** | `analysis/scripts/` + `skill/` | CLI + 调查流程 |
| **后处理功能测** | `test/analysis/` | 脚本 UT；清单见 `MATRIX.md` |
| **预留** | `runtime_config/{analysis,test}/` | 空壳，按需扩展 |
| **实卡功能/性能/拓扑** | （待建）`test/live/` 等 | 不进 config CI |

---

## 后处理脚本

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.<name> ...
```

| 脚本 | 作用 |
|------|------|
| `summarize_reports` | 报告汇总 |
| `correlate_incident` | req ↔ report + kv |
| `verify_request_kv` | report ↔ `.pt` |
| `inspect_kv_dump` | 单文件抽查 |
| `prepare_ref_inputs` / `request_from_report` | force-feed |
| `compare_kv_similarity` / `locate_first_divergence` / `compare_per_layer` | buggy vs ref |

落盘约定（config 产品）：

```
runtime/report/<type>/report_*.json
runtime/report/kv_cache/<type>/<req_id>/wave_<N>/<rank_tag>/*.pt
```

## 测试

```bash
pytest vllm_ascend/runtime_guard/test/analysis/ -q
```

缺口见 `test/analysis/MATRIX.md`（后处理 UT 仍偏少）。
"""
