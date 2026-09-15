# SPDX-License-Identifier: Apache-2.0
"""runtime_guard analysis（临时旁支）

## 定位

本分支 **基于 `main`**，只**新增**离线工具 / 实卡清单与脚本 / 旁支测试，**不包含**
`feat/runtime-guard-config` 的产品改动（detect/report/dump 实现与产品 CPU UT）。

| 分支 | 内容 |
|------|------|
| `main` | 上游基线 |
| `feat/runtime-guard-config` | 产品：detect / report / dump_kv / CPU UT（`TEST_MATRIX`） |
| `feat/runtime-guard-analysis`（本分支） | `main` + 下表白名单（临时工具箱） |

**本分支只添加/修改：**

```
vllm_ascend/runtime_guard/analysis/**
vllm_ascend/runtime_guard/test/analysis/**
vllm_ascend/runtime_guard/test/live/**
vllm_ascend/runtime_guard/test/perf/**
vllm_ascend/runtime_config/analysis/**
vllm_ascend/runtime_config/test/**
```

（另含使 import 成立的最小 `__init__.py`：`runtime_guard/`、`runtime_guard/test/`、`runtime_config/`。  
与 config 合 main 时注意包根 `__init__` 冲突。）

产品 CPU UT / `TEST_MATRIX` 在 **config**。  
**实卡 live 清单、启动脚本、golden、NPU perf** 在 **本分支**（需挂 config 产品 build 才能起服）。  
dump 布局以 **config 运行时写盘**为准，本仓库脚本按约定读盘。  
合 main 注意点见 `MERGE.md`。

---

## 模块

| 模块 | 路径 | 说明 |
|------|------|------|
| **分析后处理** | `analysis/scripts/` + `skill/` | CLI + 调查流程 |
| **后处理 UT** | `test/analysis/` | 脚本 UT；见 `MATRIX.md` |
| **实卡功能** | `test/live/` | 清单 + `scripts/` + `configs/` + `golden/` |
| **实卡 perf** | `test/perf/` | T0–T3 / C1–C6；**须 ModelRunner v1+v2**；需 NPU |
| **预留** | `runtime_config/{analysis,test}/` | 空壳，按需扩展 |

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
| `stitch_kv` | 多 rank TP 头维拼接 + DP/TP/PP 覆盖检查 + 跨 TP/PP 对比 |

落盘约定（config 产品）：

```
runtime/report/<type>/report_*.json
runtime/report/kv_cache/<type>/<req_id>/wave_<N>/<rank_tag>/*.pt
```

当前产品 `DETECTOR_SECTIONS`（config）仅：`spec_acceptance` / `output_substring` /
`token_repeat` / `logits_finite`。skills 里若仍写 `slot_*` / `kv_state`，视为**未合入**，勿当已交付。

## 测试

```bash
# 后处理 CPU UT
pytest vllm_ascend/runtime_guard/test/analysis/ -q

# 实卡：见 test/live/FUNCTIONAL_TEST_LIST.md、test/live/scripts/
# 性能：见 test/perf/README.md、test/perf/scripts/
```

缺口与运维约定：live §0.4（磁盘）、§10（注入闭环）、§14–§15（标杆/dump）。
"""
