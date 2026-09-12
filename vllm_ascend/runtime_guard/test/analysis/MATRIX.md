# SPDX-License-Identifier: Apache-2.0
"""后处理功能测试矩阵（analysis 分支专属）

路径：`vllm_ascend/runtime_guard/test/analysis/`  
跑：`pytest vllm_ascend/runtime_guard/test/analysis/ -q`

与产品 `TEST_MATRIX.md`（detector/dump/hook）分开：这里只测 **离线脚本**。

## 现状

| ID | 覆盖 | 状态 | 实现 |
|----|------|------|------|
| A1 | `report_dump_attempted` 新旧字段 | ✅ | `test_scripts.py` |
| A2 | `resolve_kv_dump_dir(dump_dir+rank)` | ✅ | `test_scripts.py` |
| A3 | `resolve_kv_dump_dir` 扫描 wave/rank | ✅ | `test_scripts.py` |
| A4 | `gather_token_rows` block 布局 | ✅ | `test_scripts.py` |

## 缺口（建议补齐）

| ID | 覆盖 | 优先级 | 做法 |
|----|------|--------|------|
| A5 | `summarize_reports` 读 `dump_attempted` | P1 | 临时 report 目录 |
| A6 | `correlate_incident` 找到 wave/rank 下 `.pt` | P0 | tmp report + kv 树 |
| A7 | `verify_request_kv` PASS/FAIL 路径 | P0 | 合成 `.pt` + report |
| A8 | `inspect_kv_dump` 打印关键字段 | P2 | 单文件 `.pt` |
| A9 | `compare_kv_similarity` / `locate_first_divergence` 同张量 cos≈1 | P0 | 两份相同 dump |
| A10 | `prepare_ref_inputs` / `request_from_report` 解析 token ids | P1 | mock HTTP 可选 |
| A11 | 无 dump_dir 旧扁平路径 fallback | P2 | 兼容性 |

**结论：后处理功能测目前不够**——核心对比/对账链路（A6/A7/A9）仍缺。
"""
