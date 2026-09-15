# SPDX-License-Identifier: Apache-2.0
"""后处理功能测试矩阵（analysis 分支专属）

路径：`vllm_ascend/runtime_guard/test/analysis/`  
跑：`pytest vllm_ascend/runtime_guard/test/analysis/ -q`

实卡侧 dump/report/标杆/注入定位见 `../live/FUNCTIONAL_TEST_LIST.md` §0.4 / §10 / §15。  
磁盘：测后及时删临时 dump；golden 只收小文件。

## 用例

| ID | 覆盖 | 状态 |
|----|------|------|
| A1 | `report_dump_attempted` | ✅ |
| A2 | `resolve_kv_dump_dir(dump_dir+rank)` | ✅ |
| A3 | `resolve_kv_dump_dir` 扫描 wave/rank | ✅ |
| A4 | `gather_token_rows` block 布局 | ✅ |
| A5 | `summarize_reports` / `dump_attempted` 列 | ✅ |
| A6 | `correlate_incident` → wave/rank `.pt` | ✅ |
| A7 | `verify_request_kv` PASS / missing FAIL | ✅ |
| A9 | `compare_kv_dumps` 同张量无坏点 | ✅ |
| A10 | `prepare_ref_inputs` force-feed JSON | ✅ |
| A12 | `diff_report_golden` 关键字段对拍 | ✅ |
| A13 | `parse_rank_tag` dp/tp/pp/cp 解析 | ✅ |
| A14 | `stitch_tp_heads` TP 头维拼接（dim=-2） | ✅ |
| A15 | `stitch_kv_dir` 多 rank 按层拼接 | ✅ |
| A16 | `compare_stitched_kv` PP 部分覆盖（missing/extra） | ✅ |
| A17 | `stitch_kv` CLI `--dump-dir` 覆盖检查 | ✅ |

可选后续：A8 `inspect_kv_dump`；A11 旧扁平路径。
"""
