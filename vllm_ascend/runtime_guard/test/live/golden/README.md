# live golden（标杆数据）

**归属**：仅 `feat/runtime-guard-analysis`。

存放**裁剪后的**验收标杆，供 §15 / §10 回归对比：

| 子目录 | 内容 | 注意 |
|--------|------|------|
| `reports/` | 各 inject / 典型命中的 report **摘要** JSON | 去掉时间戳、绝对路径、过大 token 列表；保留 `incident_type`、关键 detail |
| `dump_schema/` | 单层小样本 meta / 极小 `.pt` 或 shape JSON | **禁止**检入全量多层 KV |

## 刷新流程

1. 实卡按 §10/§15 跑通一次（v2 先，再视需要 v1）。  
2. 人工确认 report/dump 正确。  
3. 用脚本生成脱敏摘要 → 覆盖对应 golden 文件。  
4. 删除机房临时 `kv_cache` 全量 dump（§0.4）。  
5. 若流程/字段变化 → **同步更新** `analysis/skill/*/SKILL.md`。

## 对比

- Report：关键字段与 golden 一致（浮动字段白名单：`req_id`、时间、绝对路径）。  
- Dump：优先 schema/shape；数值对比走 `compare_kv_similarity`（临时全量，比完即删）。
