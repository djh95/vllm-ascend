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

## KV ref 标杆（P0-10 内容正确性）

「能 dump」≠「dump 对」。P0-10 用**内容余弦**证明 dump 的 KV 就是模型实际用的那份，
需一个同请求的 ref 标杆。分两段，`capture_kv_ref.sh` 一次做齐：

1. **git-safe meta**（可检入）：`dump_schema/kv_ref_meta.json` —— 只存 rank 布局 +
   每层 shape/num_kv_heads/block_ids，**不含 tensor 字节**。
2. **full ref**（机器本地，**禁止检入 git**）：`$RG_REF_ROOT`（默认 `/tmp/rg_kv_ref`）
   存完整 `wave_N/`，供 `p0_10_kv_compare.sh --target/--ref` 逐层余弦对比。

流程：`capture_kv_ref.sh WAVE=<baseline wave_N>`（TP=1 或已人工确认正确的 dump）
→ 之后每次 TP≥2 / 改动后 dump，`p0_09_tp_stitch.sh`（完整性）+
`p0_10_kv_compare.sh TARGET=<new wave> RG_REF_ROOT=$RG_REF_ROOT`（内容正确）。
比较完按 §0.4 删临时全量 dump；golden 只保留 meta JSON。
