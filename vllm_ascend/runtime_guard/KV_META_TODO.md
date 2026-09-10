# KV meta / kv_audit — 待优化清单

> 方案总览（问题类型 / 特性支持 / 适配量）：见同目录 [`KV_META_DESIGN.md`](./KV_META_DESIGN.md)。  
> 产品需求（不绑 PR-A 实现）：[`docs/zh/design/runtime_guard_requirements.md`](../../docs/zh/design/runtime_guard_requirements.md)。

记录时间：2026-09-07。对照当前已落地能力（见下「已覆盖」）列出后续完善项；**尚未实现**，按优先级消化。

## 已覆盖（基线，勿当缺口）

| 路径 | 挂钩 |
|------|------|
| DeviceOperator `reshape_and_cache` | `on_write_from_slot_mapping` |
| Mooncake PD recv（connector / hybrid / layerwise） | `on_pd_recv_load` + `num_tokens` |
| kv_offload H2D（native / simple） | `on_block_load(tag=kv_load)` |
| zero | `on_invalidate` |
| v1 CoW（`kv_cache_block_copies`） | `on_block_copies`（克隆 state + 已知 slot token） |
| 状态机 | `UNKNOWN` / `SLOT_PARTIAL` / `BLOCK_SEALED_FILL` / `BLOCK_SEALED_LOAD` |
| note_kv | `merge_slot_writes` → `kv_state` / `kv_slot_order` / `slot_consistency` |

---

## P0 — 正确性 / 漏挂

| # | 项 | 问题 | 建议方向 |
|---|----|------|----------|
| 1 | NZ / C8 / SFA / DSA 等旁路 scatter | 真写 KV 但未调 `on_write_from_slot_mapping`（如 `attention_v1` NZ、`sfa_v1`、`dsa_*`） | 各 scatter/store 入口复用同一 audit 钩子 |
| 2 | AscendStore load | 池化 H2D/recv 未挂 `on_block_load` / `on_pd_recv_load` | load 成功后挂 + 尽量传 `num_tokens` |
| 3 | reshape → note_kv 吞掉 `kv_state` | reshape 侧 `apply_slot_writes` finding 被丢弃；note_kv 对已推进 offset 走 stamp → **假阴性** | 缓存/上报 reshape finding，或禁止「只 degrade 不报」 |
| 4 | PD / load **内容**未验 | 只比对 meta token vs seq；load 常无 token → `check_and_fill` fill unverified，传错块仍可能 PASS | 可选 dump/hash/对拍；或明确标记 unverified、不声称内容正确 |

## P1 — 误报 / 陈旧账本

| # | 项 | 问题 | 建议方向 |
|---|----|------|----------|
| 5 | offload H2D 无 `num_tokens` | 尾块一律 `BLOCK_SEALED_LOAD` | 传入覆盖 token 数 / valid_len |
| 6 | sparse / recompute offload | 额外 D2H/H2D 未挂 | 对齐 native/simple H2D；必要时 unload invalidate |
| 7 | sleep / wake / free（无 zero） | 账本不清理 → 陈旧 SEALED/token | sleep 时 `clear()`；free 复用路径 invalidate |
| 8 | slot cap 淘汰 + seq 回填 | `_SLOT_META_CAP` 丢旧 token 后 `check_and_fill` 用 seq 重填 → **串块假阴性** | 整块淘汰、标 unverified、或 pin 活跃 req |
| 9 | `output_len==0` 整段静默 LOAD 非 0 | prefill 中途真 bug 也可能不报 | 收紧为「已知尾块续写」；FILL 已始终告警 |

## P2 — 边角 / 文档 / 卫生

| # | 项 | 问题 | 建议方向 |
|---|----|------|----------|
| 10 | `attention_v1.copy_blocks` / `swap_blocks` | 未挂；v1 热路径主要走 CoW hook | 若仍有调用则挂 `on_block_copies`；确认 v2 runner |
| 11 | `_blocks` 无上限 | 长跑碰过的 block id 只增不减 | 与 slot 类似 cap/reap，或随 free invalidate |
| 12 | 文档 / UT 漂移 | 部分仍写单一 `BLOCK_SEALED`；auto-on 说明与 `kv_state` 不一致 | 对齐 `runtime_config.md` / design / ANOMALY_INJECTION / UT |
| 13 | Mamba / GDN | 不在 PA slot 账本内 | 文档标明「仅 paged attention」；另案再做 |
| 14 | 非 load 整块覆盖独立 incident | 无第二整块写路径 call site | 有真实路径再挂 `sealed_block_overwrite` |

---

## 建议消化顺序（与 DESIGN §6 / §9 对齐）

1. **门控改到目标策略**：prefix / PD / offload **允许 L1**；仅强制关 L2 或标 unverified；sparse / sliding / Mamba **仍拒**（见 `KV_META_DESIGN.md` §9.2）  
2. P0-1 旁路写路径挂钩  
3. P0-3 reshape finding 不丢  
4. P0-2 AscendStore load + P1-5 offload `num_tokens`  
5. P0-4 内容 checksum（对齐社区 RFC，挂 load 路径）  
6. 其余 P1 → P2；sliding / sparse / SSM 单独立项  

策略结论、社区对照、上游 L1 切片：见 [`KV_META_DESIGN.md`](./KV_META_DESIGN.md) §9–§12。

---

## 门控后适配 backlog

> **当前代码（2026-09-09）：** `kv_meta_compat` 对下列特性 **一律强制关 L1+L2**；仅稠密本地 PA 可开。  
> **目标（讨论结论，待改）：** A3/A4/A9 改为「允许 L1、关 L2」；A1/A2/A7/A8 仍拒开整个 meta。细节见 DESIGN §9.2。

| # | 项 | 探测 / 拒开 tag | 粗估 | 目标门控 |
|---|----|-----------------|------|----------|
| A1 | Sliding window / attention sink | `sliding_window` | ~300–600 LOC | 仍拒（至适配完） |
| A2 | NZ / SFA / DSA 旁路 scatter | `enable_sparse_sfa_c8` / `enable_sparse_li_c8` / `enable_dsa_cp` | ~150–400 LOC | 仍拒 |
| A3 | AscendStore / PD / kv_transfer | `kv_transfer` (+ `kv_connector=…`) | ~80–200 LOC（Store） | **允许 L1**；内容靠 A6 |
| A4 | Offload（含 recompute CPU） | `kv_offload` / `recompute_cpu_offload` | ~50–150 LOC | **允许 L1** |
| A5 | reshape finding 不丢 | （缺陷修复，非特性开关） | ~80–150 LOC | — |
| A6 | Load 路径内容弱校验 | 随 load 路径 | 400+ LOC | checksum / dump |
| A7 | Sparse KV | `sparse_kv_offload` | 1.5k–3k+ LOC | 仍拒 |
| A8 | Mamba / GDN / hybrid | `hybrid_mamba_model` / `mamba_cache_mode=*` | 1k–2k+ LOC | 仍拒 |
| A9 | Prefix caching | `prefix_caching` | LOAD 语义验收 | **允许 L1**；L2 unverified |
