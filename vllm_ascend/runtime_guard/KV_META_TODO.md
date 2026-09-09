# KV meta / kv_audit — 待优化清单

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

## 建议消化顺序

1. P0-1 旁路写路径挂钩  
2. P0-3 reshape finding 不丢  
3. P0-2 AscendStore load  
4. P1-5 offload `num_tokens`  
5. 其余 P1 → P2；P0-4 内容校验单独立项（成本高）

---

## 门控后适配 backlog（已记，未做 → **当前一律拒开**）

> 2026-09-09：`kv_meta_compat` 对下列未适配特性 **强制关** KV meta（L1+L2）并
> warning；仅稠密本地 PA `reshape_and_cache` 路径可开。适配完成后再从拒开名单移除。

| # | 项 | 探测 / 拒开 tag | 粗估 |
|---|----|-----------------|------|
| A1 | Sliding window / attention sink | `sliding_window` | ~300–600 LOC |
| A2 | NZ / SFA / DSA 旁路 scatter | `enable_sparse_sfa_c8` / `enable_sparse_li_c8` / `enable_dsa_cp`（其余旁路靠 P0-1 补钩后可收紧探测） | ~150–400 LOC |
| A3 | AscendStore / PD / kv_transfer | `kv_transfer` (+ `kv_connector=…`) | ~80–200 LOC（Store）；PD 钩已有但仍整体拒开至验完 |
| A4 | Offload（含 recompute CPU） | `kv_offload` / `recompute_cpu_offload` | ~50–150 LOC |
| A5 | reshape finding 不丢 | （缺陷修复，非特性开关） | ~80–150 LOC |
| A6 | Load 路径内容弱校验 | 随 A3/A4 解除拒开后做 | 400+ LOC |
| A7 | Sparse KV | `sparse_kv_offload` | 1.5k–3k+ LOC |
| A8 | Mamba / GDN / hybrid | `hybrid_mamba_model` / `mamba_cache_mode=*` | 1k–2k+ LOC |
| A9 | Prefix caching | `prefix_caching` | 随 LOAD 语义验收后解除 |
