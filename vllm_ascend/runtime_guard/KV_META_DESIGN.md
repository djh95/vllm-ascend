# KV meta 方案说明（PR-C）

> 状态：WIP（`feat/runtime-guard-kv`）  
> 代码：`kv_audit` / `kv_block_meta` / `invariant/*` / `kv_meta_compat`  
> 待办清单：[`KV_META_TODO.md`](./KV_META_TODO.md)  
> 记录：2026-09-09

本文汇总：**两层检测方案**、**三处数据面与责任划分**、**问题类型与能否解决**、**省 KV 特性支持与适配量**。与实现不一致处以代码与 `KV_META_TODO` 为准。

---

## 1. 方案目标

给 **Paged Attention 的物理块写路径** 挂旁路账本，做 soft-assert：

- **不**替代输出侧 detector（token_repeat / logits_finite 等）
- **不**默认做 KV 浮点内容 hash（成本高；见 TODO P0-4）
- **验** 块生命周期、写序、slot 与推理 token 序列是否对得上

命中后可走既有 `report` / `dump_kv`，配合 PR-B 离线 analysis 对拍。

---

## 2. 两层检测

| 层 | 名称 | 验什么 | 开销 | 上限 |
|----|------|--------|------|------|
| **L1** | 块级状态 | `UNKNOWN` / `SLOT_PARTIAL` / `BLOCK_SEALED_FILL` / `BLOCK_SEALED_LOAD`；密封后再写、跳 offset | **小**（每块少量标量） | 长跑需 cap/reap（TODO P2-11） |
| **L2** | Slot 一致性 | 账本 slot token vs 调度推演的序列位置；`kv_slot_order` / `slot_consistency` | **主机内存大** | `_SLOT_META_CAP`；淘汰后 seq 回填可 **假阴性** |

| Incident（示意） | 层 | 含义 |
|------------------|----|------|
| `kv_state` / sealed rewrite | L1 | FILL 后再非 0 写；LOAD 后再非 0 写（规则见 design） |
| `kv_slot_order` | L1+写序 | gap / wrong_start |
| `kv_slot_token` | L2 | meta token ≠ 序列对应位置 |

**当前门控（`kv_meta_compat`）：** 凡未适配特性一律 **拒开 L1+L2**（强制关 `kv_audit` / `block_state` / 上述 invariant）；仅 **稠密本地 PA `reshape_and_cache`** 可开。`logits_finite` 不受影响。

---

## 3. 三处「KV 相关数据」与控制关系

### 3.1 Scheduler（控制面，无 KV 字节）

| 数据 | 含义 |
|------|------|
| `Request.block_hashes` | 前缀内容指纹链，用于 prefix / 外池 lookup |
| `BlockPool` hash→block | block_id、refcount、cached 标志——**不是** NPU 张量 |
| 每请求 `block_ids` / `num_computed_tokens` | 分配给谁、认为有效前缀多长 |
| free / refcount | 复用与释放 |

每步经 `SchedulerOutput` 下发到 Worker。Scheduler **管身份与映射**。

### 3.2 Worker（数据面）

| 数据 | 含义 |
|------|------|
| NPU paged KV 张量 | 真实 K/V |
| `block_table` / `slot_mapping` | 逻辑位置 → 物理 block/slot |
| 写入 | 正常 `reshape_and_cache`；或 PD/offload H2D 灌已分配块 |

Worker **管槽内字节**，默认信任调度给的 block_id。

### 3.3 池化 / 外置（历史字节）

| 形态 | 索引 | 内容 |
|------|------|------|
| CPU offload | 本地 block_id DMA | NPU↔CPU 镜像 |
| Mooncake PD | 对端/本地 block 列表 | P→D 灌字节 |
| AscendStore 等 | chunk_hash（来自 block_hashes） | 外后端历史 KV |

命中后仍由 Scheduler alloc 本地块，再由 connector 灌入。

### 3.4 控制流（简图）

```text
tokens → block_hashes → BlockPool / 外池 lookup
      → alloc block_ids, num_computed_tokens
      → SchedulerOutput → Worker block_table / slot_mapping
      → [可选] 池/PD/H2D 灌块
      → reshape_and_cache 写剩余 slot
```

runtime_guard 账本主要挂在 **Worker 写路径 + 下发的 block_ids/序列**；不直接审计 Scheduler 哈希表内部，也不默认验池内字节完整性。

---

## 4. 问题类型：谁出错、方案能否解决

| ID | 故障类型 | 更可能责任方 | L1 | L2 | 结论 |
|----|----------|--------------|----|----|------|
| W1 | Worker 写污染（错 slot / 密封再写 / 乱序） | Worker 写路径 | 强 | 强 | **能解决**（稠密 PA 且已挂钩） |
| W2 | Worker↔Scheduler 未对齐（computed/block_ids） | 同步 / CoW / 下发 | 中 | 中～强 | **能报警**；根因需对照 SchedulerOutput |
| S1 | Hash / 前缀误命中 | Scheduler hasher / BlockPool | 弱 | 中（有 token 时） | **不能直接验 hash**；L2 可能间接暴露 |
| S2 | 调度复用/释放错（串块） | Scheduler alloc/free | 强 | 强 | **能解决**（表现在 Worker 侧） |
| P1 | 池内脏读 / 镜像被踩 | 池存储 | 弱 | 弱 | **当前不能**（需 dump/hash，P0-4） |
| P2 | PD 传输错（映射/少传） | Connector | 中 | 中 | **部分**；无 token 的 LOAD 易假阴性 → **现拒开** |
| P3 | Store lookup 对、字节错 | 外池内容 | 弱 | 弱 | **当前不能** → **现拒开 kv_transfer** |
| C1 | CoW 后语义不同步 | CoW + 状态 | 有钩可降 | 有克隆可降 | **钩齐时可较好** |
| C2 | 旁路写未挂钩（NZ/SFA/DSA…） | 观测缺口 | 假阴性 | 假阴性 | **现对可探测 flag 拒开**；其余靠 P0-1 |

粗分：

| 层 | 一句话 |
|----|--------|
| Scheduler 错 | 身份/命中/分配错——Worker 按错地图读写 |
| Worker 错 | 地图对，写坏了 |
| 池/传输错 | 地图与写逻辑对，灌入的历史字节错 |
| 对不齐 | 三边对「是否有效/多长」认知不一致 |

---

## 5. 省 KV 空间技术：实现要点与特性支持

| 技术 | 如何省空间 | 与检测关系 | **当前支持** |
|------|------------|------------|--------------|
| PagedAttention（基线） | 按块分配复用 | 原生模型 | ✅ **支持** |
| GQA / MQA | 少 KV head | 仍一 token 一 slot | ✅ 布局兼容（无单独拒开） |
| KV 量化 | 低精度存 | meta 不关心 dtype | ✅ 布局兼容 |
| MLA | 压缩 latent | 旁路写风险 | ⚠️ 布局可类似；旁路未全挂 → 靠 SFA/DSA flag / P0-1 |
| Prefix caching | hash 复用块 | LOAD 语义 | ❌ **拒开** `prefix_caching` |
| PD / kv_transfer | 分机 / 传块 | LOAD；内容未验 | ❌ **拒开** `kv_transfer` |
| CPU / recompute offload | 冷块外置 | LOAD；num_tokens 缺口 | ❌ **拒开** |
| AscendStore | hash 外池 | 钩不全 | ❌ **拒开**（随 kv_transfer） |
| Sliding / sink | 只留窗口 | 对账前缀变 | ❌ **拒开** `sliding_window` |
| Sparse KV | 非密铺 slot | 新账本 | ❌ **拒开** `sparse_kv_offload` |
| SFA/DSA/C8 旁路 | 稀疏/压缩写 | 漏钩假阴性 | ❌ **拒开** 对应 enable_* |
| Mamba / GDN / hybrid | 非 PA slot | 另案 | ❌ **拒开** |
| 投机解码 | 多写/回滚 | wave 搅动 | 未单独拒开；需与 reject 路径一致（后续） |

**一句话：** 只保证 **稠密本地逐 slot scatter**；复用/外置/稀疏/滑窗/SSM **未适配前全部拒开并打 warning**。

---

## 6. 适配 backlog 与代码量（解除拒开前）

| # | 项 | 拒开 tag | 适配内容 | 粗估 LOC |
|---|----|----------|----------|----------|
| A1 | Sliding / sink | `sliding_window` | L2 按窗口裁剪；淘汰 invalidate | ~300–600 |
| A2 | 旁路 scatter | `enable_sparse_sfa_c8` / `li_c8` / `dsa_cp` + P0-1 | 各写入口复用 audit 钩子 | ~150–400 |
| A3 | Store / PD 验收 | `kv_transfer` | Store 挂 load；PD 钩已有须验完再解禁 | ~80–200（Store） |
| A4 | Offload | `kv_offload` / `recompute_cpu_offload` | 传 `num_tokens`；补钩 | ~50–150 |
| A5 | reshape finding 不丢 | （缺陷，非开关） | 禁止只 degrade 不报 | ~80–150 |
| A6 | Load 内容弱校验 | 随 A3/A4 | dump/hash/对拍 | 400+ |
| A7 | Sparse | `sparse_kv_offload` | 新账本 | 1.5k–3k+ |
| A8 | Mamba/GDN | `hybrid_mamba_*` / `mamba_cache_mode` | 状态机另案 | 1k–2k+ |
| A9 | Prefix | `prefix_caching` | LOAD 语义 + L2 unverified 策略验收 | 含在 A3/A6 |

建议顺序：门控已落地 → P0-1/P0-3 → Store/`num_tokens` → 再考虑解禁 prefix/PD/offload → sliding → sparse/SSM 单独立项。

---

## 7. 与 PR-A / PR-B 分工

| PR | 内容 |
|----|------|
| **A** `feat/runtime-guard-config` | 控制面：config、输出侧 detector、inject、dump_kv、report；**无** online KV meta |
| **B** `feat/runtime-guard-analysis` | 离线 analysis / skills（基于 A） |
| **C** 本分支 | KV meta + invariant + 钩子 + **compat 拒开** + 本文档 |

---

## 8. 关键开关与文件

| 配置 | 作用 |
|------|------|
| `report.kv_audit` / `block_state` | 边钩与块状态 |
| `invariant.kv_state` / `kv_slot_order` / `slot_consistency` | L1/L2 校验 |
| `kv_meta_compat.apply_kv_meta_compat` | bind/`_sync_kv_audit` 时探测并强制关 |

| 文件 | 角色 |
|------|------|
| `kv_audit.py` | 边钩总线 |
| `kv_block_meta.py` | 状态机与 slot token 账本 |
| `invariant/slot_consistency.py` 等 | 对账与 incident |
| `kv_meta_compat.py` | 特性拒开 |
| `KV_META_TODO.md` | P0/P1/P2 + 适配表 |
