# KV meta 方案说明（PR-C）

> 状态：WIP（`feat/runtime-guard-kv`）  
> 代码：`kv_audit` / `kv_block_meta` / `invariant/*` / `kv_meta_compat`  
> 需求（What / 验收，不绑 PR-A 实现）：[`docs/zh/design/runtime_guard_requirements.md`](../../docs/zh/design/runtime_guard_requirements.md)  
> 待办清单：[`KV_META_TODO.md`](./KV_META_TODO.md)  
> 记录：2026-09-09

本文汇总：**两层检测方案**、**三处数据面与责任划分**、**问题类型与能否解决**、**省 KV 特性支持与适配量**。  
产品口径以需求文档为准；与**当前代码**不一致处（例如 compat 仍一律拒开 L1）以代码与 `KV_META_TODO` 为实现现状，**不以现状收窄需求**。

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

**当前门控（代码，保守）：** 凡未适配特性一律 **拒开 L1+L2**（强制关 `kv_audit` / `block_state` / 上述 invariant）；仅 **稠密本地 PA `reshape_and_cache`** 可开。`logits_finite` 不受影响。

**目标门控（讨论结论，待改代码）：** 见 §9；prefix / PD / offload 应允许 **L1**，L2 默认关；sparse / Mamba / sliding 仍拒开。

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

**一句话（当前代码）：** 只保证 **稠密本地逐 slot scatter**；复用/外置/稀疏/滑窗/SSM **未适配前全部拒开并打 warning**。

**一句话（目标，§9）：** L1 可覆盖 prefix/PD/offload 的 **生命周期/写序**；内容完整性另挂 checksum；sparse/滑窗/SSM 仍拒。

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

**建议顺序（已按分层重排，指导实现）：**

1. **改门控到目标策略（§9）**：prefix/PD/offload 允许 L1；L2 默认关；sparse/Mamba/sliding 仍拒  
2. P0-1 / P0-3（旁路钩完整性、reshape finding 不丢）  
3. A4/`num_tokens` + A3 Store 钩齐 → load 路径可打 `BLOCK_SEALED_LOAD`  
4. **Checksum（P0-4 / 对齐社区 RFC）** 挂在 prefix hit / offload / PD recv——补 L1 盲区  
5. Sliding → Sparse / SSM 单独立项（不要塞进 L1 主线）

---

## 7. 与 PR-A / PR-B 分工

| PR | 内容 |
|----|------|
| **A** `feat/runtime-guard-config` | 控制面：config、输出侧 detector、inject、dump_kv、report；**无** online KV meta |
| **B** `feat/runtime-guard-analysis` | 离线 analysis / skills（基于 A） |
| **C** 本分支 | KV meta + invariant + 钩子 + **compat 门控** + 本文档；需求见 `runtime_guard_requirements.md` FR-4 / §9 |

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

---

## 9. 分层产品化与目标门控（讨论结论）

### 9.1 产品分层

| 层 | 定位 | 默认 | 建议落点 |
|----|------|------|----------|
| **L1** | 块生命周期 + 写序（W1/S2） | 可选 DFX / debug | **可做上游 vLLM**（与 backend 无关的状态机 + 钩子 API） |
| **L2** | slot token vs 序列 | **默认关**；debug / 小规模 | Ascend 或上游 debug；接受 cap 假阴性 |
| **Checksum** | 块内容弱校验 | 默认关 | 对齐社区 CRC/RFC；挂在 **load 路径**（prefix hit / offload / PD） |

原则：

- **先只做透 L1**；不要把 L2 绑进「生产可开」门槛。  
- L1 **不是 Ascend 专属**；Ascend 侧保留钩子与 compat，核心状态机可向上游收敛。  
- runtime_guard = **观测 / 取证 / soft-assert**，**不能替代** Scheduler 正确性修复与 UT。

### 9.2 目标门控（相对当前代码的变更意图）

| 特性 | 当前代码 | 目标 | 理由 |
|------|----------|------|------|
| 稠密本地 PA | 允许 L1+L2 | 同左；L2 仍建议默认关 | 基线 |
| `prefix_caching` | 拒开 L1+L2 | **允许 L1**；L2 off / unverified | 复用是常态；L1 仍抓密封后再写 |
| `kv_transfer` / PD | 拒开 | **允许 L1**；内容靠 checksum | 钩已有；LOAD 无 token 时 L2 易假阴性 |
| offload | 拒开 | **允许 L1**；补 `num_tokens` | 同 PD |
| sparse / SFA/DSA/C8 | 拒开 | **仍拒** | 旁路布局未适配 → 假阴性 |
| sliding / sink | 拒开 | **仍拒**（直到 A1） | L2 对账前缀变 |
| Mamba / hybrid | 拒开 | **仍拒** | 非 PA slot |

实现时：`kv_meta_compat` 应区分 **「拒开整个 meta」** vs **「仅强制关 L2 / 标 LOAD unverified」**，避免「最需要检测的生产路径反而开不了」。

### 9.3 根因层 vs 观测层

| 层 | 做什么 | 例子 |
|----|--------|------|
| **根因层** | 修调度/哈希/分配语义；加 UT 锁回归 | Scheduler invariant、hash 输入、LoRA/prefix bugfix |
| **观测层（本方案）** | 运行时 soft-assert + dump + 离线对拍 | L1/L2、inject、report、PR-B analysis |

「根因层补上」= 社区优先把 **正确性修复 + 单测** 合进主干；checksum / L1 是 **兜底与取证**，优先级通常低于修 bug，但高于「无任何运行时卫兵」。

---

## 10. 社区对照（vLLM 主干，指导取舍）

> 检索时点约 2026-09；以 GitHub 状态为准，合入后请更新本表。

| 方向 | 代表 | 状态（约） | 与本方案关系 |
|------|------|------------|--------------|
| Scheduler 不变量 / 正确性 | #38715、#41400 等 | 多 **closed 未清晰合入** | **根因层**；应跟进并在 Ascend 侧复现相关 UT |
| Hash / 索引正确性 | #53158（open）；LoRA hash #27211/#27577（merged）；#42125/#43996（open） | 混杂 | 补 **S1**；L1/L2 **不能替代** hash 修 |
| 内容 CRC / checksum | RFC #54363（open）；bit-flip + scheduling checksum 论文 | 研究 / RFC | 对齐 **P0-4**；挂 load 路径 |
| Connector 调试探针 | mean/std/nan 等 ad-hoc | 零散 | 弱于 checksum；可作临时 DFX |
| 本仓库 L1 块状态机 | PR-C | WIP | 社区暂无同构「块生命周期 soft-assert」；**有 upstream 空间** |

**优先级建议（省侧与上游共建时）：**

1. 跟进 / 移植 **Scheduler + hash 正确性**（根因）  
2. **L1** 做成可选、薄 API（upstream-friendly）  
3. **Checksum** 跟 RFC，不要自创第二套语义  
4. **L2** 保持 debug；勿宣传为生产默认  

---

## 11. 上游可接受性（L1）

| 维度 | 判断 |
|------|------|
| 通用性 | 高——不绑 NPU；状态机 + 写钩抽象即可 |
| 开销 | 低——每块少量标量；默认可关 |
| 与现有 RFC 关系 | 互补 checksum（内容）与 L1（生命周期），勿互相替代 |
| 合入预期 | **可选 DFX / RFC 档**，低于调度正确性修复；高于「无卫兵」 |
| 不该怎么推 | 不要绑 Ascend-only 巨型开关、不要默认强制 L2、不要声称替代 Scheduler UT |

**Upstream 切片建议：** `BlockMeta` 状态枚举 + `on_fill` / `on_load` / `on_free` 钩子协议 + 可选 `kv_state` soft-assert；backend 各自接线；配置默认 off。

---

## 12. 后续优化检查清单

改方案或合 PR 前，用本清单过一遍：

- [ ] 是否仍把 runtime_guard 当成 **观测层**，而不是调度正确性的替代？  
- [ ] 生产路径（prefix/PD/offload）能否开 **L1**？若不能，是门控过严还是钩未齐？  
- [ ] L2 是否仍默认关、并写清假阴性（cap / LOAD 无 token）？  
- [ ] 内容类故障（P1/P3）是否指向 **checksum / dump**，而不是扩 L2？  
- [ ] sparse / sliding / Mamba 是否仍 **拒开或单独立项**，未混进 L1 主线？  
- [ ] 新钩子是否覆盖所有写入口（P0-1），避免 C2 假阴性？  
- [ ] 若向上游推：API 是否 backend 中立、默认 off、与 RFC checksum 边界清晰？  
- [ ] 社区 Scheduler/hash 修复是否已评估能否直接复用 / 跟测？
