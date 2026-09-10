# Runtime Guard 需求文档

> 产品名：**Runtime Guard**（代码：`vllm_ascend/runtime_guard` + `vllm_ascend/runtime_config`）  
> 类型：需求（What / Why / 验收），不是方案。实现见 [runtime_guard_design.md](./runtime_guard_design.md)。  
> 状态：全量产品需求（含 KV meta）。落点 `feat/runtime-guard-kv`（PR-C）。  
> **不绑** PR-A 实现细节（`logits_finite` 当 detector、全部 Guard 钉 TP0、LPT 散布、非 TP0 强制 `get_output`）。  
> 读者：产品 / 值班 / 社区 reviewer / 实现同学

---

## 1. 背景与问题

vLLM-Ascend 线上会出现**间歇、难复现**的质量与数据面问题：复读、乱码、logits NaN/Inf、投机解码接受率漂移、KV 块写序/slot 与推理序列不一致。现有手段不够：

| 现状 | 缺口 |
|------|------|
| 停服、加日志、改代码再复现 | 现场已过；无法在不停服下抓同一请求的 KV |
| msprobe `dump_config` | 精度 dump 工具链，不是 on-call 控制面；v2 曾静默不 dump；与 Guard 正交 |
| 只看最终文本 | 看不到 decode 过程中的 token 窗口、block 账本、哪一层 KV 先坏 |

需要一层**默认可关、可热开、可出报告、可选 dump KV** 的运行时护栏，挂在 model runner 上，不改模型、不替代 msprobe。

---

## 2. 目标与非目标

### 2.1 目标

1. **在线发现**：decode / spec / 采样前 logits / KV 写路径上发现异常，产出结构化 `Incident`。
2. **可处置**：命中后可写 JSON report、按请求 dump paged KV（native D2H）、临时提日志级别。
3. **可运维**：一份 JSON 控制面；启动后可热更新开关与阈值；手动触发抓现场。
4. **默认可上**：默认全部检测关闭、热更关闭；生产「只 bind、不做事」路径必须接近零开销。
5. **可分析**：report + 可选 `.pt` 足够离线对拍（buggy vs 干净服务）。
6. **v1 / v2 同源**：`NPUModelRunner` v1 与 v2 都 bind 同一套 Processor；`dump_kv` 与 msprobe dump 解耦。

### 2.2 非目标

- 不替代 msprobe / `dump_config` 精度 dump（activation、aclgraph dump）。
- 不默认做整池 KV 或全层浮点 hash（成本高；内容对拍走离线 analysis）。
- 不跨 DP replica 做 world collective 热更。
- 不保证检测 100% 无假阴性（L2 slot meta 有 cap；未适配的 KV 特性拒开）。
- 不把 Guard 做成通用 DFX 框架；不接管 scheduler / 引擎调度。
- `finish` 不是异常检测，是生命周期传感器（每请求一份 report，不占 dump 配额）。

---

## 3. 用户与场景

| 角色 | 要做什么 |
|------|----------|
| 值班 / 现场 | 热开 `token_repeat` / `logits_finite`，看 `runtime/report/`；必要时 `dump_kv` 或 `manual_trigger` |
| 开发 | 用 `output_substring` 盯固定乱码模式；用 KV invariant 查块写错 |
| 性能 | 确认 T1（默认关）相对无 Guard 代码接近零差；热更壳可接受 |
| 社区合入 | 默认关、不改 CLI 表面、坏 JSON 不拖死 serving |

### 典型场景

| ID | 场景 | 期望 |
|----|------|------|
| S1 | 偶发复读 | 开 `token_repeat` + `report`；同一请求只告一次 |
| S2 | 怀疑 KV 写坏 | 适配路径上开 `kv_audit` + slot invariant + `dump_kv`；未适配特性必须拒开，不能默默假阴性当「正常」 |
| S3 | 抓一次现场 | `reload_interval>0` + `manual_dump` / `manual_trigger`；跳过 auto quota 与 input filter |
| S4 | 只盯某类 prompt | `input_filter` include/exclude prefix；手动触发不受 filter 限制 |
| S5 | 投机解码异常 | 开 `spec_acceptance`；非 spec 服务无额外语义负担 |
| S6 | 请求级审计 | 开 `detector.finish`；可带 token 计数（敏感字段另开） |

---

## 4. 功能需求

编号约定：`FR-x.y` 为必须；`FR-x.y-opt` 为可选/分期。验收口径见 §7。

### 4.1 接入与默认行为

| ID | 需求 |
|----|------|
| FR-1.1 | v1、v2 model runner 构造时 `RuntimeGuardProcessor.bind`；不配置 extra 时走默认 JSON / 内存 defaults。 |
| FR-1.2 | 默认：所有 detector / invariant `enabled=false`；`runtime_config_reload_interval=0`（启动后静态）；`dump.auto_max_times=0`。 |
| FR-1.3 | 未开任何检测且热更为 0 时，serving 语义与未 bind 前一致：`temperature=0` 输出 bit-identical（功能隔离）。 |
| FR-1.4 | Guard 失败（检测异常、写盘失败、热更 JSON 坏）不得中断推理；软失败打日志，保留上一份合法配置。 |

### 4.2 配置控制面

| ID | 需求 |
|----|------|
| FR-2.1 | 启动项：`runtime_config_path`、`runtime_config_reload_interval`、`runtime_config` overlay、`runtime_report_dir`（及可选 dump 目录）。 |
| FR-2.2 | 合并顺序：`defaults ← runtime_config_path 文件 ← additional_config.runtime_config`。热更**只重读 JSON 文件**，不再套启动 overlay。 |
| FR-2.3 | `reload_interval=0`：不轮询；仅启动配置 + 一次性控件在「未消费完」范围内生效。`manual_*` 与 `print_input_token_ids_once` **要求 interval>0**。 |
| FR-2.4 | `sync_mode=broadcast`：EngineCore leader 读文件，在 inner DP 组内同步；无 inner DP 则各 rank 读本地可读路径。`sync_mode=file`：各 rank 轮询路径。 |
| FR-2.5 | 热更**禁止**跨 DP replica 的 world collective。多 DP：每 EngineCore 一份可读 JSON（或共享盘 + `file`）。 |
| FR-2.6 | Idle DP dummy batch 仍须对齐配置热更（`sync_for_step(allow_arm=False)`），且不得在 dummy 上消费 `manual_trigger`。 |
| FR-2.7 | 非法 / 未知键、类型错误：拒绝该次刷新并保留旧配置；`invariant.*` 不得写在 `detector.*` 下。 |

### 4.3 检测：Detector

输出侧 after-sample **必须**在 last PP + **TP0**：async `unique_reply_rank` 只把 output rank 的返回值送进 `enqueue_output` / `get_output`。禁止为散布检测在其它 TP rank 上强制 `get_output()`（会把采样 D2H 塞进下一步 TP collective）。  
不要求 LPT / `detector_placement`；那是可选实现，不是产品约束。

| ID | `incident_type` | 阶段 | 需求 |
|----|-----------------|------|------|
| FR-3.1 | `spec_acceptance` | after spec | 滑动窗口上接受率 / 接受长度比越阈值则告警。非 spec 请求不误报。 |
| FR-3.2 | `token_logprob` | after sample | 窗口内 NaN / 稀有 / 乱码 / 重复。依赖 msprobe `ILLDetector`；**不可用则强制 `enabled=false` 并说明**，不得空转假装在检。开时由 Guard **强制 top-k logprobs**（不依赖用户先开 logprobs API）；该强制有可测量的 sample 开销，须可单独关闭。 |
| FR-3.3 | `output_substring` | after sample | `patterns`（字符串或 token-id 列表）在累计输出上匹配；`match_prefix` 控制仅前缀 vs 任意子序列。 |
| FR-3.4 | `token_repeat` | after sample | 滑动窗口复读分数；支持 `min_tokens`、`consecutive_hits`、`ignore_token_ids`。不依赖 logprobs。 |
| FR-3.5 | `finish` | request reap | 每个请求结束一份 **非 ill** report；不占 auto dump 配额；不触发 `stop_after_alert` 语义。 |

共享：

| ID | 需求 |
|----|------|
| FR-3.6 | `detector.stop_after_alert` 默认 true：同一 `req_id` 首次 ill alert 后不再对该请求 detect（含 invariant）。 |
| FR-3.7 | after-sample 顺序固定：`token_logprob` → `output_substring` → `token_repeat`。 |
| FR-3.8 | 累计输出以 `RequestGuardStore` 为准（async 下 runner 侧可能是 `-1` placeholder）；normalize 为 `list[int]`，丢弃 `-1`。 |
| FR-3.9 | `input_filter` 在 detect 前过滤；**不**作用于 `manual_trigger`。 |

### 4.4 检测：Invariant（soft-assert）

不进 Detector 注册表；用 `check_scope`（谁检谁写 report，不跨 rank 传 Incident）。  
KV 账本挂在 **Worker 写路径**（`reshape_and_cache` / load / CoW 等），与 after-sample 物化解耦：会写 KV 的 rank 都要记账，不得跟 `needs_sample_phase_hooks` 绑死。

| ID | 配置名 | incident | 需求 |
|----|--------|----------|------|
| FR-4.1 | `logits_finite` | `logits_finite` | 采样前 logits 含 NaN/Inf 则告警。不受 KV 特性拒开影响。 |
| FR-4.2 | `slot_consistency` | `kv_slot_token` | slot meta token vs 推理序列；note_kv 首次前缀 + finish 全前缀。 |
| FR-4.3 | `kv_slot_order` | `kv_slot_order` | 块内 slot offset 非连续（`gap` / `wrong_start`）。 |
| FR-4.4 | `kv_state` | `kv_state` | sealed 后再写：LOAD 非 0 续写在 `output_len>0` 才告警；FILL 非 0 始终告警且跳过该批记账。 |

KV 门控：

| ID | 需求 |
|----|------|
| FR-4.5 | 写路径未挂钩的特性（sparse / SFA / DSA / sliding / Mamba 等）**拒开整个 meta**，不得静默当通过。 |
| FR-4.6 | **产品门控**（以 [KV_META_DESIGN.md](../../../vllm_ascend/runtime_guard/KV_META_DESIGN.md) §9 为准，不以当前 compat 代码收窄）：稠密本地 PA 允许 L1+L2（L2 建议默认关）；`prefix_caching` / PD / `kv_transfer` / offload **允许 L1**，L2 关或标 unverified。当前实现若仍一律拒开 L1+L2，视为未达需求。 |
| FR-4.7 | 不做「非 load 的整块覆盖」独立 incident，直到挂上第二条整块写路径。 |
| FR-4.8 | `kv_audit` / `note_kv_block_writes` 不得因任意输出侧 detector 打开而改变 sample 热路径（不得因此 wrap async 或在非 TP0 上 `get_output()`）。 |

### 4.5 Incident 与 Action

| ID | 需求 |
|----|------|
| FR-5.1 | 默认 `on_trigger=["report"]`；各 detector/invariant section 可覆盖。合法 action：`report`、`dump_kv`、`set_log_level`。 |
| FR-5.2 | `report`：`{report_dir}/<incident_type>/report_<timestamp>_<req_id>.json`。默认只写计数；`save_sensitive_info` 才持久化 token ids（可截断、可选 decode）。 |
| FR-5.3 | `dump_kv`：只 dump **该请求 paged block**（`scope=request`）；`block_ids` 空且未 `dump_all_blocks` 则**跳过**，禁止拖全池。与 report 同 incident 时先 enqueue report 再 D2H。 |
| FR-5.4 | Auto dump 受 `auto_max_times` + `auto_cooldown_seconds` 限制；`0` 表示不配 auto 配额（自动 dump 不发生）。 |
| FR-5.5 | `manual_dump` / `manual_trigger`：incident_type=`manual_trigger`；跳过 auto quota、cooldown、input filter。`true`=持续到热改 false；正整数=后续 N 个有 scheduled tokens 的 wave。 |
| FR-5.6 | `set_log_level` 同步生效；report / dump 走异步队列，不阻塞采样热路径超过必要的 prepare（D2H 本身按块数一次性尖峰可接受）。 |
| FR-5.7 | 写盘 rank：last PP + action leader（通常 TP0）。检测 rank 与写盘 rank 可以不同，但 report 必须能对应到 `req_id`。 |

### 4.6 手动与观测

| ID | 需求 |
|----|------|
| FR-6.1 | `print_input_token_ids_once`：下一个有 prompt 的真实 wave 打印 token ids 并给出 filter 示例，然后清 flag。 |
| FR-6.2 | `log.print_sampling_meta` / `print_output_on_finish`：运维日志，不写入 report JSON。 |
| FR-6.3 | 提供离线 analysis 入口（summarize / correlate / verify / compare KV），与在线路径分离。 |

### 4.7 与 msprobe 的边界

| ID | 需求 |
|----|------|
| FR-7.1 | Guard `dump_kv` 不依赖 `dump_config` / PrecisionDebugger。 |
| FR-7.2 | 未设 `dump_config` 时不得 import / 初始化 msprobe dumper。 |
| FR-7.3 | `token_logprob` 是唯一允许硬依赖 msprobe ILLDetector 的 detector；缺失则关该项，其余 detector 仍可用。 |

---

## 5. 非功能需求

| ID | 类别 | 需求 |
|----|------|------|
| NFR-1 | 正确性 | 默认关 + 热更关：不改变采样结果。开检测：只读/旁路，不改 logits、不改 KV 内容（`kv_state` FILL 违规可跳过**账本**应用，仍不改设备 KV）。 |
| NFR-2 | 性能 T1 | Guard 代码已加载、全部关、reload=0：相对「无 Guard 代码」吞吐比 **T1/T0 ≥ 0.999**（C1）。 |
| NFR-3 | 性能 T2 | 仅热更壳（reload>0、检测全关）：**T2/T1 ≥ 0.999**（C2）。到期才做 collective，非每 step all_reduce。 |
| NFR-4 | 性能 T3 | 轻量检测开（repeat / substring / spec / logits_finite；logprob 视 msprobe）：**T3/T2 ≥ 0.990**（C3）。 |
| NFR-5 | 热路径 | 全关时 `sync_for_step` / `refresh_config` 早退；detector=None 等价路径禁止隐式 GPU `.tolist()` / D2H。 |
| NFR-6 | 资源 | `dump_kv` 尖峰与 blocks×layers 成正比；必须有 quota，避免打满盘。L2 slot meta 有 cap，淘汰后允许假阴性。 |
| NFR-7 | 安全 | 默认不落 prompt/output 原文；`save_sensitive_info` 显式打开。 |
| NFR-8 | 兼容 | 覆盖 v1 与 v2 runner；PP / TP；idle DP。EC encoder-only、未适配 KV 特性见缺口。 |
| NFR-9 | 可测 | 核心 detector / invariant / 配置校验 / 热路径早退有 UT；perf 对照见 `tests/perf/runtime_guard/README.md`。 |

---

## 6. 约束与假设

- 推理进程可写 `{cwd}/runtime/` 或用户指定的 report/config 路径。
- Last PP 才能看到完整采样；非 last PP 不跑**输出侧** detector。
- After-sample 的 CPU token 只在 TP0 被 RPC 物化；KV 账本不走这条路径。
- Async scheduling 下累计 token 必须走 Store，不能信 runner 的 `-1` 占位。
- `token_logprob` 需要 msprobe 且可能强制 top-k，对吞吐有额外成本，须可单独关闭。
- JSON 热更不是事务：一次坏文件不影响 serving，但可能延迟一次配置生效。

---

## 7. 验收标准

### 7.1 必须通过

1. **默认关**：无 additional-config 起服，不写 report、不 dump、不 import msprobe dumper。
2. **热开**：interval>0 时改 JSON 打开 `token_repeat`，下一轮 reload 后复读请求出现 report；改回 false 后停止新告警。
3. **隔离**：`temp=0` 下 T0/T1/T2（及可测的 T3）输出一致（C4）。
4. **配额**：`auto_max_times=1` 时第二次 auto `dump_kv` 被挡；`manual_trigger` 仍能 dump。
5. **安全 dump**：无 block_ids 且 `dump_all_blocks=false` 时不写全池 KV。
6. **msprobe 缺失**：`token_logprob.enabled=true` 被强制 false；`token_repeat` 仍可告警。
7. **坏 JSON**：serving 继续，配置停留在上一份合法值。
8. **stop_after_alert**：同一请求不会刷屏同一类 ill report。
9. **v1 与 v2**：上述 1–8 在两种 runner 上行为一致（KV dump 均为 native，不走 msprobe）。
10. **after-sample rank**：async + TP>1 时非 TP0 不强制 `get_output()`。
11. **KV 拒开**：未挂钩特性开 L1/L2 时必须明确关掉并打日志，不得空转 PASS。

### 7.2 性能门（NPU 实测，以 perf README 为准）

| 门 | 对照 | 阈值 | 备注 |
|----|------|------|------|
| C1 | T1 vs T0 | ≥ 0.999 | 默认关零成本；当前未测完则不得宣称「零开销」 |
| C2 | T2 vs T1 | ≥ 0.999 | 热更壳 |
| C3 | T3 vs T2 | ≥ 0.990 | 输出侧轻量检测开（不含强制 top-k logprob、不含非 TP0 同步 D2H） |

### 7.3 本阶段不阻塞合入（须文档化）

- EC transfer / encoder-only dump 窗口（v2 尚未支持 EC）。
- 未挂钩 KV 特性上的 L1/L2（必须拒开，而不是「检了但永远不中」）。
- prefix / PD / offload 上 **L1 可开** 若当前代码仍拒，记为实现缺口，不改需求。
- C1 实测未完成前，合入说明里不得写死「bind 零开销」。

---

## 8. 分期

| 阶段 | 范围 | 状态 |
|------|------|------|
| A 控制面 | 配置、热更、输出侧 detector、inject、report、dump_kv | 可独立合入（`feat/runtime-guard-config`） |
| B 离线分析 | report/KV 对拍脚本与 skills | 可独立合入（`feat/runtime-guard-analysis`） |
| C KV meta | 账本 + slot/state invariant + 特性门控（本需求 FR-4 / §9） | **本分支**（`feat/runtime-guard-kv`） |
| 后续 | checksum / 旁路 scatter 挂钩；C1 实测达标；更多 filter 类型 | 未承诺 |

---

## 9. 相关文档

| 文档 | 内容 |
|------|------|
| [runtime_guard_design.md](./runtime_guard_design.md) | 方案与数据流 |
| [runtime_guard_ops.md](./runtime_guard_ops.md) | 值班操作与排障 |
| [runtime_guard.md](../../source/user_guide/feature_guide/runtime_guard.md) | 用户指南 |
| [runtime_config.md](../../source/user_guide/configuration/runtime_config.md) | JSON 字段 |
| [KV_META_DESIGN.md](../../../vllm_ascend/runtime_guard/KV_META_DESIGN.md) | KV 两层检测、问题类型、目标门控 |
| [KV_META_TODO.md](../../../vllm_ascend/runtime_guard/KV_META_TODO.md) | P0–P2 与适配 backlog |
| `tests/perf/runtime_guard/README.md` | T0–T3 / C1–C4 |
| `vllm_ascend/runtime_config/templates/runtime_config.example.jsonc` | 配置样例 |
