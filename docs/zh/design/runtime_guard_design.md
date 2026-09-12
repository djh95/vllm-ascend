# runtime_guard 方案说明（vllm-ascend）

> 运行时异常检测与 incident 处置控制面。  
> 代码根目录：`vllm_ascend/runtime_guard/`  
> 配置模块：`vllm_ascend/runtime_config/`

## 1. 组件与流程

| 组件 | 模块 | 职责 |
|------|------|------|
| Runtime Config | `runtime_config/config.py`（`RuntimeConfig`） | 一份 JSON；可选热更新（启动项控制周期） |
| Detector | `detector/` | 异常检测，产出 `Incident` |
| Action | `action/` | 异步处置：`report`、`dump_kv`、`set_log_level` |
| Report | `report.py`（`ReportWriter`） | 异常短报告落盘到 `runtime/report/` |
| KV dump | `kv_cache_reader.py`（`KvCacheReader`） | 按请求 block 做 native D2H，异步写 `.pt` |
| Processor | `processor.py`（`RuntimeGuardProcessor`） | runner 侧编排：bind / `sync_for_step` / sample hooks / `_handle_alert` → ActionExecutor；`[SamplingMeta]` 在 after-sample 以 DEBUG 输出（靠日志级别，无 JSON 开关） |
| Request state | `request_state.py`（`RequestGuardStore`） | per-req 共享态；`mark_finished` 后延迟 `clear` |
| I/O snapshot | `io_snapshot.py`（`RequestIoSnapshotManager`） | report I/O 视图（normalize→Store + `snapshot`） |
| Quota | `quota.py`（`DumpQuota`） | 自动 dump 次数上限与 cooldown |
| Rank gate | `rank_gate.py` | 检测 / report：last-PP TP0；dump：last-PP 全部 TP |

对外入口：`from vllm_ascend.runtime_guard import RuntimeGuardProcessor`

```text
additional_config
  ├─ runtime_config_path / runtime_config_reload_interval
  └─ AscendConfig.runtime_config (RuntimeConfig)
         │
Worker: RuntimeGuardProcessor.bind(runner)
  execute_model 入口：runtime_guard.sync_for_step()
  ├─ dump + config 同步             # 见下：broadcast 组 hitchhike / file 模式分路
  ├─ refresh_config 应用            # 热更关则立刻 return
  └─ wave / manual_trigger 刷新
         │
  采样路径：
  check_before_sample (logits_finite: isfinite + .item；hit 则解析后入队 Incident)
  check_after_sample / get_output
    ├─ append IO、wave stamp
    ├─ logits_finite check_deferred（drain，可 dump）
    ├─ token_repeat / substring 快照 → ActionQueue（结束不等）
    └─ reap（cpu_jobs>0 则推迟 Store.clear）
         │
  Incident → ActionExecutor.handle()          # last-PP TP0
    ├─ sync_only: set_log_level
    ├─ prepare report → async commit（ActionQueue）
    └─ prepare dump_kv：只 queue_kv_dump（本步不 D2H）
         │
  下一拍 last-PP 全部 TP：sync_for_step
    ├─ config 已有 broadcast 组（PP=1）：all_reduce([config_due, dump_due])
    │    任一 due 则一次 broadcast_object（config 和/或 dump jobs）
    └─ file 模式 / 热更关：last-PP TP all_reduce(has_job)，有才 broadcast_object
       然后全部 TP（含 TP0）dump 本 rank head 分片
```

### Report / dump_kv 流水线

同一 incident 的 `on_trigger` 含 `report` 与 `dump_kv` 时，executor **先 enqueue report、再排队 dump_kv**（本步不 D2H）。  
`dump_kv` 默认只 dump 该请求占用的 **paged block**（`block_ids`），不是整池 KV。  
成功排队后 last-PP TP0 在 `{dump_root}/<type>/<req_id>/wave_<N>/request_info.json` 写请求元信息（字段策略与 report 一致：counts 必有，token ids 受 `report.save_sensitive_info` 控制）。  
`.pt` 保持 `[n_sel_blocks, block_size, …]`。

写盘前按 **leader 单卡估计 × `tp_size` + `dump.free_headroom_bytes`（默认 5GiB）** 查目标目录空闲空间（last-PP 全 TP 都会写盘），不足则跳过排队（不扣 auto quota）。`try_consume` 成功后若 queue 失败或 drain 整 arm 未 D2H 会 **refund**（还次数并清除 cooldown）。  
after-sample CPU 检测在 ActionQueue 上稍后才告警。任意路径上若请求已 `finished` 或已 reap，**一律跳过 dump**（KV 可能已释放/复用）。`logits_finite` 在 before-sample 已 `.item()` 并（hit 时）解析入队；`check_deferred` 仍在 `get_output` / after-sample drain。

### dump_kv 的 rank 覆盖（last PP × 全部 TP）

检测只在 last-PP TP0。其它 TP 没有 incident，不能独立决定 dump。用 last-PP **本 stage 的 TP 组** 把名单带过去。

| 谁 | 做什么 |
|----|--------|
| last PP + TP0 | 检测、写 report、`queue_kv_dump` 记下 `{req_id, ...}`（本步不 D2H） |
| last PP + 全部 TP（含 TP0） | **下一拍** `sync_for_step`：收到 `req_id` 后各自 dump **本 rank** 分片 |
| 其它 PP stage | **不 dump**（那些层的 KV 不在本覆盖里） |
| 其它 DP replica | 不参与（没有这份请求的 KV） |

```text
step N     last-PP TP0  检测 → 排队 {req_id, ...}
step N+1   last-PP 全部 TP  sync_for_step
           有 config broadcast 组：搭同一趟 all_reduce + 按需 broadcast_object
           否则 last-PP TP：all_reduce(has_job)，有 job 才 broadcast_object
           全部 TP（含 TP0）按名单读本地 block 表 D2H
```

同一拍排队、下一拍一起 dump：各 TP 看到的是同一 wave 的 KV，D2H 路径只有一条。代价是多等一拍 decode；下一拍请求已 finish 或本地 block 表为空则跳过。  
因 finished/reaped 跳过时，last-PP TP0 写 `{dump_root}/<type>/<req_id>/dump_skipped_finished.json`。

落盘：

```text
{dump_root}/{incident_type}/{req_id}/wave_{N}/dp{D}_tp{T}_pp{P}_cp{C}/{req_id}_{layer}_req.pt
```

拼 last-PP 下各 `tp*` 目录得到该 stage 的完整 head 切分；**没有**其它 `pp*` 目录是预期行为。

代价：config 已有 collective 时，dump 只多一个 `all_reduce` 上的 bit，无 job 且 config 未到期则不做 `broadcast_object`。File 模式（PP>1）config 各 rank 读 JSON；dump 在 last-PP TP 组自己 `all_reduce(has_job)`。不改 `SchedulerOutput`，也不在 PP 组上 collective。

## 2. Runtime Config

### 2.1 路径

| 项 | 说明 |
|----|------|
| 默认文件 | `<cwd>/runtime/config/runtime_config.json` |
| 显式路径 | `additional_config.runtime_config_path` |
| 报告根目录 | 默认 `<cwd>/runtime/report`；可 `runtime_report_dir` 覆盖 |
| 示例 | `vllm_ascend/runtime_config/templates/runtime_config.example.jsonc` |

### 2.2 同步模式 `sync_mode`

| 值 | 行为 |
|----|------|
| `broadcast`（默认） | EngineCore leader 读 JSON，在 **inner DP 组** 内 `all_reduce([config_due, dump_due])` + 按需 `broadcast_object`；无 inner DP 组 / PP>1 时各 rank 轮询本地可读路径（dump 改走 last-PP TP 组） |
| `file` | 各 rank 轮询 `runtime_config_path`（共享盘或每节点副本） |

**注意**：配置热更 **不跨 DP replica 做全 world collective**。多 DP 时每个 EngineCore 各自维护可读 JSON。

### 2.3 热更新

- 生效开关：`additional_config.runtime_config_reload_interval > 0`（进程启动时设定；JSON 内 `reload_interval_seconds` 仅作展示）。
- `interval = 0`：启动后配置静态，仅保留启动 overlay 与一次性 `manual_trigger`。
- 热更失败（ malformed JSON）：保留旧配置，服务继续。

合并顺序：启动时 `defaults ← additional_config.runtime_config`（覆盖写盘）；热更 `defaults ← JSON`。

### 2.3.1 启动落盘

启动只合成一次 effective（defaults + overlay + ctor seeds），writer 在 `_bootstrap(persist=True)` / `ensure_persisted()` 时**整份覆盖**已有 JSON。不再读盘合并、不再对显式路径 skip rewrite / dump_dir backfill。热更仍只读 JSON。

### 2.4 JSON 顶层结构

| 段 | 作用 |
|----|------|
| `sync_mode` | 配置同步方式 |
| `actions.defaults.on_trigger` | 未指定时的默认 action 列表 |
| `dump` | 自动 dump 配额、`manual_dump`（`manual_trigger` 为加载别名） |
| `detector.*` | 各 detector 开关与阈值；可 per-type 覆盖 `on_trigger` |
| `report` | 报告字段、敏感信息、block 元数据 |
| `log` | 运维日志开关（不落 report JSON） |
| `ascend_log` | Ascend 模块日志级别 |

字段详解见 [runtime_config 配置参考](../../source/user_guide/configuration/runtime_config.md)。

## 2.5 日志与 UCM 劫持

Ascend 容器（slime/CANN）自带 UCM（Unified Compute Management）日志框架：`ucm_patch.pth` 在 Python 启动时挂 import hook，`import vllm` 结束时**无条件**把 `vllm.logger.init_logger` 替换为 UCM 版（`ucm.logger.init_logger`，与 `ENABLE_UCM_PATCH` 无关）。凡经 `vllm.logger.init_logger` 创建的 logger 都会变成 UCM logger：

- 级别/格式在 C 扩展（`ucmlogger`）内实现，Python 侧 `setLevel` / `Formatter` 全失效，级别锁死 INFO（DEBUG 被 C 库硬丢弃，无 API / 环境变量可改）。
- UCM logger 不注册进 stdlib `logging.Logger.manager.loggerDict`，`logging.getLogger("vllm_ascend...")` 拿到的是另一个空 logger。

原 `init_logger_ascend` 委托 `vllm.logger.init_logger`，导致 runtime_guard / detector 等模块日志被劫持：`ascend_log` / `set_log_level` 分模块调级对它们静默失效，UT 用 `caplog` 也断言不到（`test_v12` 连挂三次）。

**绕过**：`init_logger_ascend` 改为直接 `logging.getLogger(name)` + 复用 vllm 的 `_METHODS_TO_PATCH` 挂 `info_once`/`debug_once`/`warning_once`，不再经被 patch 的 `vllm.logger.init_logger`。结果：Ascend 日志回到 stdlib 树，`setLevel` / `VLLM_LOGGING_LEVEL` / `apply_ascend_log_level` 重新生效，`caplog` 可捕获。注意仅覆盖走 `init_logger_ascend` 的模块；vLLM 自身运行时日志仍受 UCM 影响（不在本仓库控制范围）。

## 3. Detector

检测在 **last PP + TP0** 上运行（与 `is_action_leader_rank` 同 rank）。  
Async scheduling 的 `unique_reply_rank` 只把 output rank（TP0）的返回值送入 `enqueue_output` / `get_output`；在其它 TP rank 上强制 `get_output()` 会卡住下一步 `execute_model` 的 TP collective。  
Report 只在 last PP + TP0 写。`dump_kv` 的 rank 覆盖见上一节（last PP × 全部 TP）。

| incident_type | 钩子阶段 | 说明 |
|---------------|----------|------|
| `spec_acceptance` | after spec | 投机解码接受率异常 |
| `output_substring` | after sample | 输出 token 子序列匹配 |
| `token_repeat` | after sample | 滑动窗口复读分数 |
| `logits_finite` | before sample | logits NaN/Inf：每步 `isfinite` + `.item()`；仅 hit 时当场解析 bad row / indices / kind，入队 host `Incident`；`get_output` / after-sample 再 drain（便于 dump）。 |

共享行为：停检由 ``report.max_per_req`` 写满触发（默认 ``actions.defaults.on_trigger`` 含 ``report``）。  
`output_substring` / `token_repeat` 在 `ActionQueue` 上跑；`logits_finite` 在 before-sample 做 `.item()` 门闩，hit 当场解析后入队，after-sample drain。

> 在线 KV / position meta 检测器见后续 PR。

各 detector 可通过 nested `on_trigger` 覆盖 action，例如：

```json
"token_repeat": {
  "enabled": true,
  "on_trigger": ["report", "dump_kv"],
  "dump_kv": { "scope": "request" }
}
```

## 4. Action

| name | sync_only | 说明 |
|------|-----------|------|
| `report` | 否 | 写 `runtime/report/<type>/report_*.json`；记录 metrics |
| `dump_kv` | 否 | last PP × 全部 TP：各写 `{dump_root}/<type>/<req_id>/wave_<N>/<rank_tag>/*.pt`（见「dump_kv 的 rank 覆盖」） |
| `set_log_level` | 是 | 即时调整 Ascend 日志级别 |

`dump_kv` 配置（per detector）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `scope` | `request` | `request`：仅 incident 请求；`all_requests`：当前 batch 各请求（arm 自举 `iter_local_request_rows` + 每 req `block_ids_for_request`）。**`manual_trigger` 固定 `all_requests`，忽略配置。** |

Quota：`dump.auto_max_times > 0` 启用自动 dump 配额；`dump.auto_cooldown_seconds` 控制冷却。  
同一次 `dump_kv` prepare 共用一个 `arm_id`：drain 时该 arm 全部未 D2H 才 refund 一次（还次数并清 cooldown）；其后异步 `torch.save` 失败不退额度。  
`manual_dump` / `manual_trigger` 不走 auto quota（见运维文档）。

## 5. Report

落盘：`{report_dir}/<incident_type>/report_<timestamp毫秒>[_<req_id>]_pid<pid>.json`（仅 last-PP TP0）。

常见字段：`incident_type`、`req_id`、`rank`、`detail`、`dump_attempted`、`dump_arm_wave`、`dump_dir`、`dump_count` / `dump_max_times`。  
同 `(incident_type, req_id)`：`report.max_per_req`（**默认 1**）限制份数；**写满后停止该 req 的全部检测**。多份之间按 **wave** 退避（首间隔 64，之后翻倍）。默认 `on_trigger` 含 `report`；若 detector 覆盖掉 `report`，则不会因写满而停检。  
`report.save_sensitive_info=true` 时持久化 prompt/output token ids（可截断、`decode_token_ids` 控制是否解码文本）。

## 6. Model Runner 接入

| Runner | bind | 主要钩子 |
|--------|------|----------|
| v1 | `model_runner_v1.py` 构造 | `sync_for_step`、`run_sample_phase`（after_sample 等）、pre-sample wrap、async `AscendAsync*` |
| v2 | `worker/v2/model_runner.py` 构造 | 同上（native `dump_kv`；与 msprobe dump 解耦） |

Idle DP：`worker.execute_dummy_batch` 调 `sync_for_step(allow_arm=False)`，与 busy rank 对齐配置热更。

v1/v2 在 `compute_logits` 外包一层以插入 `check_before_sample`（`runner_bridge.wrap_compute_logits_for_pre_sample`）。

## 7. 非 worker 与多 engine

- **API / EngineCore**：每个 EngineCore 进程各自 `RuntimeGuardProcessor.bind`；配置 writer 为 per-EngineCore leader。
- **多 DP**：每个 DP replica 独立 JSON（或共享可读路径 + `sync_mode=file`）；不要用跨 idle DP 的 world collective 做热更。

## 8. 相关文档

- 运维与排障：[runtime_guard_ops.md](./runtime_guard_ops.md)
- 用户功能指南：[runtime_guard.md](../../source/user_guide/feature_guide/runtime_guard.md)
- 配置字段表：[runtime_config.md](../../source/user_guide/configuration/runtime_config.md)
- 启动项：[additional_config.md](../../source/user_guide/configuration/additional_config.md)
