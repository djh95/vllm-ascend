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
| Processor | `processor.py`（`RuntimeGuardProcessor`） | runner 侧编排（bind / sync / detect / action） |
| Input filter | `input_filters.py`（`InputFilterManager`） | detect 前输入过滤（`manual_trigger` 不走 filter） |
| Request state | `request_state.py`（`RequestGuardStore`） | per-req 共享态；`mark_finished` 后延迟 `clear` |
| I/O snapshot | `io_snapshot.py`（`RequestIoSnapshotManager`） | report I/O 视图（normalize→Store + `snapshot`） |
| Quota | `quota.py`（`DumpQuota`） | 自动 dump 次数上限与 cooldown |
| Rank gate | `rank_gate.py` | 检测 rank 与 report/dump 写盘 rank 门控 |

对外入口：`from vllm_ascend.runtime_guard import RuntimeGuardProcessor`

```text
additional_config
  ├─ runtime_config_path / runtime_config_reload_interval
  └─ AscendConfig.runtime_config (RuntimeConfig)
         │
Worker: RuntimeGuardProcessor.bind(runner)
  execute_model 入口：runtime_guard.sync_for_step()
  ├─ refresh_config()              # 全 rank；热更关则立刻 return
  └─ wave / manual_trigger / input_filter 刷新
         │
  采样路径：
  check_before_sample (logits_finite / position_alignment)
  check_after_sample  (token_repeat / output_substring / token_logprob / …)
  note_kv_block_writes (block_kv)
         │
  Incident → ActionExecutor.handle()
  ├─ sync_only: set_log_level
  ├─ prepare report → async commit（ActionQueue）
  └─ prepare dump_kv (D2H) → async torch.save
```

### Report / dump_kv 流水线

同一 incident 的 `on_trigger` 含 `report` 与 `dump_kv` 时，executor **先 enqueue report、再做 KV D2H**，使写盘与设备拷贝重叠。  
`dump_kv` 默认只 dump 该请求占用的 **paged block**（`block_ids`），不是整池 KV。

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
| `broadcast`（默认） | EngineCore leader 读 JSON，在 **inner DP 组** 内 broadcast；无 inner DP 组时各 rank 轮询本地可读路径 |
| `file` | 各 rank 轮询 `runtime_config_path`（共享盘或每节点副本） |

**注意**：配置热更 **不跨 DP replica 做全 world collective**。多 DP 时每个 EngineCore 各自维护可读 JSON。

### 2.3 热更新

- 生效开关：`additional_config.runtime_config_reload_interval > 0`（进程启动时设定；JSON 内 `reload_interval_seconds` 仅作展示）。
- `interval = 0`：启动后配置静态，仅保留启动 overlay 与一次性 `manual_trigger` / `print_input_token_ids_once`。
- 热更失败（ malformed JSON）：保留旧配置，服务继续。

合并顺序：`defaults ← runtime_config_path ← additional_config.runtime_config`（启动 overlay）。

### 2.4 JSON 顶层结构

| 段 | 作用 |
|----|------|
| `sync_mode` | 配置同步方式 |
| `actions.defaults.on_trigger` | 未指定时的默认 action 列表 |
| `dump` | 自动 dump 配额、`manual_dump` / `manual_trigger` |
| `detector.*` | 各 detector 开关与阈值；可 per-type 覆盖 `on_trigger` |
| `report` | 报告字段、敏感信息、block 元数据 |
| `log` | 运维日志开关（不落 report JSON） |
| `ascend_log` | Ascend 模块日志级别 |
| `input_filter` | detect 前过滤与一次性 prompt 打印 |

字段详解见 [runtime_config 配置参考](../../source/user_guide/configuration/runtime_config.md)。

### 2.5 InputFilterManager

- 单例；`refresh_config` / 热更后从 JSON 重建 filter 链。
- `manual_trigger` **不**经过 filter。
- `print_input_token_ids_once`：下一次有 prompt 的真实 wave 打印 token ids 并生成 filter 示例，然后清 flag。

## 3. Detector

检测仅在 **last PP rank** 运行；async scheduling 下仅 **TP0** 运行 anomaly check。  
Report / KV dump 写盘 rank：`last PP` + `TP0`（`is_action_leader_rank`）。

| incident_type | 钩子阶段 | 说明 |
|---------------|----------|------|
| `spec_acceptance` | after spec | 投机解码接受率异常 |
| `token_logprob` | after sample | logprob 窗口异常（NaN / 稀有 / 乱码 / 重复） |
| `output_substring` | after sample | 输出 token 子序列匹配 |
| `token_repeat` | after sample | 滑动窗口复读分数 |
| `block_kv` | KV write | block 写 wave / writer 一致性 |
| `position_alignment` | before sample | position_ids 对齐 |
| `logits_finite` | before sample | logits NaN/Inf |

共享行为：`detector.stop_after_alert`（默认 `true`）— 同一请求首次 alert 后不再重复 detect。

各 detector 可通过 nested `on_trigger` 覆盖 action，例如：

```json
"token_repeat": {
  "enabled": true,
  "on_trigger": ["report", "dump_kv"],
  "dump_kv": { "scope": "request", "dump_all_blocks": false }
}
```

## 4. Action

| name | sync_only | 说明 |
|------|-----------|------|
| `report` | 否 | 写 `runtime/report/<type>/report_*.json`；记录 metrics |
| `dump_kv` | 否 | D2H + 写 `runtime/report/kv_cache/<type>/<req_id>/*.pt` |
| `set_log_level` | 是 | 即时调整 Ascend 日志级别 |

`dump_kv` 配置（per detector）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `scope` | `request` | `request`：仅 incident 请求；`all_requests`：当前 batch 各请求 |
| `dump_all_blocks` | `false` | `true` 时忽略 block 列表约束（慎用） |

Quota：`dump.auto_max_times > 0` 启用自动 dump 配额；`dump.auto_cooldown_seconds` 控制冷却。  
`manual_dump` / `manual_trigger` 不走 auto quota（见运维文档）。

## 5. Report

落盘：`{report_dir}/<incident_type>/report_<timestamp>_<req_id>.json`

常见字段：`incident_type`、`req_id`、`detail`（detector 专有）、`rank_tag`、`dump_count` / `dump_max_times`、`wave` 等。  
`report.save_sensitive_info=true` 时持久化 prompt/output token ids（可截断、`decode_token_ids` 控制是否解码文本）。

## 6. Model Runner 接入

| Runner | bind | 主要钩子 |
|--------|------|----------|
| v1 | `model_runner_v1.py` 构造 | `sync_for_step`、`run_sample_phase`（含 ensure_logprobs / note_kv / after_sample）、pre-sample wrap、async `AscendAsync*` |
| v2 | `worker/v2/model_runner.py` 构造 | 同上（native `dump_kv`；无 v2 msprobe debugger） |

Idle DP：`worker.execute_dummy_batch` 调 `sync_for_step(allow_arm=False)`，与 busy rank 对齐配置热更。

v1/v2 在 `compute_logits` 外包一层以插入 `check_before_sample`（`runner_hooks.wrap_compute_logits_for_pre_sample`）。

## 7. 非 worker 与多 engine

- **API / EngineCore**：每个 EngineCore 进程各自 `RuntimeGuardProcessor.bind`；配置 writer 为 per-EngineCore leader。
- **多 DP**：每个 DP replica 独立 JSON（或共享可读路径 + `sync_mode=file`）；不要用跨 idle DP 的 world collective 做热更。

## 8. 离线分析

抓现场后的脚本与 Agent skills 位于 `vllm_ascend/runtime_guard/analysis/`。  
用户向导读 [Runtime Guard Feature Guide](../../source/user_guide/feature_guide/runtime_guard.md#post-incident-analysis)。

## 9. 相关文档

- 运维与排障：[runtime_guard_ops.md](./runtime_guard_ops.md)
- 用户功能指南：[runtime_guard.md](../../source/user_guide/feature_guide/runtime_guard.md)
- 配置字段表：[runtime_config.md](../../source/user_guide/configuration/runtime_config.md)
- 启动项：[additional_config.md](../../source/user_guide/configuration/additional_config.md)
