# DFX 方案说明（vllm-ascend）

> 设计 for eXcellence：运行时维测控制面。  
> 代码根目录：`vllm_ascend/dfx/`

## 1. 组件与流程

| 组件 | 模块 | 职责 |
|------|------|------|
| 1. Runtime Config | `runtime_config.py`（`DfxRuntimeConfig`） | 一份 JSON；可选热更新（启动项控制周期） |
| 2. Detector | `detector/` | 异常检测，只产出 `AnomalyAlert` |
| 3. Dump / 观测开关 | `dumper.py`（`Dumper`） | msprobe dump 生命周期；`ascend_log` / metrics / trace 开关（metrics/trace 接口已留，接线后期） |
| 4. Report | `report.py`（`DfxReportWriter`） | 异常短日志落盘到 `dfx/report/` |
| 5. Processor | `processor.py`（`DfxProcessor`） | runner 侧编排（构造 / refresh / check / report） |

对外入口：`from vllm_ascend.dfx import Dumper`（以及 `DfxProcessor` / `DfxRuntimeConfig` 等）。

```text
additional_config
  ├─ dfx_config_path / dfx_config_reload_interval
  └─ AscendConfig.dfx_config (DfxRuntimeConfig)
         │
Worker: runner.dfx = DfxProcessor(runner)
  execute_model 入口（拆两段，勿合并）
  ├─ dfx.refresh_config()          # 全 rank；热更关则立刻 return
  └─ dfx.sync_dump_pending_or()    # 仅 last-PP TP
         │
采样 / get_output
  dfx.clear_finished / check_spec / check_token_logprobs
    → detector.check_all → AnomalyAlert
    → dumper.handle_anomaly_alert  # 只管 dump
    → report_writer.write          # Report
```

> 注意：检测由 **processor 调 detector**，再用 alert 调 dumper；detector **不**直接 `enable_dump`。  
> Config / Report 也不应塞进 dumper 的 dump OR 路径，否则有人「优化跳过 early PP 的 maybe_reload」会让 world broadcast 卡死。

## 2. Runtime Config

### 2.1 路径

| 优先级 | 来源 | 路径 |
|--------|------|------|
| 1 | `additional_config.dfx_config_path` 或 `dfx-config` | 显式路径 |
| 2 | 默认 | `<cwd>/dfx/config/dfx_config.json` |

启动热更新开关（权威，JSON 不能重新打开）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `dfx_config_reload_interval` | `5` | 启动项（也可写入 JSON 的 `reload_interval_seconds` 供查看）。`0`=关闭周期刷新；默认 `5`=每隔 5 秒热更。**以启动 `additional_config` 为准**，仅改 JSON 不能重新打开 |

报告目录默认：与 config 同级的 `dfx/report/`（可用 `dfx_report_dir` 覆盖）。  
模板：`vllm_ascend/dfx/templates/dfx_config.json`。首次启动若文件不存在会按默认内容创建。

### 2.2 同步模式 `sync_mode`

| 值 | 行为 | 适用 |
|----|------|------|
| `broadcast`（**默认**） | 仅 **world global rank0** 读/写 JSON；各 rank 集体 `all_reduce(due)` + `broadcast_object` | 多机无共享盘：改 rank0 上那一份即可 |
| `file` | 每进程按启动参数间隔轮询本地/共享路径 mtime | 共享文件系统 |

广播注意：

- 必须在**所有 rank 同一拍**调用 `dfx.refresh_config()` / `sync_dfx_config()`（已挂在 runner `execute_model` 入口，与 dump OR 分开）。
- **禁止**把 config sync 折叠进「仅 last-PP」的 dump 路径。
- 范围是**单个 engine 的 `get_world_group()`**。
- `save()` 在 broadcast 模式下非 leader 会忽略。

### 2.2.1 非 worker（API / EngineCore）

Detector / dump / report **只跑在 worker**。

`ascend_log` 级别：`AscendConfig` 构造时即 `apply_ascend_log_level`（含 API/EngineCore）。当 `dfx_config_reload_interval > 0` 且进程 **未**设置 `RANK` 时，另启动守护线程 `dfx-non-worker-reload`，按间隔 **本地 file 轮询** JSON 并在变更后再次 `apply_ascend_log_level`（**不**进 worker world broadcast，**不**落盘）。Worker 在 `Dumper` 初始化 / `refresh_config` → `apply_dfx_config` 时同样应用。

Worker 仍走 `execute_model` → `refresh_config` → broadcast；**不要**在 worker 上再起并行热更线程。

### 2.2.2 外部多 engine DP

产品约定二选一（写清即可，勿混用）：

1. **每套 engine 的 rank0 一份 JSON**（`broadcast`）：各引擎互不影响，运维改各自 rank0 路径上的文件；
2. **`file` + 共享盘**：所有引擎进程轮询同一共享路径。

### 2.3 JSON 结构

```json
{
  "sync_mode": "broadcast",
  "reload_interval_seconds": 5,
  "dump": {
    "enabled": true,
    "max_times": 0,
    "cooldown_seconds": 300,
    "dump_once": false
  },
  "ascend_log": { "level": "INFO", "debug": [] },
  "metrics": { "enabled": true, "level": "INFO" },
  "trace": { "enabled": false, "level": "INFO", "otlp_endpoint": null },
  "detector": {
    "enable_spec_acceptance_check": true,
    "enable_token_logprob_check": false,
    "spec_acceptance_window": 10,
    "spec_acceptance_low_threshold": 0.3,
    "spec_acceptance_len_low_threshold": 1.4,
    "spec_acceptance_high_threshold": 0.96,
    "spec_acceptance_len_high_threshold": 2.8,
    "token_logprob_window": 64,
    "token_logprob_stride": 32,
    "token_logprob_topk": 20,
    "ill_nan_window_thresh": 1,
    "ill_rare_window_thresh": 1,
    "ill_garbled_window_thresh": 1,
    "ill_repet_window_thresh": 2
  }
}
```

| 段 | 含义 |
|----|------|
| `dump` | `enabled` / `max_times` / `cooldown_seconds`；`dump_once`：手动改 JSON 为 `true` 后，worker **下一次 `execute_model`（需有请求）** 热更读到后触发一次 dump（不计次数、忽略冷却），再写回 `false`。**必须** `dfx_config_reload_interval > 0`。空闲无请求时不会消费。还需要 `dump_config_path`（msprobe）且 `dump.enabled=true` |
| `ascend_log` | `level`：`vllm_ascend` 包根 logger 级别。`debug`：模块白名单（相对路径，如 `["dfx"]` → `vllm_ascend.dfx`）强制 DEBUG。走 Ascend 专用 handler（不受 `VLLM_LOGGING_LEVEL` 的 `vllm` handler 过滤）。无 `enabled` |
| `metrics` / `trace` | 观测开关与级别（仍待引擎接线） |
| `detector` | 各检测器开关与阈值 |

### 2.4 与 `dynamic_dump_config` 的关系

- **推荐**：改 DFX JSON（或 rank0 JSON + broadcast）。
- **兼容**：`additional_config.dynamic_dump_config` 仍可用；仅**显式**启动键参与合并（`user_overrides`）。
- **启动引导**（**仅 worker leader 落盘一次**；API/EngineCore/`init_ascend_config` 只做内存合并）：
  - **未**配 `dfx_config_path` → 内存用 `defaults ← startup`（忽略旧默认路径内容）；**不删除**磁盘文件（避免非 leader 误删 leader 刚写出的 JSON）；worker leader `ensure_persisted` 覆盖写出 `<cwd>/dfx/config/dfx_config.json`；
  - **已**配路径 → 先读该 JSON，再 `defaults ← JSON ← startup`；leader 若文件已存在则不盲目重写（以磁盘为准），文件缺失时写出；
  - 启动日志打印最终 `path=`（AscendConfig + worker Processor）。
- 旧字段映射：`dynamic_dump_max_times` → `dump.max_times`，`dynamic_dump_cooldown_seconds` → `dump.cooldown_seconds`，其余检测字段 → `detector.*`。
- **热更合并**：仅 `defaults ← JSON`（启动 overlay 只在 bootstrap 用一次并写回文件；之后以 JSON 为准）。
- **`dump_once`**：依赖热更；`dfx_config_reload_interval` 必须 `> 0`。

### 2.5 命名

| 推荐 | 说明 |
|------|------|
| `DfxRuntimeConfig` / `runtime_config.py` | 运行时热更新控制面 |
| `DfxProcessor` / `processor.py` | runner 侧编排（构造 dumper/detectors、check/clear/report） |

## 3. Detector

| 类 | 文件 | `anomaly_type` |
|----|------|----------------|
| `AnomalyDetector`（基类） | `detector/base.py` | — |
| `SpecAcceptanceDetector` | `detector/spec_acceptance.py` | `spec_acceptance` |
| `TokenLogprobDetector` | `detector/token_logprob.py` | `token_logprob` |
| `ManualDumpDetector` | `detector/manual_dump.py` | `manual_dump_once` |

基类约定：

- `refresh_from_config()`：从 live `DfxRuntimeConfig.detector` 拉开关/阈值
- `check_all` / `check_one`：返回 `list[AnomalyAlert]` / `AnomalyAlert | None`（**不**调用 Dumper）
- `on_alert_armed(alert)`：dump 成功后的可选日志钩子
- **Spec 检测条件**：runner 上存在 `speculative_config`（MTP/Eagle 等），**不**依赖仅 hybrid/Mamba 才置位的 `need_accepted_tokens`

调用链（``DfxProcessor`` 编排，runner 只挂接）：

```text
runner.dfx = DfxProcessor(runner)
  ├─ refresh_config() / sync_dump_pending_or()
  ├─ clear_finished / check_spec_acceptance / check_token_logprobs
  └─ _handle_alert → dumper.handle_anomaly_alert + save_sample_param + report_writer.write
```

> 注意：检测由 **processor 调 detector**，再用 alert 调 dumper；detector **不**直接 `enable_dump`。  
> `save_sample_param` 在 ``DfxProcessor``（alert 后的 log sink，不属于 dump 生命周期）。

`AnomalyAlert`（`detector/alert.py`）对齐 msprobe `ILLDetector.detector(...)` 的 `is_ill` / `ill_type`，并带上 dump/report 元数据。

细节：

- 投机接受率：[dumper_design.md](./dumper_design.md)
- Token/logprob：[token_logprob_anomaly_design.md](./token_logprob_anomaly_design.md)
- Async 时序：[async_issues_analysis.md](./async_issues_analysis.md)

## 4. Dump（Dumper）

职责：debugger 生命周期、pending OR 齐步、start/finalize 配对、接 `AnomalyAlert`。  
实现位置：`vllm_ascend/dfx/dumper.py`。

每步入口（runner → ``DfxProcessor``）：

1. `dfx.refresh_config()` → `sync_dfx_config()`（仅当 `dfx_config_reload_interval > 0`；**全 rank**）
2. 若 config 变更：`dumper.apply_dfx_config()` + detector refresh；`ManualDumpDetector` → alert（`consume_quota=False`）
3. `dfx.sync_dump_pending_or()`（仅 last-PP TP；**不含** config / report）

Dumper **不**调用 config reload，也 **不**写 report（report 在 processor）。

门控（异常检测）：`dump.enabled == false` 或两路 detector 均关 → 不跑检测。  
门控（自动 dump）：另需 `max_times > 0` 且未超配额 / 冷却；`max_times == 0` 时仍可检测与打 short 日志，只是不 arm dump。  
`dump_once` 由 `ManualDumpDetector` 消费；仅要求 `dump.enabled` + debugger，不受 `max_times` / cooldown 限制。  
**前提**：`additional_config.dfx_config_reload_interval > 0`（热更为关时改 JSON 的 `dump_once` 不会生效）。

## 5. Report

- 类：`DfxReportWriter`
- 目录：默认 `<dfx_root>/report/`
- 文件：`anomaly_YYYYMMDD_HHMMSS.log`（按秒一份，JSON Lines）

示例行：

```json
{
  "ts": "2026-07-28T11:00:00",
  "unix_ts": 1753666800.0,
  "anomaly_type": "spec_acceptance",
  "req_id": "req-1",
  "rank": "tp0-pp1",
  "detail": { "acceptance_rate": 0.1 }
}
```

检测触发 dump 成功时由 **DfxProcessor** 追加一行（`report_writer`）。

## 6. 非 worker 与多 engine

### 6.1 非 worker（API / EngineCore）

Detector / dump / report **只跑在 worker**。  
`ascend_log`（`level` + `debug`）：`AscendConfig` 启动时应用一次；热更开启且无 `RANK` 时由后台 file 轮询线程在 JSON 变更后再次应用（见 §2.2.1）；worker 在 config sync 后经 `DfxRuntimeConfig.apply_ascend_log_level` 委托 `vllm_ascend.logger.apply_ascend_log_level`。`metrics` / `trace` 配置段与访问器已预留，引擎接线后期再做。

### 6.2 外部多 engine DP

产品约定二选一：

1. **每套 engine 的 rank0 一份 JSON**（`broadcast`）；
2. **`file` + 共享盘**。

## 7. 启动示例

```bash
# 多机：JSON 放在 global rank0 可读路径；开启 5s 热更新
vllm serve <model> --additional-config '{
  "dfx_config_path": "/data/dfx/config/dfx_config.json",
  "dfx_config_reload_interval": 5,
  "dump_config_path": "/data/msprobe_dump.json"
}'
```

或默认路径（进程 cwd 下自动创建；默认热更间隔 5s）：

```text
./dfx/config/dfx_config.json
./dfx/report/anomaly_YYYYMMDD_HHMMSS.log
```

开启默认（或显式配置）`dfx_config_reload_interval` 后，在线改 `dump.max_times` / detector 阈值约 N 秒内各 worker 生效（broadcast）。

## 8. 相关文档

| 文档 | 内容 |
|------|------|
| [dumper_design.md](./dumper_design.md) | Dump 生命周期、PP/TP 齐步、调用链 |
| [token_logprob_anomaly_design.md](./token_logprob_anomaly_design.md) | ILLDetector 窗口与配置 |
| [async_issues_analysis.md](./async_issues_analysis.md) | 异步调度下的时序与 OR |
| 用户配置 | `docs/source/user_guide/configuration/additional_config.md` |
