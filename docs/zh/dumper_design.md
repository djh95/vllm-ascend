# Dumper 方案说明（vllm-ascend）

> 本文聚焦 **msprobe dump 生命周期与跨并行齐步**。  
> DFX 总览（Config / Detector / Dump / Report）见 [dfx_design.md](./dfx_design.md)。

## 1. 目标

`Dumper` 统一动态 dump 与异常检测触发，减少 `model_runner` 中的分散代码，保证 DP/PP/TP 下行为可预测。

## 2. 代码路径

- 核心实现：`vllm_ascend/dfx/dumper.py`（兼容：`vllm_ascend/dumper.py` re-export）
- Processor（runner 编排）：`vllm_ascend/dfx/processor.py`
- Runtime Config：`vllm_ascend/dfx/runtime_config.py`
- Detector：`vllm_ascend/dfx/detector/`
- Report：`vllm_ascend/dfx/report.py`
- v1：`vllm_ascend/worker/model_runner_v1.py`
- v2：`vllm_ascend/worker/v2/model_runner.py`

## 3. 结构与职责

`Dumper` 主要包含（**不管** config reload / report；由 ``DfxProcessor`` 编排）：

1. **应用已同步的 runtime config**
   - `apply_dfx_config()`：同步 `dump.max_times` / cooldown、刷新 log level

2. **debugger 生命周期**
   - `_init_debugger()`：按 `CUDAGraphMode` 选择 `PrecisionDebugger` 或 `AclGraphDumper`
   - `start_dump_data()` / `finalize_dump_data()`

3. **接 `AnomalyAlert`**
   - `handle_anomaly_alert()`：arm / activate dump；**不**写 report

4. **dump 开关与跨 TP 齐步**
   - `enable_msprobe_dump_if_needed()` / `sync_dump_pending_or()` / `disable_msprobe_dump_if_needed()`

5. **请求过滤** — `is_related_local_request()`

Runner 侧：

- `self.dfx = DfxProcessor(self)`；`self.dumper = self.dfx.dumper`
- 直接调用 `dfx.refresh_config()` / `dfx.clear_finished()` / `dfx.check_*`

## 4. 调用链（v1 / v2）

### 4.1 v1

1. 初始化：`Dumper(..., dfx_config=ascend_config.dfx_config)`
2. `execute_model()` 入口：`dfx.refresh_config()` → `sync_dump_pending_or()` → `start_dump_data` → forward →（非 last PP 早 `finalize`；last PP 在 `sample_tokens` 末 `finalize`）
3. 采样后：`dfx.clear_finished` → `dfx.check_spec_acceptance`；sync 当场 `dfx.check_token_logprobs`，async 在 `AscendAsyncGPUModelRunnerOutput.get_output()` 中检测

### 4.2 v2

1. 初始化：同上；`load_model` 在图模式下可提前 `start_dump_data`（构图用）
2. `execute_model()`：`dfx.refresh_config()` → `sync_dump_pending_or(..., allow_arm=not dummy_run)` → `start` → `super().execute_model` → `finalize(dump=not dummy_run)`
3. `postprocess_sampled()`：`dfx.check_spec_acceptance`
4. `sample_tokens()`：sync 当场 `dfx.check_token_logprobs`；async 包装为 `AscendAsyncOutput`，在 `get_output()` 后检测

## 5. Async 跨 TP dump 齐步（last PP）

```text
check 命中（async 仅 last-PP TP0）:
  pending_dump = True          # 不写 dump_enable

execute_model 入口:
  dfx.refresh_config()         # 全 rank：rank0 JSON → world broadcast（默认）
  sync_dump_pending_or():      # 仅 last PP
    all_reduce(SUM, pending) on tp_group.cpu_group
    any_pending = (sum > 0)
      if any_pending and allow_arm:
          各 TP: activate(dump_enable=true + reload)
          clear pending

start → forward → finalize → disable（需 _dump_forward_seen）
```

说明：

- **不区分 req_id**；OR 的是「是否 pending」布尔。
- **async 仅 TP0 check**：multiproc 只在 `output_rank`（last-PP TP0）调用 `get_output()`。
- **early PP 不参与 dump OR**，但仍必须跑 `dfx.refresh_config`（world collective）。
- **Sync + TP>1 / async**：check 仅 TP0 → `pending_dump`；下步 `execute_model` 入口 last-PP TP `all_reduce(OR)` 后全体 activate（避免 sample 中途只开部分 TP 的 debugger 导致集体通信卡住）。
- **Sync + TP=1**：可当场 activate。
- pending / dump_active 期间跳过后续 anomaly check，避免重复 arm。

## 6. DP / PP / TP

### 6.1 PP

- check / enable / dump OR / activate：仅 **last PP**
- early PP：不 dump，但必须参与 **world** 级 `sync_dfx_config`（避免 collective 卡死）

### 6.2 TP

- 日志：`tp_rank == 0`
- **check（async，或 sync 且 TP>1）**：仅 TP0；**dump**：OR 后 last-PP 全体 TP activate
- **check + dump（sync 且 TP=1）**：单卡当场 activate

### 6.3 DP

- 各 DP 副本独立；`tp_group` 不含跨 DP 进程
- Config broadcast 使用 **world**（单 engine 内跨 TP/PP/DP）；外部多 engine DP 见 [dfx_design.md](./dfx_design.md) §2.2

## 7. 路径与落盘

1. msprobe 配置：`runner.ascend_config.dump_config_path` / `dump_config`
2. DFX 运行时配置：`dfx_config_path`（默认 `<cwd>/dfx/config/dfx_config.json`）
3. 异常短报告：`<dfx_root>/report/anomaly_YYYYMMDD.log`
4. `set_msprobe_dump_state`：msprobe JSON 旁 `.lock` 持锁写 `dump_enable`
5. `save_sample_param`：在 ``DfxProcessor``（`mark_full_log` 的 alert；TP0 && last PP）

## 8. 已知限制

1. `forward_seen` 只表示「activate 后调用过 start」，不保证 msprobe 一定写出文件。
2. v1 EC producer 短路径可能在 activate 后用 encoder-only `start→finalize` 消费窗口；普通文本 serving 无此路径。
3. async 下 last PP 每步 CPU all_reduce（全员参与）；不能「仅 pending 的 rank 进 collective」。
4. Sync 不做 OR；依赖同拍各 TP check 语义。
5. Config broadcast 与 dump OR 使用不同 process group（world vs tp）；二者都要求同组全员同拍进入。
