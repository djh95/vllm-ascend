# Dumper 方案说明（vllm-ascend）

## 1. 目标

`Dumper` 的目标是统一动态 dump 与 投机接受率观测逻辑，减少 `model_runner` 中的分散代码，确保在 DP/PP/TP 并行场景下日志和 dump 行为可预测。

## 2. 代码路径

- 核心实现
  - `vllm-ascend/vllm_ascend/dumper.py`
- v1 接入点
  - `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
- v2 接入点
  - `vllm-ascend/vllm_ascend/worker/v2/model_runner.py`

## 3. 结构与职责

`Dumper` 主要包含六类能力：

1. debugger 生命周期
- `_init_debugger()`：按 `CUDAGraphMode` 选择 `PrecisionDebugger` 或 `AclGraphDumper`
- `start_dump_data()`：本轮开始前启动 debugger；若本拍为 dump-forward 则置 `_dump_forward_seen`
- `finalize_dump_data()`：本轮结束后 `stop/step` 并按需回滚 `dump_enable`；`dump=False`（dummy/capture）不消费 dump-forward 窗口

2. 投机接受率观测与触发（`enable_spec_acceptance_check`）
- `check_all_spec_acceptance()` / `check_spec_acceptance_anomaly()`：窗口接受率、阈值、触发 full log 与 dump
- `log_spec_token_details()`：打印 sampled/accepted/prompt/output token 明细

3. Token/logprob 异常检测（`enable_token_logprob_check`）
- 每请求缓冲 token + topk logprobs；满窗 / stride 后调用 msprobe `ILLDetector`
- 按 `ill_*_window_thresh` 累计命中后触发 dump（详见 [token_logprob_anomaly_design.md](./token_logprob_anomaly_design.md)）

4. dump 开关与跨 TP 齐步
- `enable_msprobe_dump_if_needed()`：门控（冷却、最大次数、每请求一次）；**async** 只置 `pending_dump`，**sync** 直接 `_activate_msprobe_dump`
- `begin_step_dump_decision()`：仅在 **last PP** 的 TP 组上对 `pending_dump` 做 CPU `all_reduce`（AND）；齐则全体 last-PP TP `activate`；early PP 直接跳过（不 dump、不 broadcast）
- `disable_msprobe_dump_if_needed()`：`activate` 后置 `_dump_needs_forward`；仅当后续又经历一次 dump 向的 `start`（`_dump_forward_seen`）后的 finalize 才回滚
- `set_msprobe_dump_state()`：持锁写 JSON `dump_enable` 并立刻 `_maybe_reload_config`

5. 本地请求过滤
- `is_related_local_request()`：只允许当前 rank 上存在且有效的请求触发 dump
- `clear_finished_requests()`：请求结束时清理 spec / token_logprob 状态

6. 观测辅助
- `save_sample_param()`：dump 激活时记录 sampling 元数据（TP0 + last PP）

## 4. 调用链（v1 / v2）

### 4.1 v1

1. 初始化：创建 `self.dumper`
2. `execute_model()` 入口：`begin_step_dump_decision(async_mode=use_async_scheduling)` → 主路径 `start_dump_data` → forward →（非 last PP 早 `finalize`；last PP 在 `sample_tokens` 末 `finalize`）
3. 采样后：`clear_finished_requests` → `check_all_spec_acceptance`；sync 当场 `check_all_token_logprobs`，async 在 `AscendAsyncGPUModelRunnerOutput.get_output()` 中检测

### 4.2 v2

1. 初始化：创建 `self.dumper`；`load_model` 在图模式下可提前 `start_dump_data`（构图用）
2. `execute_model()`：`begin_step_dump_decision(..., allow_arm=not dummy_run)` → `start` → `super().execute_model` → `finalize(dump=not dummy_run)`
3. `postprocess_sampled()`：`check_all_spec_acceptance`
4. `sample_tokens()`：sync 当场 `check_all_token_logprobs`；async 包装为 `AscendAsyncOutput`，在 `get_output()` 后检测

## 5. Async 跨 TP dump 齐步（last PP）

```text
get_output / check 命中（各 last-PP TP，可乱序）:
  pending_dump = True          # 不写 dump_enable

execute_model 入口（仅 last PP）:
  all_reduce(SUM, pending) on tp_group.cpu_group
  all_ready = (sum == tp_world_size)
  if all_ready and allow_arm:
      各 TP: activate(dump_enable=true + reload)
      clear pending
  else if any_pending 连续失败 >= 5:
      clear pending（超时）

start → forward → finalize → disable（需 _dump_forward_seen）
```

说明：

- **不区分 req_id**；AND 的是「是否 pending」布尔。
- **early PP 不参与 dump**（精度对比通常只看最后一段），无需 PP broadcast。
- **Sync** 仍在 check 时直接 activate；各 TP 同拍 check 后下一拍一起 dump，不做 AND。
- feature 关闭（两 check 关或 `max_times==0`）时整条齐步路径跳过。
- pending / dump_active 期间跳过后续 anomaly check，避免重复 arm。

## 6. DP / PP / TP

### 6.1 PP

- check / enable / `begin_step` AND / activate：仅 **last PP**
- early PP：不 dump

### 6.2 TP

- 日志：`tp_rank == 0`
- dump：last PP 上各 TP 均需 pending 齐后一起 activate（async）

### 6.3 DP

- 各 DP 副本独立；`tp_group` 不含跨 DP 进程（例：DP2/PP2/TP2 → `tp_group.world_size == 2`）

## 7. 路径与落盘

1. msprobe 配置：`runner.ascend_config.dump_config_path`
2. `set_msprobe_dump_state`：`<path>.lock` 持锁写 `dump_enable` 并 reload
3. `save_sample_param`：TP0 && last PP

## 8. 已知限制

1. `forward_seen` 只表示「activate 后调用过 start」，不保证 msprobe 一定写出文件。
2. v1 EC producer 短路径（`has_ec_transfer && is_producer`）可能在 activate 后用 encoder-only `start→finalize` 消费窗口；普通文本 serving 无此路径。
3. async 下 last PP 每步 CPU all_reduce（全员参与）；不能「仅 pending 的 rank 进 collective」。
4. Sync 不做 AND；依赖同拍 check 语义。
