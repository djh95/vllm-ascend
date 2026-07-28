# Token/Logprob 异常检测与动态 Dump 方案

> DFX 总览见 [dfx_design.md](./dfx_design.md)。实现类：`vllm_ascend/dfx/detector/token_logprob.py`。

## 1. 目标

在现有投机接受率异常检测之外，增加基于 **输出 token + top-k logprobs** 的在线异常检测（生僻字 / 乱码 / 重复 / NaN），与接受率检测共用动态 msprobe dump、`debug_log_full` 与 DFX Report 通路。

设计原则：

- **尽快检出**：小窗口、低命中阈值；重复略严一点防误报。
- **滑窗外置**：Detector 维护队列；msprobe ILLDetector 配成「一次调用 = 一窗」，避免双重滑窗。
- **可配置开关**：spec acceptance / token_logprob 分别使能（JSON 热更新）。
- **复用 dump**：命中后走 `enable_msprobe_dump_if_needed`，并写 `anomaly_type=token_logprob` 报告。

## 2. 开关与配置

**推荐**：编辑 DFX JSON（默认 `<cwd>/dfx/config/dfx_config.json`，或 `dfx_config_path`）：

| 字段（`detector` / `dump` 段） | 默认 | 说明 |
|------|------|------|
| `detector.enable_spec_acceptance_check` | `true` | 投机接受率检测 |
| `detector.enable_token_logprob_check` | `false` | token/logprob 检测（开启后 worker 会自动补齐 top-k logprobs，请求侧可不设 `logprobs`） |
| `detector.token_logprob_window` | `64` | 每请求缓冲长度 = 送检窗长 |
| `detector.token_logprob_stride` | `32` | 满窗后每新增 N token 再检 |
| `detector.token_logprob_topk` | `20` | 每位置最多保留 top20 |
| `detector.ill_nan_window_thresh` | `1` | NaN/Inf 命中窗数 |
| `detector.ill_rare_window_thresh` | `1` | 生僻字 |
| `detector.ill_garbled_window_thresh` | `1` | 乱码 |
| `detector.ill_repet_window_thresh` | `2` | 重复（半重叠两窗确认） |
| `dump.max_times` | `0` | `0` 时不触发 dump（检测也会直接 return） |
| `dump.enabled` | `true` | 总开关 |

示例（rank0 可热改，`sync_mode=broadcast` 时会广播到各 rank）：

```json
{
  "sync_mode": "broadcast",
  "dump": { "enabled": true, "max_times": 3, "cooldown_seconds": 300 },
  "detector": {
    "enable_spec_acceptance_check": true,
    "enable_token_logprob_check": true,
    "token_logprob_window": 64,
    "token_logprob_stride": 32,
    "token_logprob_topk": 20,
    "ill_repet_window_thresh": 2
  }
}
```

**兼容**：仍可通过 `additional_config.dynamic_dump_config` 启动 overlay（扁平字段，如 `dynamic_dump_max_times`），见 [dfx_design.md](./dfx_design.md) §2.4。

请求侧**不必**再手动设 `logprobs`：检测开启且 `dump.max_times > 0` 时，采样前 `DfxProcessor.ensure_logprobs_for_detection()` 会把 batch 内请求的 top-k 至少抬到 `token_logprob_topk`（默认 20）。若客户端已设更大的 `logprobs`，则保留更大值。

`--async-scheduling`：token/logprob 在采样返回时尚在 device 上，检测改在 async `get_output()`（D2H 完成并 parse 之后）执行（v1：`AscendAsyncGPUModelRunnerOutput`；v2：`AscendAsyncOutput`）。multiproc 仅 `output_rank`（last-PP TP0）会 `get_output()`，故 **async 仅 TP0 check**；命中后只置 `pending_dump`，下一拍 `execute_model` 入口 last-PP TP `all_reduce(OR)` 后再全体写 `dump_enable`（详见 [dumper_design.md](./dumper_design.md) §5）。

## 3. 架构

```text
model_runner_v1 (sample/bookkeeping 后)
  ├─ clear_finished_requests(...)   # 唯一清理由处（spec history + token 缓冲）
  ├─ check_all_spec_acceptance(...) # SpecAcceptanceDetector
  └─ check_all_token_logprobs(...)  # TokenLogprobDetector
        │
        ▼
TokenLogprobDetector 每请求 deque(maxlen=window)
  满窗 / 之后每 stride 新 token
        │
        ▼
msprobe ILLDetector.detector(topk_dicts, tokens, model_config)
  （内部 window=stride=队列长 → 单窗）
        │
        ▼
按 ill_type 累加命中次数，达 thresh
  → enable_msprobe_dump + debug_log_full + DFX report
```

### 3.1 为何不两边各滑一套

- msprobe 默认 `window_size=128, stride=64`，且 `single_window_thresh=14` 适合离线长序列。
- 在线：Detector 队列长度 = 窗长；构造 ILLDetector 后覆盖为 `window_size=stride=token_logprob_window`，并把 garbled/repeat 的内部多窗阈值置 0，使 **单次调用能返回 is_ill**。
- **多窗投票改由 `ill_*_window_thresh` 完成**，便于尽快检出且可热更新。

### 3.2 logprobs 布局

vLLM `LogprobsLists` 每行：`[sampled_logprob, top1, …, topk]`。

`_row_to_topk_dict`：按 logprob 降序取前 `token_logprob_topk`，转成 `Dict[token_id, logprob]` 再交给 detector。

MTP / 投机：一步多个 accepted token → 多行 logprobs，按序 append；使用 `cu_num_generated_tokens` 切片。

### 3.3 model_config / tk2cat

- 传入 `{"model_name": Path(model).name}` 供名称模糊匹配。
- `get_tk2cat` 依赖「末 token 为 eos」校验；生成中途常走 **无词表 top1 阈值** 路径。类别增强需预加载 tk2cat（后续优化）。

## 4. 代码落点

| 模块 | 说明 |
|------|------|
| `vllm_ascend/dfx/runtime_config.py` | JSON 热更新 / broadcast；`detector` + `dump` 段 |
| `vllm_ascend/dfx/detector/token_logprob.py` | 缓冲、命中计数、`check_all` |
| `vllm_ascend/dfx/detector/spec_acceptance.py` | 投机接受率 |
| `vllm_ascend/dfx/dumper.py` | dump 生命周期、`check_all_*` 转发、Report |
| `vllm_ascend/ascend_config.py` | `dfx_config_path`、`DynamicDumpConfig` 兼容 overlay |
| `vllm_ascend/worker/model_runner_v1.py` / `v2` | sync / async 调用点 |
| `docs/.../additional_config.md` | 用户配置表 |

## 5. 生命周期与资源

- 每请求缓冲：最多 `window × topk` 个 (id, logprob)。
- 请求结束：`clear_finished_requests` 销毁缓冲与命中计数。
- 检测时日志：`active_reqs`、`ill_type`、hits；报告见 `dfx/report/anomaly_*.log`。

## 6. 与 dump 共用策略

- 冷却 / 最大次数 / 每请求只 dump 一次：沿用 `enable_msprobe_dump_if_needed`。
- **Async**：仅 TP0 check，arm `pending_dump`；last-PP TP OR 后全体 `_activate`。**Sync**：各 last-PP TP check 内直接 activate。
- 已 `pending` / `dump_active` 时跳过后续 check，避免重复 arm。
- TP0 打详细日志；dump 仅 last PP；状态写 msprobe 配置文件。
- `debug_log_full` 在 dump 使能成功后置位，snapshot 到 `ModelRunnerOutput`。

## 7. 限制与后续

1. 未开检测或 `dump.max_times=0` 时不做 token_logprob 检测；开启后会自动强制 top-k logprobs。
2. 中途无 eos → tk2cat 可能不可用。
3. v1 / v2 均已接入：`check_token_logprobs` + async `get_output` 延迟检测；采样前 `ensure_logprobs_for_detection`。
4. 若需更激进：减小 `window`/`stride`，或将 `ill_repet_window_thresh` 设为 `1`。
5. 跨 TP 齐步与 dump 生命周期详见 [dumper_design.md](./dumper_design.md)；配置广播见 [dfx_design.md](./dfx_design.md)。
