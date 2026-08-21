# MS Service Observability

vLLM Ascend exposes two complementary observability layers:

| Layer | Tool | Output | Config |
|-------|------|--------|--------|
| **Metrics** | [MS Service Metric](https://gitcode.com/Ascend/msserviceprofiler/blob/master/ms_service_metric/README.md) | Prometheus (`/metrics`) | `vllm_ascend/observability/config/*.yaml` via `MS_SERVICE_METRIC_CONFIG_PATH` |
| **Trace** | [MS Service Profiler](https://gitcode.com/Ascend/msserviceprofiler) | Chrome Tracing / CSV | `vllm_ascend/profiling_config.py` → `~/.config/vllm_ascend/service_profiling_symbols.*.yaml` via `PROFILING_SYMBOLS_PATH` |

MS Service Metric collects runtime metrics through function hooks. Install `ms_service_metric` separately before using the metrics integration; vLLM Ascend itself does not require it to run inference.

For metrics installation and syntax, see the [MS Service Metric documentation](https://gitcode.com/Ascend/msserviceprofiler/blob/master/ms_service_metric/README.md). For trace collection workflow, see [Service Profiling Guide](../developer_guide/performance_and_debug/service_profiling_guide.md).

## Basic Usage

Enable metric collection before starting the service:

```bash
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

ms-service-metric on
vllm serve <model-path>
```

`PROMETHEUS_MULTIPROC_DIR` stores the Prometheus multiprocess metric files. Use a separate directory for each service instance. Before restarting a service for validation, remove stale files only after confirming that no running service is using the directory.

After sending inference requests, query the vLLM metrics endpoint:

```bash
curl http://127.0.0.1:8000/metrics
```

The output can be scraped by Prometheus and visualized in Grafana. Use `ms-service-metric off` to disable collection.

## Custom Metric Configuration

Set `MS_SERVICE_METRIC_CONFIG_PATH` before starting vLLM. It can point to either one YAML file or a directory containing multiple YAML files:

```bash
# Load one YAML file.
export MS_SERVICE_METRIC_CONFIG_PATH=/data/custom_metrics/custom.yaml

# Or load all first-level .yaml and .yml files in one directory.
export MS_SERVICE_METRIC_CONFIG_PATH=/data/custom_metrics

ms-service-metric on
vllm serve <model-path>
```

Directory mode loads first-level `.yaml` and `.yml` files in filename order and does not scan subdirectories recursively. The configured directory is also the root for external Handler modules referenced by `module:function` in these YAML files. Users are responsible for maintaining their custom YAML files and Handler implementations.

Run `ms-service-metric restart` after changing YAML configuration. Restart the vLLM service after changing Handler Python code because imported Python modules are not reloaded dynamically.

## Metric Modules

Configurations are split by feature domain under `vllm_ascend/observability/config/`:

| YAML file | Domain |
|-----------|--------|
| `executor_metrics.yaml` | ModelRunner execute/prepare/sample, engine memory |
| `scheduler_metrics.yaml` | Scheduler batch, preempt, prefix connector cache |
| `kv_metrics.yaml` | PD KV transfer + KV Pool load/save failures |
| `spec_decode_metrics.yaml` | MTP / Eagle / DSpark rejection sampling |
| `graph_metrics.yaml` | ACL graph capture / replay / eager |
| `parallel_metrics.yaml` | SP padding, MoE comm selection (MC2 / multistream) |
| `lifecycle_metrics.yaml` | RL：sleep/wake、权重更新、routed-experts 状态 |
| `async_scheduling_metrics.yaml` | Async scheduling placeholders / D2H / queue |
| `eplb_metrics.yaml` | Dynamic EPLB hotness / rebalance / transfer |
| `flashcomm_metrics.yaml` | FlashComm decision / collective metrics |

Handlers live in `vllm_ascend/observability/handlers/` with one module per domain.

## How Metrics and Trace Fit Together

Metrics 与 Trace **不是强制一起开**，职责不同：

| | Metrics（常开） | Trace（按需） |
|--|----------------|---------------|
| **目的** | 告警、容量、回归对比、SLO | 单次慢请求 / 卡顿根因拆解 |
| **开销** | 低（Counter/Gauge/Histogram 采样） | 高（全链路 span，短时开启） |
| **输出** | Prometheus `/metrics` | Chrome Tracing / CSV |
| **典型用法** | Grafana 看板 + Alertmanager | 指标异常后，开 Profiler 定位是哪一段变慢 |

**推荐用法**

1. **生产默认只开 Metrics**（`ms-service-metric on`）。
2. **Metrics 告警或 perf 回退后**，再开 Trace（`SERVICE_PROF_CONFIG_PATH` + `enable: 1`）对照同一阶段（如 schedule / execute / sample）。
3. **不必**为每个 Metrics 点位都配一个 Trace；有耗时 Histogram 的阶段，优先复用已有 Trace span；仅 Metrics 覆盖、尚无 Trace 的模块见大表「对应 Trace」列的「未实现」。

## Failure Metrics: Merge or Not?

**同一故障域、处置路径相同 → 合成一个 Counter，用 label 区分原因**（已采用）：

| 合成后名称 | Label | 为何可合成 |
|------------|-------|------------|
| `scheduler:preempt_total` | `{reason}` | 都是抢占事件，排查入口相同 |
| `acl_graph:capture_failures_total` | `{reason}` | 都是 capture 失败 |
| `flashcomm:collective_failures_total` | `{op,reason}` | 都是集合通信失败，按 op/reason 下钻 |
| `lifecycle:sleep_wake_failures_total` | `{op}` | sleep / wake / update_weights 同类 RL 生命周期失败 |
| `eplb:rebalance:result` | `{result,policy_type}` | 同一次 rebalance 的结果枚举 |

**不同故障域、处置路径不同 → 不要合成一个「总失败」**：

| 勿合并的例子 | 原因 |
|--------------|------|
| `kv:*` vs `kvpool:*` | PD 远端传输失败 ≠ Pool put/load 失败，修法不同 |
| `kv:invalid_blocks_total` vs `kv:load_failure_events_total` | block 损坏计数 vs load 事件次数，单位与告警语义不同 |
| `async_scheduling:stale_output_discard_total` vs `placeholder_underflow_total` | preempt 丢弃 vs 占位符下溢，根因不同 |
| `flashcomm:collective_failures_total` vs `acl_graph:capture_failures_total` | 通信 vs 图模式，合在一起会丢失定位信息 |

**Counter + Gauge 也不要合成**：Counter 看速率/累计；Gauge 看当前积压（如 `kv:xfer_failed_requests`）。告警常两者一起用，但指标本身分开。

## Observability Matrix（大表）

列说明：

- **层**：`Metrics` / `Trace`
- **类型**：`Histogram` / `Counter` / `Gauge` / `Span` / `Sub-Span`
- **添加目的**：为什么要这个点
- **对应另一层**：Metrics↔Trace 对照；「常开 Metrics」表示 Trace 非必需；「告警后再开 Trace」表示联用场景
- **新加**：本轮 observability 拆分中新增的 Metrics；Trace 多为 `profiling_config` 原有

| 模块 | 层 | 类型 | 点位名称 | 添加目的 | 级别 | 对应另一层 | 新加 | 备注 |
|------|----|------|----------|----------|------|------------|------|------|
| Request / Engine | Trace | Span | `Chat.create_chat_completion` / `Completion.create_completion` | 看 API 入口整段耗时 | P1-T | 无 Metrics 对标（用 vLLM 请求级指标即可） | 否 | profiling_config |
| Request / Engine | Trace | Span | `AsyncLLM.generate` / `EngineCore.step*` / `add_request` | 拆请求从入队到 step 的卡点 | P0-T | 告警后再开 Trace | 否 | |
| Scheduler | Metrics | Histogram | `scheduler:schedule:duration` | 调度是否成为瓶颈 | P1-M | Trace: `batchFrameworkProcessing` | 否 | 可联用 |
| Scheduler | Metrics | Histogram | `scheduler:scheduled_tokens` | 每 step 批大小，对照吞吐 | P1-M | Trace schedule 属性 `num_scheduled_tokens` | **是** | |
| Scheduler | Metrics | Gauge | `scheduler:paused_requests` | Dyntra LB 暂停积压，防饿死 | P0-M | 无专用 Trace | **是** | Dyntra 开 |
| Scheduler | Metrics | Gauge | `scheduler:remote_kv_waiting_requests` | PD 等 remote KV 队列长度 | P0-M | Trace: `tryPromoteBlockedWaiting` | **是** | |
| Scheduler | Metrics | Counter | `scheduler:preempt_total{reason}` | 抢占频率与原因分布 | P0-M | 无专用 Trace | **是** | **已按 reason 合成** |
| Scheduler | Metrics | Counter | `prefix_cache:connector_queries_total` / `hits_total` | connector 前缀命中率 | P1-M | Trace: Mooncake match / KVCache get_computed | **是** | |
| Scheduler | Trace | Span | `batchFrameworkProcessing` / 各 Ascend `*.schedule` | schedule 墙钟 + 批属性 | P0-T | Metrics: `schedule:duration` | 否 | |
| Scheduler | Trace | Sub-Span | `schedule: allocate_slots` 等 4 段 | schedule 内子阶段拆解 | P1-T | 无独立 Metrics | 否 | inline |
| Executor | Metrics | Histogram | `executor:model_runner_execute_model:duration` | forward 总耗时回归 | P1-M | Trace: `modelRunnerExec` / `forward` | 否 | 可联用 |
| Executor | Metrics | Histogram | `executor:prepare_inputs:duration` | 准备阶段是否拖慢 | P1-M | Trace: `_prepare_inputs` / `prepare input` | 否 | |
| Executor | Metrics | Histogram | `executor:sample_tokens:duration_ms{runner}` | **整段** sample 墙钟（含 grammar/rejection/draft） | P1-M | Trace: `sample_tokens` / `sample_token` | **是** | 非仅 rejection |
| Executor | Metrics | Gauge | `engine:memory:*` | warmup 后内存基线 | P2-M | 无 | 否 | 启动一次 |
| Executor | Trace | Span | `modelExec` / `modelRunnerExec` / `_model_forward` | 执行链分层耗时 | P0-T | Metrics execute duration | 否 | |
| Executor | Trace | Span | `sample_tokens` / `_sample` / `_bookkeeping_sync` | sample 内部分解 | P0-T | Metrics sample duration | 否 | |
| Executor | Trace | Sub-Span | `prepare input` / `forward` / `post process` / `sample_token` / `draft_token` | execute 主路径子段 | P0/P1-T | 对应 Metrics 粗粒度耗时 | 否 | inline |
| KV PD | Metrics | Counter | `kv:invalid_blocks_total` | 无效 block 累计（数据损坏类） | P0-M | Trace: `updateKVXferFinished` 属性 | **是** | **勿与 load_failure 合成** |
| KV PD | Metrics | Counter | `kv:load_failure_events_total` | load 失败事件次数 | P0-M | 同上 | **是** | 事件 vs block 计数分离 |
| KV PD | Metrics | Counter | `kv:xfer_finished_total` | PD 接收完成吞吐 | P1-M | Trace Mooncake finished poll | **是** | |
| KV PD | Metrics | Gauge | `kv:xfer_failed_requests` | 当前失败接收积压 | P0-M | Trace finished 属性 | **是** | 与 Counter 联看 |
| KV PD | Metrics | Gauge | `kv:remote_kv_waiting_requests` | 当前等待 remote KV | P0-M | Trace promote | **是** | |
| KV PD | Trace | Span | Mooncake 全链 ~12 spans + `KVOutputAggregator` / `updateKVXferFinished` | PD KV 端到端路径拆解 | P0-T | Metrics kv:* 告警后再开 | 否 | |
| KV Pool | Metrics | Counter | `kvpool:load_requests_total{mode}` | Pool load 尝试量 | P1-M | **Trace 未实现** | **是** | |
| KV Pool | Metrics | Counter | `kvpool:load_error_blocks_total` | Pool load 错误 block | P0-M | **Trace 未实现** | **是** | **勿并入 PD kv:\*** |
| KV Pool | Metrics | Counter | `kvpool:put_failure_keys_total{backend}` | put 失败（常容量） | P0-M | **Trace 未实现** | **是** | 业务侧记 `_latest_put_failure_keys` |
| Spec Decode | Metrics | Counter | `spec_decode:draft_tokens_total` / `accepted_tokens_total` | 接受量分子分母 | P0-M | Trace rejection / draft spans | **是** | 可联用 |
| Spec Decode | Metrics | Histogram | `spec_decode:acceptance_ratio` | 单 step 接受率分布 | P0-M | 同上 | **是** | |
| Spec Decode | Metrics | Counter | `spec_decode:shape_mismatch_total` | logits/metadata 形状异常 | P0-M | 无 | **是** | 独立故障，勿并入 acceptance |
| Spec Decode | Trace | Span | `propose_draft_*` / `capture_rejection_output` / `draft_*` | draft 与 rejection 耗时拆解 | P0/P1-T | Metrics acceptance 告警后开 | 否 | |
| Async Scheduling | Metrics | Counter | `async_scheduling:stale_output_discard_total` | preempt 后丢弃 stale 输出 | P0-M | **Trace 未实现** | **是** | **勿与 underflow 合成** |
| Async Scheduling | Metrics | Counter | `async_scheduling:placeholder_underflow_total` | 占位符下溢（soft-fail） | P0-M | **Trace 未实现** | **是** | |
| Async Scheduling | Metrics | Gauge | `output_placeholders_total` / `pending_structured_output_tokens` / `spec_tokens_scheduled_total` | 占位与 structured/spec 状态快照 | P1-M | **Trace 未实现** | **是** | |
| Async Scheduling | Metrics | Histogram | `d2h_sync_ms` / `seq_lens_barrier_ms` | D2H / MTP barrier 是否拖慢 async | P0/P1-M | **Trace 未实现** | **是** | |
| Async Scheduling | Metrics | Gauge | `async_output_queue_depth` | worker 输出队列积压 | P1-M | **Trace 未实现** | **是** | 需 `--async-scheduling` |
| ACL Graph | Metrics | Counter | `acl_graph:capture/replay/eager_total{mode}` | 走图路径比例 | P1-M | **Trace 未实现** | **是** | |
| ACL Graph | Metrics | Counter | `acl_graph:capture_failures_total{reason}` | capture 失败率与原因 | P0-M | **Trace 未实现** | **是** | **已按 reason 合成** |
| ACL Graph | Metrics | Gauge | `acl_graph:cache_entries{mode}` | 图缓存规模 | P2-M | 无 | **是** | |
| Parallel | Metrics | Histogram | `parallel:sp_padding_ratio` | SP pad 浪费比 | P1-M | Trace: `_pad_for_sequence_parallelism` | **是** | |
| Parallel | Metrics | Counter | `parallel:moe_comm_selection_total{comm}` | MoE 通信路径选择分布 | P1-M | 无专用 Trace | **是** | 按 comm label 合成 |
| FlashComm | Metrics | Counter | `flashcomm:decision_total{decision}` | 是否启用 FC 的门控决策 | P0-M | **Trace 未实现** | **是** | |
| FlashComm | Metrics | Counter | `flashcomm:path_total{path}` | 每层 matmul 路径 | P1-M | **Trace 未实现** | **是** | |
| FlashComm | Metrics | Counter | `flashcomm:collective_failures_total{op,reason}` | AG/RS 失败 | P0-M | **Trace 未实现** | **是** | **已按 op+reason 合成** |
| FlashComm | Metrics | Histogram | `active_tokens` / `padding_ratio` / `*_ms` | 通信量与耗时 | P1-M | **Trace 未实现** | **是** | flush 于 execute 末 |
| FlashComm | Metrics | Counter | `flashcomm:communication_bytes_total{op}` | 通信量 | P1-M | **Trace 未实现** | **是** | |
| EPLB | Metrics | Gauge/Counter/Histogram | `eplb:expert_hotness:*` / `rebalance:*` / `transfer:*` / 阶段 duration / `async_worker:*` | 热度、重平衡、迁移、子进程健康 | P0/P1-M | Trace Sub-Span: EPLB p2p/load/D2D wait | **是**/扩展 | 可联用 |
| EPLB | Trace | Sub-Span | `EPLB generate p2p task` / `gather moe load` / `weight D2D wait` | rebalance/迁移子段耗时 | P1-T | Metrics eplb:* | 否 | inline |
| Lifecycle / RL | Metrics | Histogram | `lifecycle:sleep:duration_ms{level}` / `lifecycle:sleep:freed_gb{level}` | sleep 耗时与释放内存 | P1-M | Trace: `NPUWorker.sleep` | **是** | |
| Lifecycle / RL | Metrics | Histogram | `lifecycle:wake:duration_ms{sleep_opt,tags}` | wake（CaMem/KV/通信恢复）耗时 | P1-M | Trace: `NPUWorker.wake_up` | **是** | `sleep_opt`=是否 sleep_mode_extra_cleanup；`tags`=all\|partial |
| Lifecycle / RL | Metrics | Histogram | `lifecycle:update_weights:duration_ms` | RL HCCL/IPC 权重更新 chunk 耗时 | P0-M | Trace: `NPUWorker.update_weights` | **是** | |
| Lifecycle / RL | Metrics | Counter | `lifecycle:sleep_wake_failures_total{op}` | sleep/wake/update_weights 失败次数 | P0-M | 同上 | **是** | **已按 op 合成** |
| Lifecycle / RL | Metrics | Gauge | `vllm:engine_sleep_state{sleep_state}` | 引擎是否 awake / weights_offloaded / discard_all | P0-M | Trace sleep/wake | **否（上游）** | 勿在 Ascend 重复实现 |
| Lifecycle / RL | Metrics | Gauge | `rl:routed_experts_state{state}` | MoE router replay：enabled / ready / capturing | P1-M | 无专用 Trace | **是** | 见下方 RL 专节 |
| Lifecycle / RL | Trace | Span | `NPUWorker.sleep` / `wake_up` / `update_weights` | RL 休眠恢复与权重更新墙钟 + 属性 | P0/P1-T | Metrics lifecycle:* | **是** | profiling_config domain=`RL` |

**Trace 缺口（有 Metrics、建议后续补 Span）**：ACL Graph、FlashComm、KV Pool、Async Scheduling、SFA。

## RL / Lifecycle Metrics

强化学习训练场景（sleep/wake 释放与恢复、权重同步、MoE router replay）的点位集中在 `lifecycle_metrics.yaml` 与 `profiling_config.py`（domain `RL`）。

### 与上游的分工

| 能力 | 谁提供 | 说明 |
|------|--------|------|
| 引擎睡眠状态枚举 | 上游 `vllm:engine_sleep_state{sleep_state}` | `PrometheusStatLogger.record_sleep_state`；`sleep_state`=`awake` / `weights_offloaded` / `discard_all` |
| sleep/wake/update_weights 耗时与失败 | Ascend `lifecycle:*` | hook `NPUWorker.sleep` / `wake_up` / `update_weights` |
| Trace 墙钟 | Ascend `profiling_config` | `NPUWorker.sleep` / `wake_up` / `update_weights` |
| routed-experts 就绪态 | Ascend `rl:routed_experts_state` | init + capture 路径 |

**不要**再实现一份 Ascend 版 `engine_sleep_state`；看板直接用上游 Gauge，耗时/失败用 Ascend lifecycle 指标。

### Metrics 明细

| 点位 | 类型 | Labels | 何时更新 | 用途 |
|------|------|--------|----------|------|
| `lifecycle:sleep:duration_ms` | Histogram | `{level}` | `NPUWorker.sleep` 结束 | sleep 是否过慢 |
| `lifecycle:sleep:freed_gb` | Histogram | `{level}` | sleep 成功后 | 是否真正释放了显存 |
| `lifecycle:wake:duration_ms` | Histogram | `{sleep_opt,tags}` | `NPUWorker.wake_up` 结束 | 恢复 CaMem/模型/通信是否正常；对照是否开启 sleep mode optimization |
| `lifecycle:update_weights:duration_ms` | Histogram | — | 每次 `update_weights` chunk | HCCL/IPC 权重更新耗时，便于优化同步路径 |
| `lifecycle:sleep_wake_failures_total` | Counter | `{op=sleep\|wake\|update_weights}` | 对应 API 抛异常 | RL 生命周期失败告警 |
| `rl:routed_experts_state` | Gauge | `{state=enabled\|ready\|capturing}` | capturer init / capture | 是否开启 `enable_return_routed_experts`、capturer 是否就绪、是否正在 capture |
| `vllm:engine_sleep_state` | Gauge | `{sleep_state}` | 上游 sleep/wake API | 当前引擎处于 awake 还是某种 sleep 等级语义 |

`wake` labels：

- `sleep_opt=1|0`：是否走了 `rl_config.sleep_mode_extra_cleanup`（SleepWakeupManager）
- `tags=all|partial`：`wake_up(tags=None)` 为全量恢复，否则为部分 tag 恢复（避免高基数）

### Trace 明细

| Span | 属性 | 用途 |
|------|------|------|
| `NPUWorker.sleep` | `level` | 与 Metrics sleep duration 对照 |
| `NPUWorker.wake_up` | `tags`、`sleep_opt` | 判断 wake 慢是否因全量恢复 / sleep optimization |
| `NPUWorker.update_weights` | — | 单次权重 chunk 墙钟，分析更新是否异常 |

### 推荐联用

1. **常开 Metrics**：`engine_sleep_state`（上游）+ `lifecycle:*` + `rl:routed_experts_state`。
2. **wake 变慢**：先看 `lifecycle:wake:duration_ms` 按 `sleep_opt`/`tags` 分组，再开 Trace `NPUWorker.wake_up`。
3. **权重同步回归**：`lifecycle:update_weights:duration_ms` 告警后，开 Trace `NPUWorker.update_weights`。
4. **router replay**：`rl:routed_experts_state{state="ready"}==0` 而业务期望开启时，检查 `enable_return_routed_experts` 与 capturer init。

## Adding a Metric Point

1. Identify the target function and add its fully qualified `module:Class.method` (or `module:function`) name as `symbol`.
2. Reuse a stable handler from `ms_service_metric.provider_handlers` when it provides the required data processing.
3. For Ascend-specific processing, add a handler under `vllm_ascend/observability/handlers/`, re-export it from `handlers/__init__.py`, and reference it as `vllm_ascend.observability.handlers:function_name`.
4. Add the metric name and Prometheus type under `metrics` when using generic phase handlers, or record inside the Ascend handler.
5. Update `tests/ut/observability/test_ms_metrics_provider.py`.

Example:

```yaml
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model
  handler: ms_service_metric.provider_handlers:model_runner_phase_handler
  metrics:
    - name: executor:model_runner_execute_model:duration
      type: histogram   # Prometheus Histogram → 点位类型: Metrics / Histogram
```

Keep configurations and handlers aligned with the vLLM Ascend implementation. A missing symbol disables only the affected metric and must not affect inference.

## Adding a Trace Point

1. Add a `- symbol:` entry to `vllm_ascend/profiling_config.py` (`SERVICE_PROFILING_SYMBOLS_YAML`), or edit the generated file under `~/.config/vllm_ascend/`.
2. Set `domain` and `name` for Chrome Tracing grouping; add `attributes` when req_id / batch size context is needed.
3. For sub-phase breakdown inside hot paths, wrap code with `record_function_or_nullcontext("span name")` — collected automatically when the `record_function_or_nullcontext` symbol hook is active (vLLM ≥ 0.15).
4. Restart vLLM after changing symbols; enable collection via `ms_service_profiler_config.json` (`enable: 1`). See [Service Profiling Guide](../developer_guide/performance_and_debug/service_profiling_guide.md).

Example (symbol hook):

```yaml
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.sample_tokens
  domain: Sample
  name: NPUModelRunner.sample_tokens
```

Example (inline sub-span):

```python
with record_function_or_nullcontext("sample_token"):
    ...
```
