# MS Service Metric

MS Service Metric collects vLLM Ascend runtime metrics through function hooks and exposes them in Prometheus format. Install `ms_service_metric` separately before using this integration; vLLM Ascend itself does not require it to run inference.

For installation, configuration syntax, and complete usage, see the [MS Service Metric documentation](https://gitcode.com/Ascend/msserviceprofiler/blob/master/ms_service_metric/README.md).

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

Configurations live under `vllm_ascend/observability/config/`. Handlers are re-exported from `vllm_ascend/observability/handlers/` so YAML can keep using `vllm_ascend.observability.handlers:function_name`.

| YAML file | Domain |
|-----------|--------|
| `base_metrics.yaml` | ModelRunner execute/prepare and engine memory |
| `eplb_metrics.yaml` | Dynamic EPLB hotness / phase duration |
| `flashcomm_metrics.yaml` | FlashComm decision / collective metrics |

## FlashComm Metrics

FlashComm needs business hooks because custom ops register at import time (YAML cannot wrap them later). Accumulators live in `flashcomm_stats.py`; Prometheus emission is via mount handlers.

| 职责 | 挂载点 | Handler |
|------|--------|---------|
| Gate 决策 Counter | `FlashCommMetricHooks.publish_forward_gate` | `flashcomm_gate_handler` |
| 集合通信失败 Counter | `FlashCommMetricHooks.note_collective_failure` | `flashcomm_failure_note_handler` |
| Forward 末 flush | `NPUModelRunner.execute_model` (v1/v2) | `flashcomm_forward_flush_handler` |

### Metrics（本分支新增）

| 点位 | 类型 | Label | 挂载点 | 用途 |
|------|------|-------|--------|------|
| `flashcomm:decision_total` | Counter | `{decision}` | `publish_forward_gate` | 是否启用 FlashComm 的门控决策 |
| `flashcomm:collective_failures_total` | Counter | `{op,reason}` | `note_collective_failure` | AG/RS 失败告警 |
| `flashcomm:active_tokens` / `padding_ratio` | Histogram | — | `execute_model` flush | 活跃 token 数与 padding 比例 |
| `flashcomm:input_bytes_total` | Counter | `{op}` | `execute_model` flush | 集合通信输入 tensor 字节数累计 |

> Trace 未实现；生产常开 Metrics，失败/慢路径再结合 HCCL 日志定位。

## Adding a Metric Point

vLLM Ascend-owned metric configurations are stored as YAML files in `vllm_ascend/observability/config/`. All YAML files in this directory are loaded automatically.

1. Identify the target function and add its fully qualified `module:Class.method` name as `symbol`.
2. Reuse a stable handler from `ms_service_metric.provider_handlers` when it provides the required data processing.
3. For Ascend-specific processing, add a handler under `vllm_ascend/observability/handlers/` and re-export it from `handlers/__init__.py` as `vllm_ascend.observability.handlers:function_name`.
4. Add the metric name and Prometheus type under `metrics` when using phase handlers, then update `tests/ut/observability/test_ms_metrics_provider.py`.

Example:

```yaml
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model
  handler: ms_service_metric.provider_handlers:model_runner_phase_handler
  metrics:
    - name: executor:model_runner_execute_model:duration
      type: histogram
```

Keep configurations and handlers aligned with the vLLM Ascend implementation. A missing symbol disables only the affected metric and must not affect inference.
