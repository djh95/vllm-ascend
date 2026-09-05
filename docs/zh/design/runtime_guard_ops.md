# runtime_guard 运维与排障

> 面向部署 / on-call。设计细节见 [runtime_guard_design.md](./runtime_guard_design.md)；  
> 配置字段见 [runtime_config.md](../../source/user_guide/configuration/runtime_config.md)。

## 1. 最小可用配置

仅开检测 + 报告（无 KV dump）：

```bash
vllm serve <model> --additional-config '{
  "runtime_config_path": "/data/runtime/config/runtime_config.json",
  "runtime_config_reload_interval": 5
}'
```

`/data/runtime/config/runtime_config.json` 示例：

```json
{
  "detector": {
    "token_repeat": { "enabled": true },
    "logits_finite": { "enabled": true }
  },
  "dump": { "auto_max_times": 0, "manual_dump": false }
}
```

也可在 **不改 JSON 文件** 时，用 `additional_config.runtime_config` 在启动时 overlay：

```bash
vllm serve <model> --additional-config '{
  "runtime_config_reload_interval": 5,
  "runtime_config": {
    "detector": {
      "token_repeat": { "enabled": true, "on_trigger": ["report", "dump_kv"] },
      "logits_finite": { "enabled": true }
    },
    "dump": { "auto_max_times": 3, "auto_cooldown_seconds": 300 }
  }
}'
```

合并顺序：`defaults ← runtime_config_path ← additional_config.runtime_config`。  
热更只重读 JSON 文件，不再套一层 overlay。

## 2. 常用操作

### 2.1 开检测 / dump_kv（可分离）

| 目标 | 配置 |
|------|------|
| 仅 report | `"on_trigger": ["report"]` 或省略（默认） |
| report + KV | `"on_trigger": ["report", "dump_kv"]`，且 `dump.auto_max_times > 0` |
| 仅 log 级别 | `"on_trigger": ["set_log_level"]` + nested `set_log_level` |

Detector 默认全关；逐项 `enabled: true` 开启。

### 2.2 手动 `dump.manual_dump` / `dump.manual_trigger`

| 字段 | 含义 |
|------|------|
| `manual_dump: false` | 关 |
| `manual_dump: true` | 持续 manual，直到热更改回 false |
| `manual_dump: N`（正整数） | 接下来 N 个有 scheduled tokens 的 wave 触发 manual |
| `manual_trigger: true` | 同上语义（JSON 布尔）；与 manual_dump 同类控制面事件 |

**要求 `runtime_config_reload_interval > 0`**。manual 事件 **跳过 auto quota/cooldown** 与 input filter。

manual 触发 incident_type 为 `manual_trigger`，默认对 batch 内各 live 请求写 report / dump（见 detector 段 `manual_trigger.on_trigger`）。

### 2.3 Report 字段与截断

| 字段 | 作用 |
|------|------|
| `report.save_sensitive_info` | 是否写 prompt/output token ids |
| `report.max_prompt_token_ids` | 截断上限（0=不限） |
| `report.max_output_token_ids` | 截断上限 |
| `report.include_block_ids` | detail 中带 GPU block_ids |
| `report.decode_token_ids` | 敏感信息模式下是否解码文本 |

查命中时若 report 过大，先关 `save_sensitive_info` 或降低 max_*。

### 2.4 KV dump 落盘

```
{report_dir}/kv_cache/<incident_type>/<req_id>/
  layer_<n>.pt
```

每个 `.pt` 含：`req_id`、`block_ids`、`dump_all_blocks`、`layer`、`source`、`tensor`。

- `scope=request`（默认）：只 dump incident 的 `block_ids`。
- `dump_all_blocks=false` 且 `block_ids` 为空：**跳过 dump**（不会拖全池 KV）。

### 2.5 日志开关

| 配置 | 作用 |
|------|------|
| `log.print_sampling_meta` | 异常请求打 `[SamplingMeta]`（TP0 + last PP） |
| `log.print_output_on_finish` | 请求结束时打 output token ids / 解码文本（TP0） |
| `ascend_log.level` / `modules` | 模块日志级别 |
| `set_log_level` action | incident 时临时提 log（如 `fatal_error` 段示例） |

### 2.6 打印一次输入 token ids（写 filter 用）

```json
"input_filter": { "print_input_token_ids_once": true }
```

需 `runtime_config_reload_interval > 0`。下一次有 prompt 的真实 wave 打印 ids 并给出 filter JSON 示例，然后自动清 false。

### 2.7 按输入过滤检测

```json
"input_filter": {
  "filters": [
    {
      "type": "input_token_id_prefix",
      "mode": "include",
      "prefixes": [[1234, 5678]]
    }
  ]
}
```

`manual_trigger` 不受 filter 影响。

## 3. 排障速查

| 现象 | 检查 |
|------|------|
| 无 report | detector 是否 `enabled`；是否 last PP + TP0；rank skip 日志 |
| 有 report 无 kv | `on_trigger` 是否含 `dump_kv`；quota/cooldown；`auto_max_times` 是否为 0 |
| dump 文件空/缺层 | `block_ids` 是否为空；D2H 日志 `[runtime_guard dump_kv]` |
| 热更不生效 | `runtime_config_reload_interval` 是否 >0；JSON 路径各 rank 是否可读 |
| 重复刷屏 report | `stop_after_alert` 是否为 false |
| manual 不触发 | reload interval；是否 idle dummy wave；`manual_dump` 计数是否用尽 |
| 性能下降 | 先关全部 detector 仅留 reload，再逐项开启（见 feature guide） |

## 4. 日志关键字

| 关键字 | 含义 |
|--------|------|
| `[runtime_guard sync]` | 每 wave 配置同步 |
| `[runtime_guard manual_trigger]` | 手动 dump 控制面 |
| `[runtime_guard dump_kv]` | KV D2H / 写盘 |
| `[runtime_guard action]` | action prepare/commit 失败 |
| `[runtime_guard print_input]` | 一次性 prompt 打印 |
| `record_runtime_guard_report` | metrics 计数（observability） |

## 5. 磁盘与 quota

- 每次 auto `dump_kv` 消耗 quota（`auto_max_times` 用尽后 blocked，直到 cooldown 或重启）。
- manual 路径不消耗 auto quota。
- 长期开 `dump_kv` 注意 `{report_dir}/kv_cache/` 磁盘；定期归档或调低 `auto_max_times`。

## 6. 相关文档

- [runtime_guard_design.md](./runtime_guard_design.md)
- [runtime_guard.md](../../source/user_guide/feature_guide/runtime_guard.md)
- [runtime_config.md](../../source/user_guide/configuration/runtime_config.md)
- 离线分析：`vllm_ascend/runtime_guard/analysis/README.md`
