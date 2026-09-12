# runtime_guard 运维与排障

> 面向部署 / on-call。  
> 设计细节见 [runtime_guard_design.md](./runtime_guard_design.md)；  
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

### 2.2 手动 `dump.manual_dump`

| 字段 | 含义 |
|------|------|
| `manual_dump: false` | 关 |
| `manual_dump: true` | 持续 manual，直到热更改回 false |
| `manual_dump: N`（正整数） | 接下来 N 个有 scheduled tokens 的 wave 触发 manual |
| `manual_trigger`（别名） | 加载时映射为 `manual_dump`；新配置请写 `manual_dump` |

**要求 `runtime_config_reload_interval > 0`**。manual 事件 **跳过 auto quota/cooldown**。

manual 触发 incident_type 为 `manual_trigger`。**始终**注入 `dump_kv`，且 **强制** `scope=all_requests`（配置里的 `dump_kv.scope` 无效）。`on_trigger` 仍可配 `report` / `set_log_level` 等；省略时默认含 `report`，再自动补上 `dump_kv`。

每个有 scheduled tokens 的 wave 触发一次后 **扣减** `manual_dump` 计数（在 handle 之后；`manual_dump: true` 持续模式不扣）。

### 2.3 Report 字段与截断

落盘（仅 last-PP TP0）：`{report_dir}/<type>/report_<毫秒时间戳>[_<req_id>]_pid<pid>.json`

顶层常见字段：`incident_type`、`req_id`、`rank`、`dump_attempted`（on_trigger 含 `dump_kv`，非 D2H 已成功）、`dump_arm_wave`、`dump_dir`、`dump_count` / `dump_max_times`、`detail`。

同 `(incident_type, req_id)`：`report.max_per_req`（默认 **1**）；写满后 **停检该 req**；多份时 **wave 退避**（64×2ⁿ）。默认 `on_trigger` 含 `report`。与 `dump.auto_cooldown_seconds` 无关。

| 字段 | 作用 |
|------|------|
| `report.save_sensitive_info` | 是否写 prompt/output token ids |
| `report.max_prompt_token_ids` | 截断上限（0=不限） |
| `report.max_output_token_ids` | 截断上限 |
| `report.max_per_req` | 同 (type, req) 最多几份；写满停检（默认 1） |
| `report.include_block_ids` | detail 中带 GPU block_ids |
| `report.include_slot_mapping` | detail 中带本 wave slot_mapping（默认关） |
| `report.decode_token_ids` | 敏感信息模式下是否解码文本 |

查命中时若 report 过大，先关 `save_sensitive_info` 或降低 max_*。

### 2.4 KV dump 落盘与 rank

**范围：last PP × 全部 TP。其它 PP 不 dump。**

| Rank | dump 时机 |
|------|-----------|
| last PP TP0 | 检测命中时只排队；**下一拍**与其它 TP 一起 D2H |
| last PP 全部 TP | **下一拍** `sync_for_step`：收到 `{req_id, ...}` 后各自读本地 block 表再 D2H |
| 其它 PP | 不 dump |

```
{dump_root}/<incident_type>/<req_id>/wave_<N>/
  request_info.json          # last-PP TP0，arm 时写
  dp{D}_tp0_pp{last}_cp{C}/  # 与其它 TP 同一拍 D2H
  dp{D}_tp1_pp{last}_cp{C}/
  ...
    {req_id}_{layer}_req.pt
```

`dump_root` 默认 `<report_dir>/kv_cache`。每个 `.pt` 含：`req_id`、`block_ids`、`layer`、`rank_tag`、`tp_rank` / `pp_rank` / `cp_rank`、`num_kv_heads`、`tensor`。

拼 **同一 `req_id` + 同一 `wave_*` 下 last PP 的各 `tp*` 目录** 得到该 stage 的 head 切分。没有更早 `pp*` 目录是预期的（那些层没 dump）。

- `scope=request`（默认，**自动检测**）：各 rank 用本地 `block_ids_for_request`；请求已 finish 或表为空则跳过。
- `scope=all_requests`（自动检测可配；**manual_trigger 固定为此**）：arm 时用 `iter_local_request_rows` 枚举本地 batch，**每个 req 各自** `block_ids_for_request`。
- **同一步每个 `req_id` 最多 dump 一次**（pending 队列按 `req_id` 去重；多 detector / `all_requests` 重复命中不叠写）。
- 成功排队后 last-PP TP0 在 `{dump_root}/<type>/<req_id>/wave_<N>/request_info.json` 写请求元信息。
- 因 **finished/reaped** 跳过时，**last-PP TP0** 在  
  `{dump_root}/<type>/<req_id>/dump_skipped_finished.json` 留标记（`stage=arm|drain`）。
- 本地 `block_ids` 为空：**跳过 dump**（不会拖全池 KV）。
- `.pt` tensor 保持 `[n_blocks, block_size, …]`（仅该请求占用的 block）。
- **Quota：** 一次 `prepare` 只 `try_consume` 一次；同 `arm_id` 的多个 req job 在 drain 时若 **全部** 未产出 D2H，才 **refund 一次**（还次数并清 cooldown）。异步 `torch.save` 失败 **不** refund（额度按 D2H 机会计，不按落盘成功）。
- TP=1：无名单广播，下一拍只 dump 一份 `tp0` 目录。
- dump 通知：PP=1 且 config broadcast 时 hitchhike 同一趟 `all_reduce`；PP>1 file 模式在 last-PP TP 组用 `tp_group.cpu_group`（否则 `device_group`）做 `all_reduce(has_job)`（见 `processor._drain_kv_dump`），无 job 则不 `broadcast_object`；两者皆无时仍每步 `broadcast_object` 保齐步。

### 2.5 日志开关

| 配置 | 作用 |
|------|------|
| `log.print_output_on_finish` | 请求结束时打 output token ids / 解码文本（TP0） |
| `ascend_log.level` / `modules` | 模块日志级别；`[SamplingMeta]` 在 after-sample 打 DEBUG（开 `runtime_guard` DEBUG 即可见） |
| `set_log_level` action | incident 时临时提 log（detector 段可嵌套 `set_log_level`） |

## 3. 排障速查

| 现象 | 检查 |
|------|------|
| 无 report | detector 是否 `enabled`；是否 last PP + TP0；rank skip 日志 |
| 有 report 无 kv | `on_trigger` 是否含 `dump_kv`；quota/cooldown；`auto_max_times` 是否为 0 |
| 只有 `tp0` 没有其它 `tp*` | 是否 TP>1；其它 last-PP TP 是否跑到了**下一拍** `sync_for_step`；日志 `[runtime_guard dump_kv]` |
| 缺少更早 PP 的层 | 预期：当前不 dump 非 last PP |
| dump 文件空/缺层 | `block_ids` 是否为空；D2H 日志 `[runtime_guard dump_kv]` |
| 热更不生效 | `runtime_config_reload_interval` 是否 >0；JSON 路径各 rank 是否可读 |
| 重复刷屏 report | `report.max_per_req`（默认 1，写满停检）；确认 `on_trigger` 含 `report` |
| manual 不触发 | reload interval；是否 idle dummy wave；`manual_dump` 计数是否用尽 |
| 性能下降 | 先关全部 detector 仅留 reload，再逐项开启（见 feature guide） |

## 4. 日志关键字

| 关键字 | 含义 |
|--------|------|
| `[runtime_guard sync]` | 每 wave 配置同步 |
| `[runtime_guard manual_trigger]` | 手动 dump 控制面 |
| `[runtime_guard dump_kv]` | KV 排队 / D2H / 写盘（last-PP 全部 TP 下一拍一起 dump） |
| `[runtime_guard action]` | action prepare/commit 失败 |
| `record_runtime_guard_report` | metrics 计数（observability） |

## 5. 磁盘与 quota

- 每次 auto `dump_kv` 消耗 quota（`auto_max_times` 为进程内累计上限；用尽后需 **refund** 或 **重启** 才再放行。`auto_cooldown_seconds` 只限制两次**成功 consume** 的间隔，不解 cap）。
- `try_consume` 成功后若未能入队（例如 queue 不可用）或 drain 整 arm 未 D2H 会 **refund**（还次数并 **清除 cooldown**），避免空扣后卡冷却。
- 空闲空间门槛：leader 单卡 payload 估计 × `tp_size` + `free_headroom_bytes`（全 TP 写盘）。
- manual 路径不消耗 auto quota。
- 长期开 `dump_kv` 注意 `{report_dir}/kv_cache/` 磁盘；定期归档或调低 `auto_max_times`。

## 6. 相关文档

- [runtime_guard_design.md](./runtime_guard_design.md)
- [runtime_guard.md](../../source/user_guide/feature_guide/runtime_guard.md)
- [runtime_config.md](../../source/user_guide/configuration/runtime_config.md)
