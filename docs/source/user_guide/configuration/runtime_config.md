# runtime_config

JSON schema for Runtime Guard. Default path: `<cwd>/runtime/config/runtime_config.json`.

Annotated example: `vllm_ascend/runtime_config/templates/runtime_config.example.jsonc`.

Startup keys (`runtime_config_path`, `runtime_config_reload_interval`, overlay dict) are documented in [Additional Configuration](./additional_config.md#runtime_guard).

## Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sync_mode` | str | `"broadcast"` | `"broadcast"` (leader read + in-DP broadcast / local poll) or `"file"` (each rank polls path) |
| `reload_interval_seconds` | number | `0` | Display only; effective interval is `runtime_config_reload_interval` at process start |
| `actions` | object | see below | Default incident actions |
| `dump` | object | see below | Auto dump quota and manual dump controls |
| `ascend_log` | object | see below | Ascend logger level overrides |
| `log` | object | see below | Ops logging switches (not stored in report JSON) |
| `report` | object | see below | Report content and truncation |
| `detector` | object | see below | Detector sections + shared flags |

## actions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `defaults.on_trigger` | list[str] | `["report"]` | Actions when a detector section omits `on_trigger` |

Valid action names: `report`, `dump_kv`, `set_log_level`.

## dump

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_max_times` | int | `0` | Max auto `dump_kv` captures per process lifetime. `0` disables auto dump quota |
| `auto_cooldown_seconds` | float | `300` | Minimum seconds between auto dumps after a **successful** quota consume (refund clears this cooldown) |
| `manual_dump` | bool \| int | `false` | Manual dump control: `false`, `true` (continuous until cleared), or positive int (remaining waves) |
| `dump_dir` | str \| null | derived | KV dump root (default `<report_dir>/kv_cache`). Layout: `<dump_root>/<incident_type>/<req_id>/dp*_tp*_pp*_cp*/*.pt`. Coverage is last PP × all TP (not other PP). |
| `free_headroom_bytes` | int | `5368709120` (5 GiB) | Skip `dump_kv` when `statvfs` free space is below **estimated payload + this headroom**. Not a fixed free-space floor. |

Manual dump / manual trigger skip auto quota and cooldown. Requires hot-reload interval &gt; 0.

## report

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_sensitive_info` | bool | `false` | Persist prompt/output token ids in reports |
| `decode_token_ids` | bool | `true` | Decode ids to text when sensitive info saved |
| `max_prompt_token_ids` | int | `1000` | Truncate persisted prompt ids (`0` = unlimited) |
| `max_output_token_ids` | int | `1000` | Truncate persisted output ids |
| `include_block_ids` | bool | `true` | Include GPU block ids in report detail |
| `include_slot_mapping` | bool | `false` | Include slot_mapping slice in report |
| `max_per_req` | int | `1` | Max report files per `(incident_type, req_id)`; at cap, stop detecting that request. Wave backoff (64×2ⁿ) between writes when cap &gt; 1 |

## log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `print_output_on_finish` | bool | `false` | Log output token ids/text when any request finishes |

`[SamplingMeta]` is emitted at DEBUG on the after-sample path (TP0 + last PP). Enable with `ascend_log` / logger level for `vllm_ascend.runtime_guard` — there is no JSON toggle.

## ascend_log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | str | `"INFO"` | Base Ascend log level |
| `debug` | list[str] | `[]` | Module path prefixes forced to DEBUG under `vllm_ascend` |
| `modules` | object | `{}` | Per-logger overrides, e.g. `{"vllm.worker": "WARNING"}` |

## detector (shared)

Stop-detect is controlled by ``report.max_per_req`` (write-full), not a shared detector flag.
Default ``actions.defaults.on_trigger`` includes ``report``.

Each nested detector section supports:

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Master switch (default `false`) |
| `on_trigger` | list[str] | Override actions for this incident type |
| `dump_kv` | object | Per-type dump options: `scope` (`request` \| `all_requests`) |
| `set_log_level` | object | For `set_log_level` action: `level`, `modules` |

### spec_acceptance

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window` | int | `10` | Rolling window size |
| `low_threshold` | float | `0.3` | Low acceptance rate threshold |
| `len_low_threshold` | float | `1.4` | Length ratio at low rate |
| `high_threshold` | float | `0.96` | High acceptance rate threshold |
| `len_high_threshold` | float | `2.8` | Length ratio at high rate |

### output_substring

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `patterns` | list | `[]` | Token-id subsequences or string patterns to match |
| `add_special_tokens` | bool | `false` | Include special tokens when encoding string patterns |
| `match_prefix` | bool | `false` | Match only at output prefix vs anywhere |

### token_repeat

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window` | int | `32` | Sliding content window |
| `repeat_sum_threshold` | int | `64` | Alert when sum of repeat scores exceeds this |
| `min_tokens` | int | `32` | Minimum content tokens before alerting |
| `consecutive_hits` | int | `1` | Required consecutive over-threshold steps |
| `ignore_token_ids` | list[int] | `[]` | Token ids excluded from window scoring |

### logits_finite

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Alert on NaN/Inf logits before sampling |

Each step runs a device ``isfinite`` reduction and one ``.item()`` gate on the
pre-sample hook. On a hit only, bad rows / ``logits_indices`` / ``finite_kind``
are resolved while tensors are live and host ``Incident``s are enqueued.
``check_deferred`` on after-sample / ``get_output`` drains that queue (for dump
timing). Retired: ``check_every_tokens`` (multi-step window) — ignored if present.

### manual_trigger

Not a detector — control-plane for `dump.manual_dump`. Always runs `dump_kv` over the live batch (`scope=all_requests`; configured `dump_kv.scope` is ignored). `on_trigger` may still list `report` / other actions; if omitted, `actions.defaults` apply and `dump_kv` is injected.

Each armed wave with scheduled tokens decrements `manual_dump` after handle (`true` = continuous, no decrement).

```json
"manual_trigger": {
  "on_trigger": ["report", "dump_kv"]
}
```
## Example (detection + dump on repeat)

```json
{
  "sync_mode": "broadcast",
  "dump": {
    "auto_max_times": 5,
    "auto_cooldown_seconds": 300
  },
  "detector": {
    "token_repeat": {
      "enabled": true,
      "window": 32,
      "repeat_sum_threshold": 64,
      "on_trigger": ["report", "dump_kv"],
      "dump_kv": { "scope": "request" }
    }
  },
  "report": {
    "save_sensitive_info": true,
    "max_output_token_ids": 500
  }
}
```

## Related docs

- [Runtime Guard feature guide](../feature_guide/runtime_guard.md)
- [runtime_guard_design.md](../../../zh/design/runtime_guard_design.md)
- [runtime_guard_ops.md](../../../zh/design/runtime_guard_ops.md)
