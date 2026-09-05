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
| `input_filter` | object | see below | Detect-time input filters |

## actions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `defaults.on_trigger` | list[str] | `["report"]` | Actions when a detector section omits `on_trigger` |

Valid action names: `report`, `dump_kv`, `set_log_level`.

## dump

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_max_times` | int | `0` | Max auto `dump_kv` captures per process lifetime. `0` disables auto dump quota |
| `auto_cooldown_seconds` | float | `300` | Minimum seconds between auto dumps after quota consume |
| `manual_dump` | bool \| int | `false` | Manual dump control: `false`, `true` (continuous until cleared), or positive int (remaining waves) |
| `dump_all_blocks` | bool | `false` | Global default for `dump_kv` when per-detector override absent |

Manual dump / manual trigger skip auto quota, cooldown, and input filters. Requires hot-reload interval &gt; 0.

## report

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_sensitive_info` | bool | `false` | Persist prompt/output token ids in reports |
| `decode_token_ids` | bool | `true` | Decode ids to text when sensitive info saved |
| `max_prompt_token_ids` | int | `1000` | Truncate persisted prompt ids (`0` = unlimited) |
| `max_output_token_ids` | int | `1000` | Truncate persisted output ids |
| `include_block_ids` | bool | `true` | Include GPU block ids in report detail |
| `include_slot_mapping` | bool | `false` | Include slot_mapping slice in report |
| `block_last_write_wave` | bool | `false` | Track last write wave per physical block |
| `block_last_writer` | bool | `false` | Track last writer req_id per block |

## log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `print_sampling_meta` | bool | `false` | Log sampling metadata for anomalous requests (TP0 + last PP) |
| `print_output_on_finish` | bool | `false` | Log output token ids/text when any request finishes |

## ascend_log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | str | `"INFO"` | Base Ascend log level |
| `debug` | list[str] | `[]` | Module path prefixes forced to DEBUG under `vllm_ascend` |
| `modules` | object | `{}` | Per-logger overrides, e.g. `{"vllm.worker": "WARNING"}` |

## detector (shared)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `stop_after_alert` | bool | `true` | After first alert per request, stop further detection for that request |

Each nested detector section supports:

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Master switch (default `false`) |
| `on_trigger` | list[str] | Override actions for this incident type |
| `exec_scope` | str | Placement hint: `auto` / `leader` / `any` / `all` / `external` |
| `dump_kv` | object | Per-type dump options: `scope` (`request` \| `all_requests`), `dump_all_blocks` |
| `set_log_level` | object | For `set_log_level` action: `level`, `modules` |

### spec_acceptance

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window` | int | `10` | Rolling window size |
| `low_threshold` | float | `0.3` | Low acceptance rate threshold |
| `len_low_threshold` | float | `1.4` | Length ratio at low rate |
| `high_threshold` | float | `0.96` | High acceptance rate threshold |
| `len_high_threshold` | float | `2.8` | Length ratio at high rate |

### token_logprob

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window` | int | `64` | Analysis window |
| `stride` | int | `32` | Stride between windows |
| `topk` | int | `20` | Top-k logprobs tracked |
| `ill_nan_window_thresh` | int | `1` | NaN window alert threshold |
| `ill_rare_window_thresh` | int | `1` | Rare-token window threshold |
| `ill_garbled_window_thresh` | int | `1` | Garbled window threshold |
| `ill_repet_window_thresh` | int | `2` | Repetition window threshold |

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

### block_kv

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `check_wave_regression` | bool | `true` | Detect block write wave going backwards |
| `check_same_wave_writer` | bool | `true` | Detect conflicting writers same wave |

### slot_consistency

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Check slot mapping against block-table metadata on KV writes |

### position_alignment

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Check position_ids alignment on scheduled tokens |

### logits_finite

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Alert on NaN/Inf logits before sampling |

### manual_trigger

Not a detector — control-plane section for manual dump events. Configure `on_trigger` and `dump_kv` defaults for `incident_type=manual_trigger` reports.

Example:

```json
"manual_trigger": {
  "on_trigger": ["report", "dump_kv"],
  "dump_kv": { "scope": "all_requests" }
}
```

## detector_placement

Top-level object (sibling of `detector`). Spreads detectors across TP ranks via `detector/placement.py`. Hot-reload may replan and reset per-request detector history.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | str | `"auto"` | `"auto"` (planner) or `"manual"` |
| `manual` | object | `{}` | Map detector name → TP rank when `mode=manual` |
| `pin` | bool | (see defaults) | Prefer sticky assignment across reloads when supported |

## input_filter

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `filters` | list | `[]` | Filter chain applied before detect (not for `manual_trigger`) |
| `print_input_token_ids_once` | bool | `false` | Log prompt token ids once on next real batch, then clear |

Supported filter type: `input_token_id_prefix` with `mode` (`include` \| `exclude`) and `prefixes` (list of token-id prefix lists).

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
      "dump_kv": { "scope": "request", "dump_all_blocks": false }
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
