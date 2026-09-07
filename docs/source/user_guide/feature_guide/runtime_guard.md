# Runtime Guard

Runtime Guard is vLLM Ascend's online anomaly detection and incident response layer. It watches decode-time signals (token repetition, garbled output, non-finite logits, speculative acceptance drift, and more), writes structured reports under `runtime/report/`, and optionally captures per-request KV cache blocks via native device-to-host dump (`dump_kv`).

## When to use

- Intermittent quality bugs: repetition, gibberish, sudden NaN/Inf
- Need on-call artifacts (JSON report + optional `.pt` KV slices) without patching the model
- Suspected KV issues: capture with `dump_kv` (online KV-meta detectors ship in a follow-up)

Default deployment: **detectors off, hot-reload off** — negligible overhead until you enable features in `runtime_config.json`.

## Quick start

**Detect + report only**

```bash
vllm serve Qwen/Qwen3-8B --additional-config '{
  "runtime_config_reload_interval": 5,
  "runtime_config": {
    "detector": {
      "token_repeat": { "enabled": true },
      "logits_finite": { "enabled": true }
    }
  }
}'
```

**Detect + report + KV dump on hit**

```bash
vllm serve Qwen/Qwen3-8B --additional-config '{
  "runtime_config_path": "/data/runtime/config/runtime_config.json",
  "runtime_config_reload_interval": 5
}'
```

Use the annotated template at `vllm_ascend/runtime_config/templates/runtime_config.example.jsonc` and set:

- `detector.<name>.enabled`: `true`
- `detector.<name>.on_trigger`: `["report", "dump_kv"]`
- `dump.auto_max_times`: e.g. `3` (required for auto dump quota)

## Startup options

Configure through `--additional-config` (or `LLM(..., additional_config=...)`):

| Key | Type | Description |
|-----|------|-------------|
| `runtime_config_path` | str | Path to `runtime_config.json`. Default: `<cwd>/runtime/config/runtime_config.json` |
| `runtime_config_reload_interval` | float | Hot-reload period in seconds. `0` = static after startup (default) |
| `runtime_config` | dict | Startup overlay merged into JSON defaults |
| `runtime_report_dir` | str | Override report root (default `<cwd>/runtime/report`) |

See [Additional Configuration](../configuration/additional_config.md#runtime_guard) and the full [runtime_config reference](../configuration/runtime_config.md).

## Architecture (summary)

```text
RuntimeGuardProcessor.bind(runner)
  → sync_for_step()        # config + wave + manual triggers
  → detector hooks         # before/after sample (and after spec)
  → ActionExecutor         # report | dump_kv | set_log_level (async queue)
```

Design details (Chinese): [runtime_guard_design.md](../../../zh/design/runtime_guard_design.md)  
Operations runbook (Chinese): [runtime_guard_ops.md](../../../zh/design/runtime_guard_ops.md)

## On-disk layout

```text
runtime/
  config/runtime_config.json
  report/
    <incident_type>/report_*.json
    kv_cache/<incident_type>/<req_id>/*.pt
```

## Detectors

| Type | Stage | Typical use |
|------|-------|-------------|
| `token_repeat` | after sample | Stutter / repetition |
| `output_substring` | after sample | Forbidden or garbage token patterns |
| `logits_finite` | before sample | NaN/Inf logits |
| `token_logprob` | after sample | Logprob window anomalies |
| `spec_acceptance` | after spec | Spec-decode acceptance drift (via `run_sample_phase` → `check_after_spec`; v2 stashes accept stats in `postprocess_sampled`) |

All detectors default to **disabled**. Enable individually under `detector.<name>.enabled`.

Online KV / position meta detectors are **not** in this release; use `dump_kv` for KV capture (offline compare tooling ships later).

> **Wiring:** v1/v2 `sample_tokens` call `RuntimeGuardProcessor.run_sample_phase` for post-pre-sample hooks (`ensure_logprobs` / `mark_finished` / `check_after_spec` / waves / sync `check_after_sample`). Pre-sample `check_before_sample` stays on the compute_logits wrap (before grammar; `logits_finite` only). Async `check_after_sample` runs in `AscendAsync*` `get_output()`.
## Actions

| Action | Effect |
|--------|--------|
| `report` | Write JSON incident report (+ metric counter) |
| `dump_kv` | D2H paged KV blocks for the request, save `.pt` files |
| `set_log_level` | Raise log verbosity synchronously on trigger |

Default `on_trigger` is `["report"]`. Per-detector overrides:

```json
"token_repeat": {
  "enabled": true,
  "on_trigger": ["report", "dump_kv"],
  "dump_kv": { "scope": "request", "dump_all_blocks": false }
}
```

## Performance

- **No additional-config / defaults**: bind-only path; intended to be noise-free.
- **Hot-reload only** (`reload_interval > 0`, all detectors off, dump off): small periodic JSON sync; UT bounds ~1–2% CPU on reload path.
- **Detectors on**: cost depends on enabled checks (light: `token_repeat`; heavier: `token_logprob` with logprobs).
- **dump_kv on hit**: one-time D2H spike proportional to blocks × layers.

Live NPU A/B checklist: `tests/perf/runtime_guard/README.md`.

## Related docs

- [runtime_config.md](../configuration/runtime_config.md) — JSON field reference  
- [runtime_guard_design.md](../../../zh/design/runtime_guard_design.md) — full design  
- [runtime_guard_ops.md](../../../zh/design/runtime_guard_ops.md) — ops / troubleshooting
