# Runtime Guard

Runtime Guard is vLLM Ascend's online anomaly detection and incident response layer. It watches decode-time signals (token repetition, garbled output, non-finite logits, KV block integrity, and more), writes structured reports under `runtime/report/`, and optionally captures per-request KV cache blocks via native device-to-host dump (`dump_kv`).

## When to use

- Intermittent quality bugs: repetition, gibberish, sudden NaN/Inf
- Suspected KV corruption or block metadata inconsistency
- Need on-call artifacts (JSON report + optional `.pt` KV slices) without patching the model

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
  → detector hooks         # before/after sample, KV writes
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
| `block_kv` | KV write | Block wave / writer inconsistency |
| `slot_consistency` | KV write | Slot mapping vs block-table consistency |
| `position_alignment` | before sample | Position id mismatch |
| `spec_acceptance` | after spec | Spec-decode acceptance drift (via `run_sample_phase` → `check_after_spec`; v2 stashes accept stats in `postprocess_sampled`) |
| `token_logprob` | after sample | Logprob window anomalies (`ensure_logprobs_for_detection` runs at the start of `run_sample_phase`) |

All detectors default to **disabled**. Enable individually under `detector.<name>.enabled`.

> **Wiring:** v1/v2 `sample_tokens` call `RuntimeGuardProcessor.run_sample_phase` for post-pre-sample hooks (`ensure_logprobs` → `note_kv` / `mark_finished` / `check_after_spec` / waves / sync `check_after_sample`). Pre-sample `check_before_sample` stays on the compute_logits wrap (before grammar). Async `check_after_sample` runs in `AscendAsync*` `get_output()`.
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

## Post-incident analysis

After reports and optional KV dumps are captured:

- Package index: `vllm_ascend/runtime_guard/analysis/README.md`
- Scripts: `python -m vllm_ascend.runtime_guard.analysis.scripts.<name> ...`

Common flow:

1. `summarize_reports` — table of incidents  
2. `correlate_incident` — link `req_id` to report + kv dirs  
3. `verify_request_kv` — reconcile report vs `.pt` files  
4. `request_from_report` (or `prepare_ref_inputs`) + clean-service `dump_kv` — reference capture  
5. `compare_kv_similarity` (two reports) or `locate_first_divergence` / `compare_per_layer` — compare buggy vs ref KV

Agent skills (Cursor): `.agents/skills/runtime-guard-*` → full bodies under `vllm_ascend/runtime_guard/analysis/skill/`.

## Related docs

- [runtime_config.md](../configuration/runtime_config.md) — JSON field reference  
- [runtime_guard_design.md](../../../zh/design/runtime_guard_design.md) — full design  
- [runtime_guard_ops.md](../../../zh/design/runtime_guard_ops.md) — ops / troubleshooting
