# runtime_guard anomaly injection test matrix

> Goal: prove each detector actually fires on its target anomaly, not just
> passes UT on synthetic happy-path inputs. Live NPU injection is the only
> way to close the "detection didn't trigger" risk — UT proves the detector
> logic, injection proves the wire-up.

## Why this exists

Live B5–B14 covered "detector enabled + real workload sees real anomaly" for
output_substring / token_repeat / block_kv / spec_acceptance / slot_consistency
(prefill path) / position_alignment / logits_finite. But:

- **slot_consistency injection path NOT live-verified** — only the happy
  "first check ok checked=1243" path was hit. The mismatch-detection path
  (when KV actually disagrees with the slot's recorded token) is untested.
- **stop_after_alert cross-step skip behavior NOT live-verified** — every
  prior detector alert was followed by server shutdown, so "subsequent steps
  skip this req_id" was never observed.
- **Per-detector threshold boundaries NOT tested** — only `enabled=true/false`
  was toggled; window/threshold numerical behavior is UT-only.

This matrix covers those gaps via direct code injection (lower setup cost
than reverting real vllm-ascend bugfixes).

## Two paths — when to use which

| Path | Method | Fidelity | Setup cost | Use for |
|------|--------|----------|-----------|---------|
| **B (direct inject)** | env-gated debug hook writes wrong KV / NaN logits / loop tokens | Medium (synthetic but exercises the real detector code path) | 1-2 hours | All scenarios below |
| **A (revert real bugfix)** | `git revert <sha>` of a vllm-ascend KV-pollution PR + shrink `--block-size` | High (real bug replay) | 0.5-1 day (rebuild if csrc) | Cross-check B's slot_consistency result |

Path A candidates (DSV2-Lite compatible) from
`task_spec/kv_cross_request_contamination_survey_20260903.md`:

- vllm main #18957 ComputedBlocksTracker outdated (refcount; pure Python, no rebuild)
- vllm-ascend #5030 KV Pool TP rank mismatch (C++; needs .so rebuild)
- vllm main #51482 LIFO free_blocks reuse order (pure Python)

Path A is gated on path B passing first — A is the cross-check, not the
primary evidence.

## Path B injection matrix (9 scenarios + 2 cross-cutting)

Single env var `RG_INJECT` controls all hooks; absent → 0 overhead in prod.

```
RG_INJECT=scenario_name[:step_trigger][:param]
```

Injection entry point: `_refresh_config_body` end (per-step, all detectors
pre-flight). Hooks dispatch to `runner_hooks.py` / `kv_block_meta.py` /
`detector/manager.py` as needed.

### Detector coverage (7/8)

| # | Scenario | Injection point | What gets corrupted | Detector | Expected report field |
|---|----------|------------------|----------------------|----------|----------------------|
| 1 | `nan_logits` | runner post-logits | `logits[0,5]=NaN` | logits_finite | `kind=nan` + row |
| 2 | `inf_logits` | runner post-logits | `logits[0,3]=Inf` | logits_finite | `kind=inf` + row |
| 3 | `forbidden_substring` | post-sampler | replace sampled_tokens with `李白` after first decode step | output_substring | `pattern=李白` |
| 4 | `token_loop` | post-sampler | repeat last sampled token 32 times | token_repeat | `repeat_sum` + `window` |
| 5 | `spec_all_reject` | spec_acceptance pre-call | `accepted_token_nums=[0]*bs` | spec_acceptance | `rate≈0` + `window=10` |
| 6 | `kv_wave_regression` | kv_block_meta tracker | write block X wave=10, then wave=5 | block_kv | `wave_regression` |
| 7 | `kv_same_wave_writer` | kv_block_meta tracker | two reqs same wave same block | block_kv | `same_wave_writer` + `writer_req_ids` |
| 8 | `slot_mismatch_prefill` | block write path | slot stores token A, KV contains token B | slot_consistency (mode=first) | `last_writer_req_id` |
| 9 | `slot_mismatch_decode` | decode-pre KV overwrite | active block written with another req's KV | slot_consistency (mode=step) | `step` + `last_writer_req_id` |
| 10 | `position_shift` | sampler pre-entry position_ids | `position_ids[5:] += 1` | position_alignment | `position` + `expected` |

### Cross-cutting scenarios (verify stop_after_alert)

| # | Scenario | Operation | Verification point |
|---|----------|-----------|--------------------|
| 11 | `stop_after_alert_skip` | run scenario #1 once, continue stepping | that req_id no longer enters detector; log shows `skipped stopped_req_ids` for subsequent steps |
| 12 | `stop_after_alert_false_negative` | toggle config `stop_after_alert=false`, run #1 | each step fires a new report (proves mechanism is toggleable) |

### Coverage summary

| Detector | Scenario | Status |
|----------|----------|--------|
| logits_finite | #1, #2 | pending |
| output_substring | #3 | pending |
| token_repeat | #4 | pending |
| spec_acceptance | #5 | pending ⚠️ DSV2-Lite not running MTP currently; either enable `--num-speculative-tokens` + Eagle speculator or fall back to UT-only coverage |
| block_kv | #6, #7 | pending |
| slot_consistency | #8, #9 (first + step mode both) | pending |
| position_alignment | #10 | pending |
| token_logprob | — | ❌ **design-skipped** — depends on msprobe, force-disabled in no-msprobe branch |

**7/8 detectors covered by injection; 1 design-skipped.**

## Injection mechanism design

1. **Zero prod path**: `RG_INJECT` env unset → `inject.py` `inject_for_step()` returns immediately. No overhead.
2. **Reentrant**: each scenario is an independent function with isolated state.
3. **Observable**: injection triggers print `[INJECT] scenario=X step=N` so it can be cross-referenced with detector hit log lines.
4. **One-shot**: each scenario fires once unless explicitly re-armed (avoids accidental cascade in scenarios #11/#12).

## Implementation skeleton

```
vllm_ascend/runtime_guard/
├── inject.py                          # NEW: env parser + dispatch
└── inject_scenarios/
    ├── __init__.py
    ├── logits.py                      # #1, #2
    ├── sampler.py                     # #3, #4, #5
    ├── kv_meta.py                     # #6, #7, #8, #9
    └── position.py                    # #10

tests/ut/runtime_guard/
├── test_inject_scenarios.py          # NEW: synthetic UT for each scenario (no NPU needed)
└── ANOMALY_INJECTION.md              # this file

tests/perf/runtime_guard/
└── run_inject.sh                     # NEW: live runner — start guard server,
                                       # loop RG_INJECT over scenarios, collect reports
```

Inject call site (one line added to `processor._refresh_config_body` end, before
the `return`):

```python
from vllm_ascend.runtime_guard.inject import inject_for_step
inject_for_step(self, allow_arm=allow_arm, scheduler_output=scheduler_output)
```

`inject_for_step` checks `os.environ.get("RG_INJECT")` once at module load
(caches the parsed scenario); returns immediately if empty.

## Live run procedure

For each scenario #1-#12:
1. Confirm guard server up on cards 6-7 (DeepSeek-V2-Lite, TP=2, port 8017)
2. Reset runtime_config.json: `stop_after_alert=true`, all 6 detectors `enabled=true`,
   `slot_consistency.mode` per scenario (`first` for #8, `step` for #9)
3. `curl -X POST .../manual_trigger` to clear any stale state
4. `RG_INJECT=scenario_name:step_trigger[:param] python tests/perf/runtime_guard/run_inject.py`
5. Wait for injection log `[INJECT] scenario=X step=N` to appear
6. Check `runtime_report_dir/<incident_type>/` for matching detector report
7. Cross-reference detector log line with `[INJECT]` line — both must appear in the same step window

## Pass criteria

- #1-#10: detector report file exists with expected `kind` / `pattern` / `wave_regression` / `last_writer_req_id` field
- #11: log shows `[runtime_guard clear] on_clear hook` for the alerted req_id; subsequent step's detector `_precheck` log shows the req_id in `stopped_req_ids()`
- #12: multiple reports exist for the same req_id across steps (proves stop_after_alert=false toggles off the skip)
