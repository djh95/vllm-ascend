# runtime_guard anomaly injection test matrix

> Goal: prove each detector actually fires on its target anomaly, not just
> passes UT on synthetic happy-path inputs. Live NPU injection is the only
> way to close the "detection didn't trigger" risk — UT proves the detector
> logic, injection proves the wire-up.

> **Note (PR-A):** online `block_kv` / `slot_consistency` / `position_alignment`
> are stripped pending the KV-meta follow-up. Scenarios #6–#10 below remain as
> backlog for that PR; ship injection only for shipped detectors.

## Why this exists

Live injection historically covered shipped detectors plus now-deferred KV/position
ones. On **PR-A**, only enable/inject:

`output_substring` / `token_repeat` / `spec_acceptance` / `logits_finite` /
`token_logprob`.

Deferred backlog (scenarios #6–#10 below): `block_kv` / `slot_consistency` /
`position_alignment`.

Still open even for shipped detectors:

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

## Path B injection matrix (5 scenarios + 2 cross-cutting; #6–#10 deferred)

Single env var `RG_INJECT` controls all hooks; absent → 0 overhead in prod.

```
RG_INJECT=scenario_name[:step_trigger][:param]
```

Injection entry points (as-built, one guarded call each in `processor.py`;
`step` counts pre-sample waves and defaults to 5):

| Hook | Scenarios |
|------|-----------|
| `check_before_sample` | #1 `nan_logits`, #2 `inf_logits` (corrupt `logits[0, col]` pre-detect, one-shot) |
| `check_after_spec` | #5 `spec_all_reject` (zero `accepted_token_nums` pre-detect, one-shot) |
| `check_after_sample` | #3 `forbidden_substring` (row-0 token cycles pattern ids), #4 `token_loop` (row-0 pinned to the pre-trigger token for `param` waves, default 40) |

### Detector coverage (shipped detectors on this branch)

> **Status (as-built):** `vllm_ascend/runtime_guard/inject.py` implements
> scenarios #1–#5 (`nan_logits` / `inf_logits` / `forbidden_substring` /
> `token_loop` / `spec_all_reject`) with synthetic UT
> (`tests/ut/runtime_guard/test_inject_scenarios.py`, no NPU). Live runs
> (§Live run procedure) are pending an NPU window; the live runner script is
> still to be added under `tests/perf/runtime_guard/`.

| # | Scenario | Injection point | What gets corrupted | Detector | Expected report field |
|---|----------|------------------|----------------------|----------|----------------------|
| 1 | `nan_logits` | runner post-logits | `logits[0,5]=NaN` | logits_finite | `kind=nan` + row |
| 2 | `inf_logits` | runner post-logits | `logits[0,3]=Inf` | logits_finite | `kind=inf` + row |
| 3 | `forbidden_substring` | post-sampler | replace sampled_tokens with `李白` after first decode step | output_substring | `pattern=李白` |
| 4 | `token_loop` | post-sampler | repeat last sampled token 32 times | token_repeat | `repeat_sum` + `window` |
| 5 | `spec_all_reject` | spec_acceptance pre-call | `accepted_token_nums=[0]*bs` | spec_acceptance | `rate≈0` + `window=10` |

Deferred backlog (KV-meta follow-up branch): #6 `kv_wave_regression` /
#7 `kv_same_wave_writer` (`block_kv`), #8 / #9 `slot_mismatch_*`
(`slot_consistency`), #10 `position_shift` (`position_alignment`).

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
| token_logprob | — | ❌ **design-skipped** — needs msprobe install; `DetectorManager.apply_runtime_config` force-sets `enabled=false` when msprobe is missing |

**4/5 shipped detectors covered by injection; 1 design-skipped.**

## Injection mechanism design

1. **Zero prod path**: `RG_INJECT` env unset → `inject.py` `inject_for_step()` returns immediately. No overhead.
2. **Reentrant**: each scenario is an independent function with isolated state.
3. **Observable**: injection triggers print `[INJECT] scenario=X step=N` so it can be cross-referenced with detector hit log lines.
4. **One-shot vs armed**: `nan_logits` / `inf_logits` / `spec_all_reject` fire
   once; `forbidden_substring` / `token_loop` stay armed from `step` (they need
   multiple waves to cross the detector window), with `token_loop` bounded by
   its `param` waves. Re-arm = restart the process.

## Implementation (as-built)

```
vllm_ascend/runtime_guard/
└── inject.py                          # env parser + all 5 scenario hooks (no NPU deps)

vllm_ascend/runtime_guard/processor.py # 3 guarded call sites:
                                       #   check_before_sample / check_after_spec / check_after_sample
                                       #   (`if inject.ENABLED:` → zero overhead when env unset)

tests/ut/runtime_guard/
├── test_inject_scenarios.py           # synthetic UT per scenario (no NPU)
└── ANOMALY_INJECTION.md               # this file

tests/perf/runtime_guard/
└── run_inject.sh                      # TODO (live window): start guard server,
                                       # loop RG_INJECT over scenarios, collect reports

Deferred with the KV-meta backlog: inject_scenarios/{kv_meta,position}.py
(#6–#10) — add submodules only when those scenarios ship.
```

## Live run procedure

For each scenario #1-#5, #11-#12:
1. Confirm guard server up on cards 6-7 (DeepSeek-V2-Lite, TP=2, port 8017)
2. Reset runtime_config.json: `stop_after_alert=true`, all 5 shipped detectors `enabled=true`
3. `curl -X POST .../manual_trigger` to clear any stale state
4. `RG_INJECT=scenario_name:step_trigger[:param] python tests/perf/runtime_guard/run_inject.py`
5. Wait for injection log `[INJECT] scenario=X step=N` to appear
6. Check `runtime_report_dir/<incident_type>/` for matching detector report
7. Cross-reference detector log line with `[INJECT]` line — both must appear in the same step window

## Pass criteria

- #1-#10: detector report file exists with expected `kind` / `pattern` / `wave_regression` / `last_writer_req_id` field
- #11: log shows `[runtime_guard clear] on_clear hook` for the alerted req_id; subsequent step's detector `_precheck` log shows the req_id in `stopped_req_ids()`
- #12: multiple reports exist for the same req_id across steps (proves stop_after_alert=false toggles off the skip)
