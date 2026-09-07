# runtime_guard anomaly injection test matrix

> Goal: prove each detector actually fires on its target anomaly, not just
> passes UT on synthetic happy-path inputs. Live NPU injection is the only
> way to close the "detection didn't trigger" risk — UT proves the detector
> logic, injection proves the wire-up.

## Why this exists

Live B5–B14 covered "detector enabled + real workload sees real anomaly" for
output_substring / token_repeat / slot_consistency (prefill path) /
logits_finite. But:

- **slot_consistency injection path NOT live-verified** — only the happy
  "first check ok" path was hit. The mismatch-detection path
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

### Detector / invariant coverage

| # | Scenario | Injection point | What gets corrupted | Check | Expected report field |
|---|----------|------------------|----------------------|-------|----------------------|
| 1 | `nan_logits` | runner post-logits | `logits[0,5]=NaN` | logits_finite | `kind=nan` + row |
| 2 | `inf_logits` | runner post-logits | `logits[0,3]=Inf` | logits_finite | `kind=inf` + row |
| 3 | `forbidden_substring` | post-sampler | replace sampled_tokens with `李白` after first decode step | output_substring | `pattern=李白` |
| 4 | `token_loop` | post-sampler | repeat last sampled token 32 times | token_repeat | `repeat_sum` + `window` |
| 5 | `spec_all_reject` | spec_acceptance pre-call | `accepted_token_nums=[0]*bs` | spec_acceptance | `rate≈0` + `window=10` |
| 6 | `kv_slot_order_gap` | note_kv / reshape path | skip an offset in a contiguous write batch | `kv_slot_order` | `incident_type=kv_slot_order`, `violation=gap` |
| 7 | `kv_slot_order_jump` | note_kv | write starting past `next_offset` | `kv_slot_order` | `incident_type=kv_slot_order`, `violation=wrong_start` |
| 8 | `slot_mismatch_first` | stamp wrong token into slot meta before first check | slot token ≠ seq | slot_consistency (phase=first) | `incident_type=kv_slot_token`, `mismatches` |
| 9 | `slot_mismatch_finish` | corrupt slot meta before reap | slot token ≠ seq at finish | slot_consistency (phase=finish) | `incident_type=kv_slot_token` |
| 10 | `position_shift` | sampler pre-entry position_ids | `position_ids[5:] += 1` | (removed) | was position_alignment |
| 11 | `slot_token_swap` | wrong block mapping | map req to other req slots | slot_consistency | `kv_slot_token` mismatches |

### Cross-cutting scenarios (verify stop_after_alert)

| # | Scenario | Operation | Verification point |
|---|----------|-----------|--------------------|
| 11 | `stop_after_alert_skip` | run scenario #1 once, continue stepping | that req_id no longer enters detector; log shows `skipped stopped_req_ids` for subsequent steps |
| 12 | `stop_after_alert_false_negative` | toggle config `stop_after_alert=false`, run #1 | each step fires a new report (proves mechanism is toggleable) |

### Coverage summary

| Check | Scenario | Status |
|----------|----------|--------|
| logits_finite | #1, #2 | pending |
| output_substring | #3 | pending |
| token_repeat | #4 | pending |
| spec_acceptance | #5 | pending ⚠️ DSV2-Lite not running MTP currently; either enable `--num-speculative-tokens` + Eagle speculator or fall back to UT-only coverage |
| slot_consistency (`kv_slot_token`) | #8, #9, #11 | pending |
| kv_slot_order | #6, #7 | pending |
| token_logprob | — | ❌ **design-skipped** — depends on msprobe, force-disabled in no-msprobe branch |

**Note:** legacy `block_kv` / wave·writer injection scenarios are removed; KV
cross-req detection is owned by the slot meta state machine
(`UNKNOWN` / `SLOT_PARTIAL` / `BLOCK_SEALED`).

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
    └── kv_meta.py                     # #6–#9, #11

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
2. Reset runtime_config.json: `stop_after_alert=true`, relevant checks `enabled=true`
   (`invariant.kv_slot_order` for #6–#7; `invariant.slot_consistency` for #8–#9/#11;
   no `mode` key — token checks are `phase=first` once + `phase=finish` at reap)
3. `curl -X POST .../manual_trigger` to clear any stale state
4. `RG_INJECT=scenario_name:step_trigger[:param] python tests/perf/runtime_guard/run_inject.py`
5. Wait for injection log `[INJECT] scenario=X step=N` to appear
6. Check `runtime_report_dir/<incident_type>/` for matching report
   (`kv_slot_token` / `kv_slot_order` for slot scenarios)
7. Cross-reference detector log line with `[INJECT]` line — both must appear in the same step window

## Pass criteria

- #1-#10: report file exists with expected `kind` / `pattern` / `violation` / `mismatches` field
- #11: log shows `[runtime_guard clear] on_clear hook` for the alerted req_id; subsequent step's detector `_precheck` log shows the req_id in `stopped_req_ids()`
- #12: multiple reports exist for the same req_id across steps (proves stop_after_alert=false toggles off the skip)
