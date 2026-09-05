# SPDX-License-Identifier: Apache-2.0
"""runtime_guard / runtime_config test matrix (adapted from skills/test, native dump_kv).

Priorities: **P0** every PR / smoke; **P1** full suite; **P2** env-dependent.

## P0 — correctness & isolation (must not affect inference)

| ID | Area | What | Expect |
|----|------|------|--------|
| I1 | Isolation | Code present but **no** `--additional-config` / default reload=0, detectors off | Same tokens as baseline (temp=0); TPS within noise |
| I2 | Isolation | Feature wired, detectors off, dump off, reload=0 | Same as I1 (noop hot path) |
| I3 | Isolation | reload>0, all detectors off, dump off (**B** in perf skill) | TPS ≈ I1 (≤~1–2% typical); no report spam |
| I4 | Soft-fail | Malformed JSON on hot-reload | Keep old config; service alive |
| I5 | Soft-fail | Unknown / wrong-type fields | Soft warn; no crash |
| I6 | dump_kv | Empty `block_ids` + `dump_all_blocks=false` | **No** full-cache D2H; skip / no dump |
| I7 | Output | Report / print_output must not mutate sampler outputs | HTTP body unchanged vs I1 |
| O1 | Output integrity | `print_output_on_finish` text vs HTTP (no MTP) | String-equal (trim) |

## P0 — function smoke (UT + short e2e)

| ID | What | Expect |
|----|------|--------|
| S1 | Config load defaults | `dump_enabled` false; detectors off |
| S2 | Hot-reload mtime/content | `sync_runtime_config` / `reload` picks detector enable |
| S3 | manual_dump / manual_trigger | Report + optional dump_kv files under `kv_cache/` |
| S4 | token_repeat detector (unit) | Hits on synthetic repeats; miss on unique ids |
| S5 | Wave + quota | wave stamp / consume / cooldown |
| S6 | ActionQueue | Async commit; full→inline fallback |
| S7 | Analysis scripts | summarize / verify / locate help + synthetic compare |

## P1 — detectors / report / dump_kv

| ID | What | Expect |
|----|------|--------|
| T1–T3 | output_substring / token_repeat / logits_finite | Hit/miss; stop_after_alert |
| R1–R6 | block_ids / sensitive ids / truncate | Schema matches config flags |
| D1–D5 | dump_kv scope=request, quota, cooldown | Files only for armed req; quota respected |
| C1–C4 | v1 vs v2 runner hook parity | Same report fields / detector hits |

## P1 — performance (live NPU; see dfx-perf-bench rotation)

| Label | Config | Claim |
|-------|--------|-------|
| A | No additional-config | Baseline TPS |
| B' | Framework on, reload=0, detectors off | ≈ A |
| B | reload>0, detectors off, dump off | ≈ A (hot-reload only) |
| B+det | one light detector (token_repeat) | small CPU overhead |
| B+dump | dump_kv armed rarely | amortized; spike only on hit |

Pass: mean TPS(B)/TPS(A) ≥ 0.98 (or within measured noise on that SKU); A vs pre-PR A' within noise.

## P2 — stress / DP / MTP

| ID | What |
|----|------|
| X* | DP>1 sync, long prefill dump size |
| O2+ | MTP output integrity (known risk area) |

## P0 — isolation / with-vs-without (what CI can and cannot prove)

| Experiment | In this repo UT? | How |
|------------|------------------|-----|
| **A0** Hook present, reload=0, detectors/dump off → `refresh_config` ≪ 1 ms | **Yes** | `test_hot_path_overhead.py` |
| **A1** reload&gt;0, detectors/dump off → bounded CPU | **Yes** | same |
| **A2** Empty `block_ids` never full-cache D2H | **Yes** | `test_detectors_and_kv.py` |
| **A3** Soft-fail bad JSON | **Yes** | `test_runtime_config_core.py` |
| **E1** Live NPU: **no PR code** (main / without bind) vs **PR + no additional-config** | **No (needs NPU + two builds)** | See `tests/perf/runtime_guard/README.md` |
| **E2** Live: PR + reload=0 vs PR + reload&gt;0 detectors off | **No (needs live server)** | same README + skills/test `dfx-perf-bench` |
| **E3** Live: output tokens identical temp=0 across A/B | **No (needs live server)** | curl + compare |

CI proves **CPU hot-path bounds** and **functional isolation**.  
**Throughput parity with/without the patch** is **E1/E2** only — not runnable in this Mac/CI env without Ascend.

## P0 — review regression suite (2026-09-01 white-box review; `test_review_regressions.py`)

| ID | Finding | What | Expect |
|----|---------|------|--------|
| V1 | P0-4 | `matched_layers` ordering | Natural sort: layer_2 < layer_10 (first-divergence marker trustworthy) |
| V2 | P0-5 | `dumps_report_json` | np.int64 / torch scalar / NaN never lose the report |
| V3a–d | P0-1 | soft-fail contract | Detector/hook exceptions never reach engine loop / async copy thread / sampler |
| V4 | P0-2 | Shipped `runtime_config.example.jsonc` | Loads + validates as-is; reload(force) succeeds |
| V5 | P0-3 | Bootstrap invalid content | Falls back to defaults; service starts |
| V6 | C1 | `sync_mode` hot-reload | Frozen at first apply (DP collective safety) |
| V7 | A4 | `classify_violation` labels | wrong_start / non_consecutive / misaligned correct; gaps dominate |
| V8a/b | B2 | Wave stamps lifecycle | discard on reap; no unbounded `_sample_waves` growth |
| V9a/b | B3 | ActionQueue full/stop | Heavy (dump) jobs dropped, never inline; stop works with full queue (drain + sentinel) |
| V10 | C5 | Unknown detector sub-key | Reload rejected loudly (typo like `windw` no longer silently defaults) |
| V11 | A3 | token_logprob window hot-resize | Buffers rebuilt (`maxlen`), hit counters reset |
| V12/12b | A1 | logits_finite attribution | Unattributable rows alert (`unresolved_row_to_request`), never misattribute; decode rows still per-request |
| V13 | P0-2 | JSONC parsing | `//`, `/* */` comments + trailing commas accepted (string-aware) |
| V14/14b | B'4 | DumpQuota | Atomic `try_consume` (cap + cooldown); blocked consume doesn't burn; `refund` |
| V15/15b | B'2 | ReportWriter dedupe | Same (type, req) cooldown + cap; different req unaffected; cooldown expires |
| V16/16b | — | detector `clear_finished` | Per-req state (buf/history) fully dropped, no cross-request leaks |
| V17 | B9 | spec_acceptance short batch | accepted shorter than req_ids: no IndexError |
| V18 | B'6 | dump payload | Carries `tp_rank` / `num_kv_heads` for cross-TP comparison |

Related behavior contracts changed by the review fixes (mirrored in old UTs):
`_slice_blocks` returns `(tensor, used_ids)` (B'5c payload alignment); ActionQueue
`submit(job, heavy=)` replaces `sync_fallback=` (light → inline fallback, heavy → drop);
Store-side `sample_waves` FIFO removed — drain gating via `WaveTracker.pending` probe.
