# runtime_guard perf terminology + required comparisons

> **Do not mix labels with `dfx-perf-bench` SKILL.** That skill uses A=no-DFX /
> B=DFX+reload>0+detector-off for a *different* project (PR 12209 DFX). Here
> A/B inside `perf_ab_quick.py` means detector on/off — different axis.
> This README is the authoritative source for the runtime_guard project.

## Terminal-state config labels (T-labels)

All perf tests are comparisons between two of these terminal states. A
"server" is described by which T-state it runs in.

| Label | Code | `--additional-config` | `reload_interval_seconds` | detector state | What it represents |
|-------|------|------------------------|---------------------------|-----------------|--------------------|
| **T0** | merge-base `37e382498` (NO runtime_guard code) | — | — | — | Pure upstream vllm-ascend; no `RuntimeGuardProcessor.bind` |
| **T1** | current HEAD | none (plain `vllm serve ...`) | `0` (default) | all off | Guard infra loaded but everything off (production "set-and-forget" shape) |
| **T2** | current HEAD | `runtime_config_path=...` + `runtime_config_reload_interval=3` | `3` | all off | Guard infra + hot-reload ON, detectors off (test harness shape) |
| **T3** | current HEAD | same as T2 | `3` | 6 enabled (spec_acceptance/output_substring/token_repeat/block_kv/position_alignment/logits_finite) | Guard + hot-reload + active detection |

Note: T0 requires `git worktree add <path> 37e382498` (or `git checkout`).
csrc is unchanged between merge-base and HEAD, so **no `.so` rebuild needed**.

## Required comparisons

| ID | Compare | What it proves | Status (2026-09-05) |
|----|---------|----------------|------|
| **C1** | T0 vs T1 | Guard infrastructure (`RuntimeGuardProcessor.bind` + `sync_for_step` + `refresh_config` early-return + hook installation) has zero overhead in default-off state | ❌ **NOT DONE** — never checked out merge-base on a live server |
| **C2** | T1 vs T2 | Hot-reload poll shell (per-step `sync_for_step` / interval gate; collective only near due) | ✅ ~0.4% pre-opt (perf_baseline=8.12 vs perf_ab A=8.09); re-measure after idle not-due short-circuit |
| **C3** | T2 vs T3 | Detector enable cost (within reload=3) | ✅ 0.6% (perf_ab A=8.09 vs B=8.04) |
| **C4** | Functional isolation | `temp=0` outputs bit-identical across T0–T3 | ⚠️ partial (B14 verified T2/T3 only; T0/T1 not bit-compared) |

## 0.999 target — what each comparison must hit

| Target | Comparison | Bar | Currently |
|---------|-----------|-----|----------|
| Guard infra zero-cost | C1 | T1/T0 ≥ 0.999 | ❌ unmeasured |
| Hot-reload acceptable | C2 | T2/T1 ≥ 0.999 | ❌ 0.996 (0.4% gap, fails) |
| Detector acceptable | C3 | T3/T2 ≥ 0.990 | ✅ 0.994 (0.6%, within typical noise) |

## C1 gap — why it matters and what's blocking

C1 is the most fundamental perf question: **does loading the guard infrastructure cost anything when everything is off?**

The ~0.4% gap previously measured in C2 is **NOT** a per-step `all_reduce`
cost. With skew / `reload_clearly_not_due`, collectives run only near the
reload interval; the residual is mostly the Python idle shell on T2
(`sync_for_step` → refresh/consume when due, advance-only when not due).
C1 measures the guard-infra-only overhead (T1 early-return / advance-only)
and is the right experiment for "guard adds zero overhead in default-off
startup."

To run C1:
1. Stop current guard server (cards 6-7)
2. `cd /workspace/vllm-ascend && git worktree add /tmp/rg-perf-base 37e382498`
3. From worktree: `vllm serve ...` (plain, no `--additional-config`)
4. Wait health=200
5. Run `perf_baseline.py` (point `RG_PERF_OUT_BASELINE` at a separate path to
   not clobber existing T1 data) → records T0 tps
6. `kill_server` + `git worktree remove /tmp/rg-perf-base`
7. Restart T1 server (current HEAD, no additional-config) — OR reuse existing
   `perf_baseline.jsonl` if environment matches (same NPU, same time window)
8. Compare T0 vs T1: expect gap ≤0.1% if guard infra is truly zero-cost

If C1 ≤0.1%: guard infra is clean; C2's residual is hot-reload poll cost,
which production can avoid by setting `reload_interval=0` (then prod == T1 ≈ T0).

If C1 >0.1%: guard infra itself has cost; profile `sync_for_step` even on the
advance-only / early-return path.

## Scripts

| Script | What it measures | Output env var |
|--------|------------------|------|
| `perf_baseline.py` | T1 (or T0 if run from merge-base worktree) | `RG_PERF_OUT_BASELINE` (default `.../logs/perf_baseline.jsonl`) |
| `perf_ab_quick.py` | C3 (T3 "B" vs T2 "A" within reload=3) | `RG_PERF_OUT_AB` (default `.../logs/perf_ab_quick.jsonl`) |

Both pull URL / config paths from `perf_lib.py` env vars (see top of that file).

## Known measurement pitfalls

- **First round is faster**: drop warmup (handled by `warmup(rounds=1)` in both scripts).
- **`pkill -f 'vllm serve'` matches own shell**: use `pgrep -f '[v]llm serve'` bracket trick + PID kill from `npu-smi info | grep VLLMWorker`.
- **TP workers survive parent kill**: explicitly kill worker PIDs from npu-smi.
- **Cross-session noise**: ±1% is normal across sessions; for tight comparisons
  (C1, C2), run both states in the same session back-to-back.
- **`run_rg.sh baseline` ≠ T0**: that script still points `PYTHONPATH` at the
  current branch, so the runtime_guard code IS loaded — it's T1, not T0.
  T0 requires checkout merge-base.

## Cannot automate here

No Ascend NPU in the agent environment. All C1–C4 are manual NPU-side runs
(except C2/C3 which are already done and recorded).
