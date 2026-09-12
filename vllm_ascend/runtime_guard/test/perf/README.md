# runtime_guard perf（实卡 NPU · analysis 旁支）

> **归属**：`feat/runtime-guard-analysis`  
> **路径**：`vllm_ascend/runtime_guard/test/perf/`  
> **不在** `feat/runtime-guard-config` 维护（config 只留产品代码 + CPU UT）。  
> 全部对比均需 **Ascend NPU**；本机无卡只能改脚本/文档，不能出正式数字。
>
> **脚本落点**：下文 C1–C6 / 起服 / 交叉轮换等**现在没有的启动脚本与执行脚本，全部补到本旁支**  
> `test/perf/scripts/`（及所需 configs），禁止回流 config。

## ModelRunner v1 / v2（强制）

与 live 功能清单相同：**每一组 C 对比都必须分别在 v2、v1 上各跑完整交叉轮换**，结果分列，禁止合成一个比值。

| Runner | 启用 | 顺序 | 结果目录建议 |
|--------|------|------|----------------|
| **v2** | `VLLM_USE_V2_MODEL_RUNNER=1` | **先** | `logs/v2/` |
| **v1** | 关闭上述开关 | **后** | `logs/v1/` |

记录模板：`C1 | v2 | T1/T0=…` 与 `C1 | v1 | T1/T0=…` 两行。

启动脚本（待按环境补齐）见 `scripts/`：

```
test/perf/scripts/
  serve_t1.v2.sh / serve_t1.v1.sh
  serve_t2.v2.sh / …
  run_c1_c2_cross_rotate.sh   # 内含 RUNNER=v1|v2
  run_c3_ab.sh
```

---

# Terminology + required comparisons

> **Do not mix labels with `dfx-perf-bench` SKILL.** That skill uses A=no-DFX /
> B=DFX+reload>0+detector-off for a *different* project (PR 12209 DFX). Here
> A/B inside `perf_ab_quick.py` means detector on/off — different axis.
> This README is the authoritative source for the runtime_guard project.

## Terminal-state config labels (T-labels)

All perf tests are comparisons between two of these terminal states. A
"server" is described by which T-state it runs in.

| Label | Code | `--additional-config` | `reload_interval_seconds` | detector state | What it represents |
|-------|------|------------------------|---------------------------|-----------------|--------------------|
| **T0** | merge-base without runtime_guard product bind | — | — | — | Pure upstream; no `RuntimeGuardProcessor.bind` |
| **T1** | product HEAD (config branch build) | none (plain `vllm serve ...`) | `0` (default) | all off | Guard infra loaded but everything off |
| **T2** | product HEAD | `runtime_config_path=...` + reload interval | `3` | all off | Hot-reload ON, detectors off |
| **T3** | product HEAD | same as T2 | `3` | 4 detectors on | Hot-reload + active detection |

Note: T0 needs a worktree/checkout of a commit **without** product bind.
csrc usually unchanged → often **no `.so` rebuild**. Confirm merge-base SHA on the
machine before quoting numbers.

## Required comparisons（每条 × runner v1 + v2）

| ID | Compare | What it proves | Status |
|----|---------|----------------|--------|
| **C1** | T0 vs T1 | Guard infra zero-ish overhead when default-off | ⏳ pending · **须分 v1/v2** |
| **C2** | T1 vs T2 | Hot-reload poll shell cost | ⏳ pending · **须分 v1/v2** |
| **C3** | T2 vs T3 | Detector enable cost | 部分历史数 · **须分 v1/v2 重测** |
| **C4** | Functional isolation | `temp=0` outputs bit-identical across T0–T3 | ⚠️ partial · **须分 v1/v2** |
| **C5** | T3 vs T3+`dump_kv` on_trigger | Dump arm/D2H 开销（命中路径） | ☐ 未建 · **须分 v1/v2** |
| **C6** | Idle leak-back | 压测后 RSS/HBM 回落（见 `perf_lib` leakback） | ☐ 未建正式门禁 · **须分 v1/v2** |

## 0.999 target — what each comparison must hit

| Target | Comparison | Bar |
|---------|-----------|-----|
| Guard infra zero-cost | C1 | T1/T0 ≥ 0.999（**v1、v2 各自**） |
| Hot-reload acceptable | C2 | T2/T1 ≥ 0.999 |
| Detector acceptable | C3 | T3/T2 ≥ 0.990 |
| Dump path acceptable | C5 | 约定阈值下相对 T3 跌幅可接受（阈值待定） |

## Cross-rotation methodology (mandatory for C1/C2)

All A/B comparisons between terminal states must be run **cross-rotated
(交叉轮换)**, never serial. Serial (all rounds of state A, then all rounds of
state B) makes the first state look faster because it benefits from cold-start
and NFS page-cache warm-up.

**Rule**: configs alternate round-by-round — `A → B → A → B → ...` (or
`A → B → C → A → B → C → ...` for 3-state), N ≥ 3 rounds per state (use N=6
for the final C1/C2 gate). Ratio from **per-state geometric mean across rounds**.

For C1+C2 combined, rotate three states in one pass:
`T0 → T1 → T2 → T0 → T1 → T2 → ...` (6 cycles). Derive `C1 = T1/T0`,
`C2 = T2/T1` from the same interleaved data.

**Then repeat the entire rotation under the other ModelRunner.**

## Scripts

| Script | What it measures | Output env var |
|--------|------------------|------|
| `perf_baseline.py` | T1 (or T0 from merge-base worktree) | `RG_PERF_OUT_BASELINE` |
| `perf_ab_quick.py` | C3 (T3 "B" vs T2 "A") | `RG_PERF_OUT_AB` |
| `test_refresh_config_cost.py` | CPU 微基准：reload=0 early-return（可选 CI） | — |
| `scripts/*.sh` | 起服 / 交叉轮换封装（**待按机房补**） | 写入 `logs/{v1,v2}/` |

Env defaults live in `perf_lib.py` (`RG_PERF_URL`, `RG_PERF_CFG`, `RG_PERF_NPU`, …).
Override per lab; do not commit secrets.

## Known measurement pitfalls

- **First round is faster**: drop warmup (`warmup(rounds=1)`).
- **`pkill -f 'vllm serve'` matches own shell**: use `pgrep -f '[v]llm serve'` + PID kill from `npu-smi`.
- **TP workers survive parent kill**: kill worker PIDs from npu-smi.
- **Cross-session noise**: ±1% normal; C1/C2 must use cross-rotation.
- **Runner mix-up**: forgetting `VLLM_USE_V2_MODEL_RUNNER` → invalid row; label every jsonl line with `runner=v1|v2`.
- **T0 worktree**: current-branch `PYTHONPATH` ≠ T0.

## Cannot automate without NPU

No Ascend NPU in typical agent/CI → C1–C6 are **manual NPU runs**.  
`test_refresh_config_cost.py` may run on CPU as a smoke for early-return cost only.
