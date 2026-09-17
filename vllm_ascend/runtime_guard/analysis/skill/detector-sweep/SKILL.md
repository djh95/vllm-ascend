---
name: runtime-guard-detector-sweep
description: >-
  Use during NPU/vllm bug reproduction on a runtime_guard-instrumented
  vllm-ascend deployment. Enable all detectors simultaneously so the anomaly
  is caught directly — the detector that hits tells you the bug class, often
  pinpointing the problem without needing source patches or tensor dumps.
---

# runtime_guard Detector Sweep

## When to use
- About to reproduce an intermittent NPU bug (token repetition, NaN, garbled output, KV corruption)
- Reproduction rate is low (~20%), so each attempt is expensive — maximize info per attempt
- runtime_config at `{RG_CFG}` (hot-reloadable if `reload_interval_seconds > 0` or `additional_config.runtime_config_reload_interval > 0`); resolve `{CONTAINER}` / `{WS}` / `{REPRO}` from memory or case README

> Detectors run in **both eager and graph mode** (they hook logits/output/block-meta, not the dump path). `{MODE}` only matters for the dump step, not for the sweep.

## Available runtime_guard detectors

**Shipped on `feat/runtime-guard-config` today** (`DETECTOR_SECTIONS`):

| Key in config | Catches |
|---|---|
| `detector.logits_finite` | NaN / Inf in pre-sample logits |
| `detector.token_repeat` | Output-side repetition |
| `detector.output_substring` | Pattern in output (string or token-id list) |
| `detector.spec_acceptance` | Spec-decode acceptance anomaly (MTP on only) |

**Not in product defaults yet** (do not enable; ignore if seen in old notes):
`token_logprob`, `slot_consistency` / `kv_slot_token`, `kv_slot_order`, `kv_state`.

No dedicated "block write timing/order" detector yet — see "Block timing/order" section below.

## Why sweep all detectors
Each **shipped** detector targets a different bug class. Enabling all four costs little.
The detector that hits tells you the bug class — often without needing a full KV dump first.

## Procedure

1. Edit `{RG_CFG}` (hot-reloadable), e.g.:
   `docker exec {CONTAINER} bash -c 'vi {RG_CFG}'`
   Set these `enabled: true` (product paths under `detector.*`, not legacy `invariant.*`):
   - `detector.logits_finite.enabled` ← NaN/Inf catcher
   - `detector.token_repeat.enabled`
   - `detector.spec_acceptance.enabled` (only if spec decode / MTP on)
   - `detector.output_substring.enabled` (only if you have a target pattern)
   Prefer `report.max_per_req` / per-detector stop semantics from current product docs
   (legacy `stop_after_alert` may not exist).

2. Leave dump off initially (`dump.auto_max_times: 0` + `dump.manual_dump: false`). Goal here is detector hits, not full tensor / KV dumps. Optional: keep `on_trigger: ["report"]` only (no `dump_kv`) so reports land without consuming dump quota.

3. Wait `reload_interval_seconds` (default often 5s when hot-reload is on) — runtime_guard picks up the new config without restart.

4. **Reproduce multiple times** — a single curl is not enough; bug is intermittent (~20% repro rate). Run N iterations (default N=20, scale up if no hits): `docker exec {CONTAINER} bash -c '{REPRO}'`. Use the case's repro script (concurrent vs sequential curl per the original trigger).

5. Collect all hits across all N iterations:
   ```bash
   python -m vllm_ascend.runtime_guard.analysis.scripts.summarize_reports \
     --report-dir ./runtime/report --limit 50
   ```
   Or list files: `ls -t {WS}/runtime/report/*/report_*.json | head -50`.
   Read every report — note which `incident_type` fired, in what order, at which step/block_id/slot.

## Handoff

This skill covers **sweep** only — enable all detectors to catch the anomaly. Once the bug class is confirmed, hand off to `runtime-guard-investigation`: Step 5 (dump 现场) → Step 6 (ref capture) → Step 7 (compare, find first divergent).

### Sweep (this skill)
- Goal: discover which detector(s) catch the anomaly across multiple repro attempts.
- All detectors `enabled: true`, `stop_after_alert: false`, dump off (`dump.auto_max_times: 0` + `dump.manual_dump: false`).
- Run N=20+ curl iterations.
- Output: a list of (detector / incident_type, hit_count, first_hit_coordinates) across all iterations.

## Sweep → 下一步 决策树

Sweep 跑了 ≥ 5 次真命中 (去 FP 后) 看命中模式:

- **单 detector 一致命中** (5/5 同一个) → bug 类确认, 进 Step 5 dump 现场 → Step 6/7 ref 对比
- **多 detector 一致同 fired** (5/5 同组合, e.g. token_repeat + token_logprob.ill_nan) → 复合 bug 类, 进 Step 5 dump; 跨 detector 命中模式本身是定位线索
- **不同 attempt 命中不同 detector** → 复现不稳定 / 多 bug 并存 / 阈值需调. **不切**, 留 sweep 调阈值再跑
- **跑 5+ 次无命中** (curl 复现了但 detector 不响) → bug 类不在 runtime_guard detector 集, 回 Step 1 重判现象
- **全是 FP** → 阈值太宽, 走 `runtime-guard-config-recommender` 调参, **不切**

**不该切的模式**:
- 第 1 次命中就切 — 可能是 fluke
- 抓现场后还全开 detector — 配额被多 detector 同时命中占满
- 想靠 detector 触发 dump 抓 first divergent — detector 滞后, dump 是 post-bug 状态, 拿不到 first-wrong; first-wrong 靠 Step 7 从 report 的 `output_token_ids` 推 (不靠再抓一次现场)

## Interpreting hits

| Hit detector | Likely bug class | Next move |
|---|---|---|
| `logits_finite` | Numerical instability → NaN/Inf in forward pass | Arm `dump_kv` + ref compare; if KV clean, bug is post-KV (logits/sampling) |
| `token_repeat` | Output degeneration (could be KV corruption or sampling bug) | `dump_kv` + `locate_first_divergence` (buggy vs ref) |
| `spec_acceptance` | Spec decode mis-acceptance | Inspect draft/proposal scoring path |
| `output_substring` | Specific leak/echo pattern | Narrow down which token position the pattern starts |

**Multiple hits**: note order in report timestamps. First hit is most likely root cause, later ones are downstream consequences.

## NaN detector emphasis

`logits_finite` is the **direct NaN catcher** — checks if logits are finite at each decode step. If you suspect NaN:
1. Enable `detector.logits_finite.enabled`
2. Optionally also `token_repeat` / `output_substring` for co-symptoms
3. If `logits_finite` hits → arm `dump_kv`, `verify_request_kv`, then ref-compare
4. Inject path for live: `RG_INJECT=nan_logits` / `inf_logits` (see live §10)

## Block / slot meta (not shipped)

`slot_consistency` / `kv_slot_order` / `token_logprob` are **not** in current product
`DETECTOR_SECTIONS`. Do not recommend enabling them until config lands those keys.
KV corruption without a detector hit → rely on `manual_dump` / `dump_kv` + ref compare.

## Notes
- Hot-reload requires `reload_interval_seconds > 0` (or `runtime_config_reload_interval > 0`); if 0, restart D / worker
- **Reload 被拒保留旧配置**（JSONC 解析失败 / 未知 detector key / 数值类型错）— 改完 sweep 配置后 grep worker log 确认拾取
- Config accepts JSONC（`//` / `/* */` 注释 + 尾逗号）
- Detectors run on every step where they have hooks; CPU overhead is negligible vs NPU forward
- Reports include block_id/slot/step when applicable — coordinates for targeted dump
- Post-capture: `runtime-guard-analysis`；live 清盘见 §0.4

## Related skills
- `runtime-guard-investigation` — overall flow
- `runtime-guard-config-recommender` — thresholds / auto_max_times / budget
- `runtime-guard-analysis` — post-capture summarize / correlate / verify / inspect
