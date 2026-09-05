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

| Key in config | Class | Catches |
|---|---|---|
| `logits_finite` | `LogitsFiniteDetector` | **NaN / Inf in logits** — the numerical-instability catcher |
| `token_logprob` | `TokenLogprobDetector` | Rare/ill-conditioned token probabilities; also has `ill_nan_window_thresh` for NaN-in-logprob detection |
| `token_repeat` | `TokenRepeatDetector` | Output-side repetition (loops, n-gram stuck) |
| `block_kv` | `BlockKvDetector` | Per-block KV cache sanity — KV corruption |
| `position_alignment` | `PositionAlignmentDetector` | Position-id / slot-mapping / block-table off-by-one |
| `spec_acceptance` | `SpecAcceptanceDetector` | Speculative-decode acceptance rate anomaly (only if spec decode on) |
| `output_substring` | `OutputSubstringDetector` | Specific pattern in output (leak, echo, prompt injection) |

No dedicated "block write timing/order" detector yet — see "Block timing/order" section below.

## Why sweep all detectors
Each detector targets a different bug class. If only one is enabled and it misses, the repro attempt is wasted. Enabling all costs ~nothing (CPU-side checks on already-computed tensors). The detector that hits tells you the bug class directly — often pinpoints the problem without source patches or tensor dumps.

## Procedure

1. Edit `{RG_CFG}` (hot-reloadable), e.g.:
   `docker exec {CONTAINER} bash -c 'vi {RG_CFG}'`
   Set ALL of these `enabled: true`:
   - `detector.logits_finite.enabled` ← NaN catcher
   - `detector.token_logprob.enabled` (also set `ill_nan_window_thresh: 1`)
   - `detector.token_repeat.enabled`
   - `detector.block_kv.enabled`
   - `detector.position_alignment.enabled`
   - `detector.spec_acceptance.enabled` (only if spec decode on)
   - `detector.output_substring.enabled` (only if you have a target pattern)
   Also set `detector.stop_after_alert: false` so detectors keep running after first hit (catch multiple anomalies per run — the first hit is most likely root cause, later ones are downstream symptoms).

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
| `token_logprob` (NaN flag) | NaN in logprob computation (post-softmax) | Same — closer to sampling head if KV matches ref |
| `token_repeat` | Output degeneration (could be KV corruption or sampling bug) | `dump_kv` + `locate_first_divergence` (buggy vs ref) |
| `block_kv` | KV cache corruption — **smoking gun for KV bug** | Go straight to `dump_kv` + per-layer compare vs ref |
| `position_alignment` | Position-id / slot-mapping / block-table off-by-one | Inspect report coordinates + request `block_ids` in dump_kv; check P→D boundary |
| `spec_acceptance` | Spec decode mis-acceptance | Inspect draft/proposal scoring path |
| `output_substring` | Specific leak/echo pattern | Narrow down which token position the pattern starts |

**Multiple hits**: note order in report timestamps. First hit is most likely root cause, later ones are downstream consequences.

## NaN detector emphasis

`logits_finite` is the **direct NaN catcher** — checks if logits are finite at each decode step. If you suspect NaN anywhere:
1. Enable `logits_finite` first
2. Also enable `token_logprob` with `ill_nan_window_thresh: 1` (catches NaN-in-logprob specifically, in case logits look finite but probs are NaN)
3. If `logits_finite` hits → arm `dump_kv`, verify, then ref-compare; if KV diverges, first bad layer from `locate_first_divergence`; if KV matches, bug is closer to logits/sampling
4. If only `token_logprob` hits → bug is in logprob computation or sampling, not in forward KV write

## Block timing / order (no dedicated detector yet)

If you suspect a race condition (block written by P after D already read it, double-written, or out-of-order block reuse):
- Use `block_kv` — it catches the consequence of ordering bugs (stale or corrupted KV)
- Capture `dump_kv` on hit; inspect report `block_ids` / wave metadata and compare buggy vs ref with `runtime-guard-ref-kv-dump`
- Future work: a custom detector that records block write timestamps and flags out-of-order writes — would need a new `BlockOrderDetector` in `vllm_ascend/runtime_guard/detector/`

## Notes
- Hot-reload requires `reload_interval_seconds > 0` (or `runtime_config_reload_interval > 0`); if 0, restart D / worker
- **Reload 被拒保留旧配置**（JSONC 解析失败 / 未知 detector key / 数值类型错）— 改完 sweep 配置后 grep worker log 确认拾取；命中模式没变化先怀疑 reload 没生效，再怀疑阈值
- Config accepts JSONC（`//` / `/* */` 注释 + 尾逗号）
- Detectors run on every step where they have hooks; CPU overhead is negligible vs NPU forward
- `stop_after_alert: false` is critical for sweep — by default a single hit may stop further detection
- Reports include block_id/slot/step when applicable — these are the coordinates you need for targeted dump
- Post-capture offline read of reports/KV: `runtime-guard-analysis` (`summarize_reports` / `correlate_incident` / `verify_request_kv` / `inspect_kv_dump`)

## Related skills
- `runtime-guard-investigation` — overall flow
- `runtime-guard-config-recommender` — thresholds / auto_max_times / budget
- `runtime-guard-analysis` — post-capture summarize / correlate / verify / inspect
