---
name: runtime-guard-config-recommender
description: >-
  Recommend optimal runtime_config parameters based on observed bug phenomenon.
  Use when the user describes a bug and you need to recommend which detectors to
  enable, what thresholds, auto_max_times, auto_cooldown_seconds, manual_dump,
  stop_after_alert, on_trigger (report + dump_kv), etc. for runtime_config.json.
  Avoids trial-and-error config tuning. Native dump_kv only — no msprobe.
---

# runtime_guard Config Recommender

## When to use
- User describes a bug phenomenon (output repeat, NaN, garbled chars, KV corruption, etc.)
- Need to generate `runtime_config.json` tailored to that bug class
- Need to compute `auto_max_times` (dump_kv quota) from repro rate + disk budget
- Phase transition: sweep (all detectors, `stop_after_alert=false`) → Step 5 抓现场 (`dump.auto_max_times>0` + `on_trigger` includes `dump_kv`)

## Procedure

### Step 1: Classify bug phenomenon

Use `runtime-guard-investigation` Step 1 decision tree to reach a leaf (E1-E9 or NOT-runtime_guard).

### Step 2: Recommend detector config per bug class

| Bug class | Detector(s) | Key params | Notes |
|---|---|---|---|
| **E1 token repeat** | `token_repeat` | `window: 32`, `repeat_sum_threshold: 64`, `min_tokens: 32`, `consecutive_hits: 1`, `ignore_token_ids: []` | 最轻量; 复读/退化首选; `on_trigger: ["report","dump_kv"]` |
| **E2 生僻字** | `token_logprob` | `ill_rare_window_thresh: 1`, `window: 64`, `stride: 32`, `topk: 20` | 需 worker 自动补 top-k logprobs |
| **E3 乱码** | `token_logprob` | `ill_garbled_window_thresh: 1` | 同上 |
| **E4 NaN/Inf** | `logits_finite` + `token_logprob` 双开 | `logits_finite.enabled: true`; `token_logprob.ill_nan_window_thresh: 1` | `logits_finite` 抓 forward logits NaN; `token_logprob` 抓 post-softmax |
| **E5 含特定字符串** | `output_substring` | `patterns: ["..."]` (string 或 list[int] token id), `match_prefix: false` 或 `true` | 须知要找什么; 每 req 告警一次 |
| **E6 spec 接受率异常** | `spec_acceptance` | `window: 10`, `low_threshold: 0.3`, `high_threshold: 0.96`, `len_low_threshold: 1.4`, `len_high_threshold: 2.8` | 仅 spec decode (MTP/Eagle) on 时有效 |
| **E7 KV block 元数据错** | `block_kv` | 默认 `check_wave_regression: true`, `check_same_wave_writer: true` | 检 block write wave 单调性 + writer 冲突; 必开 dump_kv |
| **E8 position 错位** | `position_alignment` | (查源码 `position_alignment.py` 当前无阈值) | 仅 1-D RoPE text path |
| **E9 PD 分离未定类** | 全开 sweep | 全 `enabled: true`, `stop_after_alert: false` | 走 `runtime-guard-detector-sweep` |
| **定界 / 逐层对比** | (无新 detector) | 抓 buggy + ref `dump_kv` 后跑 `locate_first_divergence` / `compare_per_layer` | 见 `runtime-guard-ref-kv-dump` |

### Step 3: Compute `auto_max_times` from repro rate

公式：`auto_max_times = N_expected_repros × buffer × safety_factor`

- N_expected_repros: 用户报告的复现次数目标 (e.g. 抓 3 次现场对比)
- buffer: 复现率倒数 (复现率 20% → buffer=5)
- safety_factor: 1.5 (避免配额耗尽前没抓到)

例: 复现率 20%, 想抓 3 次现场 → `auto_max_times = 3 × 5 × 1.5 = 22.5` ≈ 25

**也要看 Step 4b dump_kv 盘预算**: 取两个公式的**较小者**. 不够就降 `N_expected_repros` 或保持 `dump_all_blocks: false`.

### Step 4: Set `auto_cooldown_seconds`

- detector 命中后, 两次 auto dump_kv 之间最小间隔
- 默认 300s (5 min). 复现率高 (>30%) → 调小到 60-120s 避免抓不到下一次
- 复现率低 (<10%) → 保持 300s 或更大, 避免冷却占满 auto_max_times 配额

### Step 4b: dump_kv size budget (与 Step 3 auto_max_times 联动)

```
auto_max_times ≤ (disk_free × 0.7) / per_req_kv_size
```

- `per_req_kv_size`: 实测一次 `{report_dir}/kv_cache/<type>/<req_id>/` 目录大小（`du -sh`）
- 多进程各写一份时再按写盘 rank 数放大分母
- 磁盘紧时降级路径 (按顺序):
  1. `scope: request` + `dump_all_blocks: false`（默认，只切本请求 block）
  2. 减小 `auto_max_times` / `N_expected_repros`
  3. 提高 `auto_cooldown_seconds` 降低误报连发

实测见 `runtime-guard-investigation` Step 4.

| 场景 | stop_after_alert | 原因 |
|---|---|---|
| **抓现场 dump_kv (默认)** | `false` | 一个 req 检出后继续检, 看跨 detector 命中模式; 每个命中可触发 dump (配合 auto_max_times 限配额) |
| **只要一次 dump** | `true` | 已确认 bug 类, 一次命中就够, 避免重复写 report / 占配额 |

### Step 6: Always-on baseline fields

**启动时 — dump 配额关**（先确认服务 / detector 正常）:
```json
{
  "reload_interval_seconds": 5,
  "dump": {
    "auto_max_times": 0,
    "manual_dump": false,
    "dump_all_blocks": false
  },
  "detector": {
    "stop_after_alert": true,
    "<detector_name>": {
      "enabled": true,
      "on_trigger": ["report"],
      "<Step 2 阈值>": "..."
    }
  },
  "report": {
    "save_sensitive_info": true,
    "decode_token_ids": true
  }
}
```

**服务起来后热更到压测态** (改 `{RG_CFG}`):
```json
{
  "reload_interval_seconds": 5,
  "dump": {
    "auto_max_times": "<Step 3 计算值>",
    "auto_cooldown_seconds": "<Step 4 选择>",
    "manual_dump": false,
    "dump_all_blocks": false
  },
  "detector": {
    "stop_after_alert": false,
    "<detector_name>": {
      "enabled": true,
      "on_trigger": ["report", "dump_kv"],
      "dump_kv": { "scope": "request", "dump_all_blocks": false },
      "<Step 2 阈值>": "..."
    }
  }
}
```

> **schema 要点**: 没有 `dump.enabled` 字段 — dump 活跃派生自 `auto_max_times>0 || manual_dump active`.
> - Native `dump_kv` action 由 `on_trigger` 触发; 配额走 `auto_max_times` / cooldown.
> - `dump.auto_max_times>0` 与 `dump.manual_dump` active 互斥 → 报错.

**配置格式与校验（热更安全）**:
- 支持 **JSONC** — `//` / `/* */` 注释、尾逗号都合法. 起步直接复制 shipped 模板
  `vllm_ascend/runtime_config/templates/runtime_config.example.jsonc`（模板本身可直接 load + validate）.
- **未知 key 拒绝**: detector 子 key 拼错（如 `windw`）→ reload 整体**失败并保留旧配置**，不再静默按默认值跑.
  改完配置行为没变时，先 grep worker log 的 `unknown key` / reload 失败信息，别急着调阈值.
- 数值字段严格类型校验: int 字段拒绝 bool/None/字符串/非整 float（`2.7` 不会静默变 `2`）; 阈值 float 有上下界（0..1 等）.
- `sync_mode` 首次 apply 后**冻结**，热更改不动（DP collective 安全）— 需要换同步模式要重启.

**禁忌**:
- `auto_max_times>0` 同时 `manual_dump` active — 互斥报错, 二选一
- 单 detector 时 `stop_after_alert: false` + `auto_max_times: 1000` — 命中后该 req 持续重检占满配额
- `manual_dump: true` (持续 arm) — 易一次耗光配额; 压测用正整数 (e.g. `2`) 限定拍数, `true` 仅限调试临时用

**`report.save_sensitive_info` 必须开** (Step 5/6/7 依赖): report 里的 `output_token_ids` / `prompt_token_ids` 默认**不落盘**. ref force-feed 与 first-wrong 定位都依赖 ids — 抓现场前确认已开。

### Step 7: Cross-rank / sync caveats

- runtime_guard detector 只在 worker (TP0 ∧ PP0 ∧ last-PP) 跑; dump_kv 经同步在 leader 写盘.
- TP>1 时, **同 EngineCore 内各 TP 必须读同一份 `{RG_CFG}`** (同 path), 否则 idle fast-path 一侧跳过 OR, 另一侧进 `all_reduce` 会挂死.
- 跨 DP 不自动同步 config — 改两边各自可读的 JSON.

## Output

生成完整 `runtime_config.json`, 标注每个字段为什么这样选, 指向 `runtime-guard-detector-sweep` (sweep 找 bug 类) 或 `runtime-guard-investigation` Step 5 (抓现场 dump_kv).

## Related skills
- `runtime-guard-investigation` — overall flow; this skill supports Step 5 (stress test config) 和 sweep→dump 切换.
- `runtime-guard-detector-sweep` — sweep workflow; this skill generates its config.
- `runtime-guard-analysis` — post-capture summarize / correlate / verify / inspect.
- `runtime-guard-ref-kv-dump` — ref capture + two-table compare after verify PASS.
