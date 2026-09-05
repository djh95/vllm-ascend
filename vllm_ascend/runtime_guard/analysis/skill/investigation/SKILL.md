---
name: runtime-guard-investigation
description: >-
  Use when debugging an intermittent NPU/vllm-ascend bug (token repetition,
  garbled output, NaN, KV corruption) on a runtime_guard-instrumented deployment.
  Captures the end-to-end investigation workflow with native dump_kv only:
  triage → repro → env setup → dump_kv sanity → stress + capture → verify →
  ref dump_kv → compare_kv_similarity / locate_first_divergence / compare_per_layer. No msprobe.
---

# runtime_guard-Based Bug Investigation (native dump_kv only)

This skill is a living document — refine it as each case teaches new lessons.

正文路径：`vllm_ascend/runtime_guard/analysis/skill/investigation/SKILL.md`  
分析脚本：`python -m vllm_ascend.runtime_guard.analysis.scripts.<module>`  
抓现场后的离线汇总/对账：见 `runtime-guard-analysis`。  
抓 ref + 两表对比：见 `runtime-guard-ref-kv-dump`。

**Dump 路径只有 native `dump_kv`。** 不要走 msprobe / PrecisionDebugger / AclGraphDumper / `dump_tensor_data` / msprobe-target-config / `/Users/djh/code/dfx`。

## Required inputs (resolve BEFORE Step 1)

Each investigation needs 4 inputs. Resolve from memory (`project_env_layout` / `project_pd_arch`) or ask the user; do NOT assume or reuse another case's values:

1. **运行环境** — container name + in-container paths + `{MODE}` + `{MODEL}` + `{TOPO}` + NPU cards (→ Case context table below).
2. **启动脚本** — how the service is started (`{START}`).
3. **请求脚本** — how the bug is triggered (`{REPRO}`).
4. **谁来跑服务** — two modes:
   - **skill 跑**: skill runs `{START}` + `{REPRO}` via `docker exec {CONTAINER}`.
   - **用户自己跑**: user starts the service and sends requests themselves; skill only reads reports/dumps and orchestrates analysis.

Confirm mode 4 at the start. If the user prefers to drive startup/requests, don't `docker exec` the scripts — ask them to run it and report readiness/results.

## Case context (fill in per case, do NOT reuse another case's values)

| Var | Meaning | Example (Case 4) | Example (Case 3) |
|---|---|---|---|
| `{CONTAINER}` | container name | `case_4` | `wc_test` |
| `{WS}` | in-container workspace (scripts / runtime_guard / my edits) | `/workspace/djh-testcase4` | `/workspace` (scripts) + `/workspace/djh-testcase3` (my edits) |
| `{SRC}` | vllm-ascend source tree in container | `/workspace/djh-testcase4/vllm-ascend-bak` | `/vllm-workspace/vllm-ascend` |
| `{RG_CFG}` | runtime_guard config location (separate file vs inline `--additional-config`) | separate `runtime/config/runtime_config.json` | inline in `run_dsv2.sh` / additional-config |
| `{START}` | startup script (启动服务) | `scripts/run_dsv2_as_p.sh` + `scripts/run_dsv2_as_d.sh` | `bash run_dsv2.sh` |
| `{REPRO}` | request script (触发 bug 的请求) | `scripts/curl.sh` via `scripts/repro_loop.sh` (concurrent curl) | `bash curl_dsv2.sh` (10× sequential) |
| `{MODE}` | compilation mode | graph `FULL_DECODE_ONLY` | eager (`--enforce-eager`) |
| `{DUMP}` | dump mechanism (always native) | `dump_kv` | `dump_kv` |
| `{REPORT}` | anomaly report root | `{WS}/runtime/report` | `{WS}/runtime/report` |
| `{KV_ROOT}` | native KV dump root under report | `{REPORT}/kv_cache` | `{REPORT}/kv_cache` |
| `{MODEL}` | model / weights path | DeepSeek-V2-Lite | DeepSeek-V2-Lite |
| `{TOPO}` | single-process vs PD-separated | PD-separated (P TP=2 + D DP=2 TP=1) | single-process TP=4 |

Start every investigation by resolving the 4 required inputs above (from memory or by asking the user). All `{...}` placeholders below are these vars. Don't skip the "谁来跑服务" question — don't auto-run the service unless the user wants you to.

## Investigation goal

Find the **first divergent point** between the buggy runtime KV and a reference `dump_kv`, then map it to a root cause (layer × token × slot / transfer boundary).

Two-phase refinement (both on native `.pt` dirs):
1. **Phase E.1 (default first)**: Find first **KV cache** divergence — `locate_first_divergence` on buggy vs ref dirs. Locate first divergent KV point (token × layer).
2. **Phase E.2 (only if KV nearly identical)**: Bug is likely **post-KV** (logits / sampling / decode path after write). Confirm with `compare_per_layer`; then lean on detector class + report `output_token_ids` — do **not** escalate to msprobe.

## Hard constraints (do not violate)

- **Don't switch `{MODE}` as a "control experiment".** Eager vs graph change the numeric path. If the bug reproduces in one mode, stay there. A failed repro in the other mode is NOT evidence (too few samples + different path).
- **Dump only via runtime_guard native `dump_kv`.** No source-patch `torch.save`. No msprobe / PrecisionDebugger / AclGraphDumper / `dump_tensor_data`.
- **Don't trust prior-session claims without re-verifying.** Always `docker exec {CONTAINER} ...` to confirm current state.
- **Dump / quota / path issues**: fix `{RG_CFG}` (`on_trigger`, `auto_max_times`, `report_dir`) and disk; don't invent external dumpers.

## Dump story (single pipeline)

```
Online (buggy):
  detector hit
    → on_trigger: ["report", "dump_kv"]
    → {REPORT}/<incident_type>/report_*.json
    → {KV_ROOT}/<incident_type>/<req_id>/*.pt

Ref (clean):
  clean service + same {MODE}/{MODEL}/{TOPO}
    → prepare_ref_inputs from buggy report
    → force-feed prompt_token_ids + output_token_ids
    → dump_kv (manual_trigger / dedicated type)
    → {KV_ROOT}/<type>/<ref_req_id>/*.pt

Compare:
  locate_first_divergence / compare_kv_similarity / compare_per_layer
    --buggy-dir <bad> --ref-dir <ref> --report <report_*.json>
```

## Environment cheat sheet (from `{CONTAINER}`)

- Apply changes: `docker cp <host> {CONTAINER}:<path>` or `docker exec {CONTAINER} bash -c '...'`
- Anomaly reports: `{REPORT}/<incident_type>/report_*.json`（或 `{RG_CFG}` / `report_dir` 指向的根目录）
- Native KV dump: `{KV_ROOT}/<incident_type>/<req_id>/*.pt`
- **启动服务前先查显卡占用，被占用就等释放再跑**: `npu-smi info`。HBM 被别的进程占用时不要硬启。
- **注意磁盘空间**: `dump_kv` 按请求落盘，长上下文仍可很大。跑之前 `df -h {REPORT}`；及时清理误报 `kv_cache/.../<req_id>/` 和过期 report。

## Investigation flow

### Step 1: Triage — can runtime_guard even help?

Before any work, judge whether runtime_guard is the right tool for the bug class.

**runtime_guard 完全解决不了 (say so and stop)**:
- 卡死/死锁/hang — runtime_guard itself uses collective comms and will hang itself; not a diagnostic for hang bugs.
- 启动 crash / import error — runtime_guard only works after service starts.
- OOM / 显存泄漏 / ghost memory — runtime_guard doesn't monitor memory.
- 性能问题 (吞吐/QPS 下降) — use profiling / perf bench, not this skill.
- 模型权重错 / checkpoint 加载错 — runtime_guard doesn't read weights.
- 跨进程拓扑错 (TP/PP/DP misconfig) — runtime_guard assumes topology is correct; misconfig breaks runtime_guard itself.
- 网络层错 (Mooncake transport broken) — runtime_guard doesn't hook network layer.
- kernel 内部数值发散但未到 NaN/logits 异常且 KV 也正常 — detectors 不命中；若无 output 现象且不想抓 KV，本 skill 帮不上。

**runtime_guard 能解决 (proceed)**:
- 输出层异常: 复读 (token_repeat), 生僻字/乱码 (token_logprob), NaN logits (logits_finite), 输出含特定字符串 (output_substring)
- KV block 元数据错位/重写冲突 (block_kv)
- Position-id 错位 (position_alignment)
- Spec decode 接受率异常 (spec_acceptance)

**runtime_guard 有帮助但需配合 dump_kv + ref 对比 (proceed with caveat)**:
- PD 分离 KV 传输 bug — detector 命中给现场坐标 + `dump_kv`，再用 ref 对比完成最终定位
- 数值发散仅当发散到 logits/logprob 层才能命中 detector；定界仍靠 buggy vs ref `dump_kv`

If triage says runtime_guard can't help, tell the user directly. Don't burn cycles on the wrong tool.

#### Symptom → runtime_guard applicability decision tree

Walk the tree by observable symptom. Stop at first matching leaf.

```
Q1: 服务能起来吗?
├─ 否 (启动 crash / import error / 立即 OOM)
│   → NOT runtime_guard. 修启动先.
│
└─ 是 → Q2: 服务跑起来后现象是什么?

Q2: 服务跑起来后现象是什么?
├─ A. 跑一阵就 hang / 死锁 / 不响应
│   → NOT runtime_guard. runtime_guard 自己用 collective, 不能诊断 hang.
├─ B. 跑一阵就 OOM / 显存涨
│   → NOT runtime_guard. 内存问题, 用 npu-smi 监控 + 查泄漏 PID.
├─ C. 吞吐/QPS 降了, 但单请求 latency 正常
│   → NOT runtime_guard. 性能问题, 用 profiling / perf bench.
├─ D. 训练 loss spike / 训练发散
│   → NOT runtime_guard. 推理侧工具, 训练侧用其他工具.
├─ E. 偶尔输出不对 (语义错 / 乱码 / 复读 / NaN), 吞吐正常
│   → runtime_guard 适用. 进 Q3.
├─ F. 输出看着对, 但某层 KV / block 怀疑不对 (无 output 现象)
│   → detector 可能不命中. 跳过 Step 5 sweep, 直接 Step 4 sanity
│      → Step 5/6 用 manual_dump / manual_trigger 抓 dump_kv
│      → Step 7 ref 对比 (need dump_kv + ref compare).
└─ G. 拓扑/网络/权重错 (TP/PP/DP 配错, Mooncake broken, 权重文件坏)
    → NOT runtime_guard. 修配置/网络/权重先.

Q3: 输出异常具体是哪类? (可多选, 同时开多个 detector)
├─ E1. 输出 token 复读 / 卡在某个词循环
│   → token_repeat (最轻量, 首选; on_trigger: report + dump_kv)
├─ E2. 输出含生僻字 / 罕见 token
│   → token_logprob.ill_rare_window_thresh (需 worker top-k logprobs)
├─ E3. 输出乱码 / 无意义字符序列
│   → token_logprob.ill_garbled_window_thresh
├─ E4. 输出 NaN / Inf token / 包含 <unk> 等
│   → logits_finite + token_logprob.ill_nan_window_thresh
├─ E5. 输出含特定字符串 (prompt 泄露 / echo / 注入痕迹)
│   → output_substring (需预知 pattern)
├─ E6. Spec decode (MTP/Eagle) 接受率异常
│   → spec_acceptance (需 spec decode on)
├─ E7. 怀疑 KV block 写入错 (block 重用错 / 写入顺序乱 / writer 冲突)
│   → block_kv + dump_kv + ref compare
├─ E8. 怀疑 position-id 错位 (RoPE 相关, 1-D text path)
│   → position_alignment
└─ E9. 具体类未定 (PD 分离 / 多 detector 候选)
    → 全开 sweep (走 runtime-guard-detector-sweep), 让数据说话
```

### Step 2: Repro availability + quick smoke (1-2h)

Before any code/env work, ask the bug reporter (or recall from prior sessions):

- 复现概率？(约百分之几)
- 大约多久能复现？(N 次请求中平均多少次命中)
- 复现条件？(特定 prompt? 并发? 特定 `{MODE}`?)
- 启动脚本？(`{START}`)  请求脚本？(`{REPRO}`)
- 谁来跑：skill 用 `docker exec {CONTAINER}` 跑，还是用户自己起服务 + 发请求？

If repro path is provided, **先跑一下，但不要太久 (1-2h)**:
- Confirm service can launch (no startup crash). **启动前先 `npu-smi info` 查显卡是否被占用；被占用就等释放再启动**。
- Confirm problem is reproducible in current environment.
- Rough repro rate estimate.

- skill 跑: `docker exec {CONTAINER} bash -c '{START}'` 起服务 → 等 ready → `docker exec {CONTAINER} bash -c '{REPRO}'`.
- 用户自己跑: 请用户起服务 + 发请求，回报 ready 与否 + 是否复现。

If 1-2h 内未复现: 不要继续盲目跑, 检查 (a) 复现条件是否完整 (b) 环境是否变了 (c) 复现率是否真如报告. 跟 reporter 确认.

If service can't launch → not a runtime_guard problem, fix startup first.

### Step 3: Env setup — runtime_guard source

**Check runtime_guard / vllm-ascend source state**:
```
docker exec {CONTAINER} bash -c 'cd {SRC} && git log --oneline -5 && git status -s'
```
- Confirm runtime_guard is present at HEAD under `{SRC}/vllm_ascend/runtime_guard/`. If a "fault" image has local commits (`add runtime_guard`, `bug fix`, etc.), note them but don't assume they ARE the fault — most are instrumentation; the bug itself is usually a runtime/numerical divergence the runtime_guard detector must catch.
- If runtime_guard not present → pull/build a vllm-ascend tree that includes it (in-tree package).

**Verify pip-installed vllm-ascend matches source HEAD**:
```
docker exec {CONTAINER} bash -c 'pip show vllm-ascend | grep -E "Version|Editable"'
```
- If stale (pip version ≠ source HEAD), `pip install -e .` to realign. Note: editable installs point at `{SRC}` — check `Editable project location`.

No msprobe install / import checks. `{DUMP}` is always native `dump_kv`.

### Step 4: Dump sanity test — can dump_kv write?

Before stress-testing, verify the native dump pipeline works in `{MODE}`:

1. Configure minimal `{RG_CFG}` (see `runtime-guard-config-recommender`). Default often `{WS}/runtime/config/runtime_config.json`（也可经 `--additional-config` / `runtime_config_path` 指向）.
2. Start service with dump quota off: `dump.auto_max_times: 0` + `dump.manual_dump: false`. Detectors may stay on with `on_trigger: ["report"]` only.
3. Confirm runtime_guard loaded: grep worker log for runtime_guard / runtime_config markers.
4. 热更 `{RG_CFG}`: `dump.auto_max_times: 3` (或短时 `manual_dump: 1`); target detector(s) `on_trigger: ["report", "dump_kv"]` with `dump_kv: { "scope": "request", "dump_all_blocks": false }`.
5. **等 ≥ `reload_interval_seconds`**（或 `additional_config.runtime_config_reload_interval`, 建议 ≥5s）, grep worker log 确认拾取; 未拾取前不要 curl.
6. Trigger one request → detector 命中或 manual_dump / manual_trigger → confirm:
   - `{REPORT}/<type>/report_*.json`
   - `{KV_ROOT}/<type>/<req_id>/*.pt`
7. **Measure per-request dump size**: `du -sh {KV_ROOT}/<type>/<req_id>/`. Long prompts cost more.
8. **Check disk free**: `df -h {REPORT}/` → 记 `disk_free`. 及时清理误报 `kv_cache` 目录。
9. **Compute dump_kv budget**: `auto_max_times ≤ (disk_free × 0.7) / per_req_kv_size`（必要时再除 rank 数若多进程各写一份）. 与 `runtime-guard-config-recommender` Step 3 复现率公式取**较小者**.
10. **磁盘紧时降级** (按顺序): `scope: request` + `dump_all_blocks: false`（默认）; 减小 `auto_max_times`; 降低 `N_expected_repros`.

If dump doesn't produce files — diagnose by symptom:

| 现象 | 可能原因 | 诊断命令 | 处置 |
|---|---|---|---|
| **A. 启动报错** | 旧 schema / auto+manual 互斥 / JSON 语法错 | `grep -E "runtime_guard\|runtime_config\|ValueError\|JSONDecodeError" worker.log` | 改 `{RG_CFG}` 重启 |
| **B. 启动 OK 但 runtime_guard 未加载** | config path 未挂 / additional-config 未传 | `grep -E "runtime_guard\|runtime_config" worker.log` | 修启动参数 / `{RG_CFG}` 路径 |
| **C. detector 命中但无 `.pt`** | `dump_kv` 不在 `on_trigger` / quota=0 / cooldown | report 里 `dump_armed`; `grep dump_kv worker.log` | 加 `on_trigger` + `auto_max_times>0` |
| **D. detector 没命中** | 阈值错 / bug 类不对 | `ls {REPORT}/*/report_*.json` 空 | 调阈值或 sweep；或改用 manual_dump |
| **E. report 有但 kv 目录空** | quota/cooldown 挡住 / 非 leader rank / block_ids 空 | `verify_request_kv`; quota log | 修配额 / 确认 leader 写盘 |
| **F. 部分 rank 有 dump 部分无** | 各 rank `{RG_CFG}` 不同步 | 跨 rank 比对 `{KV_ROOT}` | 各 rank 读同一份 `{RG_CFG}` |
| **G. dump 时盘满/IO 错** | 磁盘不足 / 权限 | `df -h`; `ls -la {REPORT}` | 清盘 / 改权限 |
| **H. `.pt` 存在但 torch.load 失败** | 半写 / 截断 | `ls -la <file>`; `inspect_kv_dump` | 删 truncated 重抓 |

通用诊断顺序: A→B→D→C→E→F→G→H.

**热更被拒时保留旧配置**（JSON/JSONC 解析失败、未知 detector key、数值类型错）— 服务不崩、继续用旧配置跑.
所以"改了配置行为没变"≠ 没生效就是阈值问题: 先 grep worker log 的 `runtime_config` reload 拒绝信息再下结论.

### Step 5: Stress test, reproduce + dump_kv successfully

Once dump_kv pipeline verified, this is the **single "reproduce + dump" phase**.

**关键**: detector 阈值触发是**滞后**的 (token_repeat 要重复 token 累计超阈值才命中). detector 命中的 dump 是 post-bug 状态 — 这是预期. first-wrong 估算归到 Step 7（从 report `output_token_ids` + `locate_first_divergence`）.

执行:

1. 热更 `{RG_CFG}` 到压测态 (`runtime-guard-config-recommender`):
   - sweep / 目标 detector ON (`runtime-guard-detector-sweep` 或单类)
   - `stop_after_alert: false`
   - `dump.auto_max_times: <budget>`
   - `dump.manual_dump: false`（靠 detector auto-arm）
   - `on_trigger: ["report", "dump_kv"]`
2. **等 ≥ `reload_interval_seconds`**, grep 确认拾取; 未拾取前不要 curl.
3. Launch stress: `docker exec {CONTAINER} bash -c '{REPRO}'` (scale N up if no hits).
4. **Periodic check**:
   ```bash
   python -m vllm_ascend.runtime_guard.analysis.scripts.summarize_reports \
     --report-dir {REPORT} --limit 20
   ls -t {REPORT}/*/report_*.json | head -5
   ls {KV_ROOT}/*/*/ | head
   ```
5. **Handle false positives — 先信 detector, 再用 output 反向验证**:
   - detector 命中时先信 detector, 再用 output + prompt 反向验证. **不能因 output 看着对就推翻 detector.**
   - 真命中保留 report + `kv_cache/.../<req_id>/`; 误报 `rm -rf` 对应误报目录释放盘.
6. **Tune `auto_max_times` / cooldown** if false positive rate high.
7. **Iterate** until true-positive dump captured.

Exit criteria: at least one `report_*.json` with non-empty `{KV_ROOT}/<type>/<req_id>/*.pt`, AND report's `output_token_ids` shows the buggy sequence (`report.save_sensitive_info: true`).

抓到现场后用 `runtime-guard-analysis` 做 summarize / correlate / verify / inspect（不改 config、不重跑服务）.

### Step 5.5: 对账 dump_kv ↔ report（先验基本一致性，再抓 ref）

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.verify_request_kv \
  --report {REPORT}/<incident_type>/report_....json \
  --report-dir {REPORT}
```

检查要点:

- **block 容量**: block_ids 装得下 N 个 token（N = len(prompt_token_ids)+len(output_token_ids)）。
- **KV `.pt` 可读**: tensor 有限、层文件存在；layout 为 `{KV_ROOT}/<incident_type>/<req_id>/*.pt`。
- **文件数**: 与 `min-files` / 层数大致对齐。

对不上 = 抓错 req / 配额或路径问题, 先修再继续。

**两点注意**:
- report 的 token_count 与 token_ids 长度可能差几个 — 只作信息展示，不判 FAIL。
- N 不能从 KV buffer「非零个数」推 — 空 slot 是脏数据；N 从 report token_ids 拿。

关联同一 `req_id`:

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.correlate_incident \
  --report-dir {REPORT} --req-id <req_id>
```

### Step 6: Capture reference dump_kv (Phase E begins)

Use `runtime-guard-ref-kv-dump` — full methodology there. Key points:

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.prepare_ref_inputs \
  --report {REPORT}/<type>/report_....json --out ref_inputs.json
```

- Reference must match buggy `{TOPO}` / `{MODE}` / `{MODEL}`.
- Clean service; arm `dump_kv` on `manual_trigger` (or dedicated type).
- Force-feed `force_feed_token_ids` from `ref_inputs.json` (token-id API, not string re-tokenize).
- Optional 3-pass (prefill / decode / full) — each via `dump_kv`.
- Collect ref dir: `{KV_ROOT}/<type>/<ref_req_id>/`.

### Step 7: Compare, find first divergent KV (Phase E.1)

**对比输入**: buggy `{KV_ROOT}/.../<bad_req>/` + ref dir + buggy report（推 first-wrong / 对齐 token）.

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.locate_first_divergence \
  --buggy-dir {KV_ROOT}/<type>/<bad_req>/ \
  --ref-dir   {KV_ROOT}/<type>/<ref_req>/ \
  --report    {REPORT}/<type>/report_....json \
  --block-size 128 --cos-thresh 0.99

python -m vllm_ascend.runtime_guard.analysis.scripts.compare_per_layer \
  --buggy-dir {KV_ROOT}/<type>/<bad_req>/ \
  --ref-dir   {KV_ROOT}/<type>/<ref_req>/ \
  --report    {REPORT}/<type>/report_....json
```

- **Table 1**: per-token min-cos → first bad token.
- **Table 2**: that token’s per-layer cos / maxdiff → first divergent layer.
- **容差**: 可先 ref-vs-ref（同输入跑两遍）估 noise floor；超出 baseline 才是真发散.

**Decision** (取决于 `{TOPO}`):
- PD-separated: D 端收到的 KV ≠ ref prefill → bug in P prefill 或 P→D transfer；D decode 层 KV ≠ ref decode → bug in D decode write.
- single-process: 某层 KV ≠ ref → bug in that layer’s KV write / attention path.
- KV nearly identical → Phase E.2.

### Step 8: Phase E.2 — KV 一致时的下一步

If `locate_first_divergence` / `compare_per_layer` show KV nearly identical:

1. Bug is likely **after** KV write: logits / logprob / sampling / output assembly.
2. Re-read detector hits (`logits_finite` / `token_logprob` / `token_repeat`) and report `output_token_ids` for the first wrong token.
3. Stay on native tools — do **not** open msprobe. If the case needs op-level hidden-state dumps outside `dump_kv`, tell the user that is **out of scope** for this skill and stop or hand off explicitly.

## dump_kv cheat sheet

- **Trigger**: detector `on_trigger` includes `"dump_kv"`, or `manual_dump` / `manual_trigger` with quota.
- **Quota**: `dump.auto_max_times` + `dump.auto_cooldown_seconds`. `auto_max_times>0` 与 `manual_dump` active **互斥**.
- **Schema**: 无 `dump.enabled` — dump 活跃派生自 `auto_max_times>0 || manual_dump active`.
- **Per-detector opts**: `"dump_kv": { "scope": "request", "dump_all_blocks": false }`（默认只切本请求 block）.
- **Layout**: `{REPORT}/kv_cache/<incident_type>/<req_id>/*.pt` — dict with `req_id`, `block_ids`（实际落盘的，越界 id 已剔除）, `layer`, `tp_rank`, `num_kv_heads`, `dump_all_blocks`, `source`, `tensor`, ...
- **Layer 顺序**: 分析脚本对层名 natural sort（`layer_2` < `layer_10`），first-divergence 层号可信.
- **Reports need ids**: `report.save_sensitive_info: true`（Step 6/7 依赖 `prompt_token_ids` / `output_token_ids`）.
- **Cross-rank**: 同 EngineCore 内各 TP 必须读同一份 `{RG_CFG}`.

## Related skills (in-tree only)

- `runtime-guard-detector-sweep` — Step 5 sweep workflow.
- `runtime-guard-config-recommender` — thresholds / `auto_max_times` / dump_kv budget.
- `runtime-guard-analysis` — summarize / correlate / verify_request_kv / inspect.
- `runtime-guard-ref-kv-dump` — prepare_ref_inputs / request_from_report / compare_kv_similarity / locate_first_divergence / compare_per_layer.
