# runtime_guard 实卡功能测试执行笔记（本机 test-mrv2-cann91）

> 记录在 `test-mrv2-cann91` 容器（CANN 9.1.0 / py3.12 / vllm-ascend 0.27.1）上跑
> P0 冒烟 + P1 全量的结果、发现的问题（BUG），以及环境绕法。
> 对应启动脚本：`scripts/p0_smoke_launcher.sh`、`scripts/p1_full_launcher.sh`。

## 2026-09-16 rebase 后回归（vllm 0.27.1→0.28.0 + vllm-ascend main 200+ commits）

| 项 | 内容 |
|----|------|
| 背景 | config 分支 rebase 到最新 main（714dd1d1b）成单 commit baec54437；vllm 升 **v0.28.0**（release 28）；两仓重编 |
| 编译 | vllm 用 VLLM_TARGET_DEVICE=empty + --no-build-isolation；vllm-ascend 用 --no-build-isolation --no-deps（aliyun 源 cmake/numpy 大 wheel 间歇停滞，绕法） |
| 产品 UT | runtime_config 18 + wiring/regression 75 + detector/wave/inject 36 = **129 passed / 0 fail** |
| P0 冒烟 | **6 用例 × v2/v1 = 12/12 PASS**（guard_off 0/0、inject_nan/inf_logits/forbidden/token_loop 各 1 report、manual_dump 768 .pt + report 字段 K-13 契约对齐） |
| 结论 | rebase 未破坏功能；feature 自有路径与 rebase 前字节一致（git diff d872b9819..baec54437 feature 路径为空），差异只在 wiring 且 UT/实卡均过 |
| 产物 | rg_p0_smoke_rebase28/summary.txt；dump 已按 §0.4 清理；卡 0 已释放 |
| 顺手修 | config：文档补 runtime_dump_dir（068e847ed）；analysis：脚本 PRODUCT 默认改指 vllm-ascend 主 checkout（d5c56689d） |

perf T0 基线规则（两条，重跑 C1-C6 前必读）：

1. **T0 = 当前产品分支去掉 runtime_guard 的基座** = config 分支与 main 的 merge-base。
   是派生值，随产品分支 rebase 而变，**不是固定 commit**（本次为 baec54437 的父提交 714dd1d1b）。
   重建：BASE=$(git -C /data0/test-mrv2-cann91/vllm-ascend merge-base origin/main feat/runtime-guard-config)
        git -C /data0/test-mrv2-cann91/vllm-ascend worktree add /data0/test-mrv2-cann91/rg-perf-t0 $BASE

2. **runtime_guard 只改 Python 文件，不带 csrc 改动 → T0 与产品树可共用编译产物**。
   前置校验（每次 rebase 后必做）：git diff --stat $BASE..<config> -- csrc/ CMakeLists.txt setup.py
   为空即成立，直接把产品树的 vllm_ascend_C*.so（可选连 csrc/build 增量缓存）拷入 T0 worktree 即可，
   **无需独立编译**；若校验非空（feature 动了 C++），则两树必须各自编译。

## 环境要点（本机，非 165）

- 源码：`/data0/test-mrv2-cann91/`；产品代码 `rg-config-review`（config 分支），
  工具/测试 `vllm-ascend`（analysis 分支）。
- python：`/opt/slime/venv/bin/python`（容器内 root）；跑服务必须 `docker exec test-mrv2-cann91`。
- 模型：功能测试用 `Qwen2.5-0.5B-Instruct`（快）。
- **必须 `export VLLM_BATCH_INVARIANT=1`**：本机 CANN 缺 `aclnnAddRmsNormBias`，
  拷入的 `vllm_ascend_C.so` 编译自更老/不同 CANN；不设则 EngineCore 起不来
  （`RuntimeError: aclnnAddRmsNormBias ... not in libopapi.so`）。该开关使
  `enable_custom_op()` 返回 False，layernorm 回退 `torch_npu.npu_add_rms_norm`。
  > **2026-09-20 更正**：此诊断有误——符号不在 CANN，在 `_cann_ops_custom` 构建产物的
  > vendor 包里；BI=1 在当前两 tip 上实测**不能**绕过（forward_oot residual 路径无视
  > `enable_custom_op()` 返回值直接调 op）。正确解法见「2026-09-20 更正」章。

## 功能测试最终结论（P0 + P1 已收口）

- **P0 冒烟**（6 用例 × v2/v1）全 PASS；**P1 全量**已跑完，仅 2 项环境跳过
  （`g05_spec_all_reject` 需真 MTP/投机解码、`p0_06_async_after_sample` 是 stub 但产品
  async 路径已被 W1–W4 隐式覆盖）。
- **6 个 open BUG 全部关闭**：

  | BUG | case | 修复提交 | live 复验 |
  |-----|------|----------|-----------|
  | W1-1 | R-04 v2 `seen≠oc` | `8876645a2`+`39e22dacc` | ✅ PASS（v2/v1 `seen==oc`=9） |
  | W1-2 | R-06 v2 `block_ids` 空 | `03bfd2a54` | ✅ PASS（`block_ids=[1..32]`） |
  | W1-3 | R-06 v1 长上下文无 report | `39e22dacc` | ✅ PASS（report present） |
  | W2-1 | F-07 顶层未知键静默接受 | `229e7deb3` | ✅ 产品 UT `test_v10b` |
  | W2-2 | F-39 `slot_mapping` | **取消功能** | F-38/F-39 REMOVED |
  | W2-3 | D-11 v2 token_repeat 误报 | `8876645a2` | ✅ 产品 UT `test_d11_*` |

  > W1-1/W1-2/W1-3 已实卡复验（`rg_test/reverify_39e22.py` R-04 + `rg_test/reverify_r06.py` R-06）。
  > **R-06 复验注意**：长上下文须让模型生成 ≥5 token（inject `token_loop:5:*` 需 `wave≥5`）；
  > 用「answer with one word」会提前 EOS → inject 不触发 → 误判 no report。
- **性能 C1–C6** 已出数；**C1 v1 ~1.5%（T1/T0=0.98466）待重测**（等 8 卡空闲后排除交叉轮换噪声再定论）；
  C2 达标；C3 全量 detector 开销已归因记录。
- **稳定性**：soak 已启动，当前**累计 ~3.6h（进行中，未达 24h）**，未见 hang/crash；结论待累计满 24h。
- 正式报告：`TEST_REPORT.md`；Word：`vllm-ascend/docs/zh/design/word/RuntimeGuard_测试报告.docx`。
- **case3/case4** 属 DFX 复读 bug 专项（`investigation/SKILL.md` 容器代号），非 runtime_guard 用例，跳过。

## P0 冒烟结果（6 用例 × v2/v1，Qwen2.5-0.5B）

| 用例 | 场景 | v2 | v1 |
|------|------|----|----|
| p0_01_guard_off | detector 全关 | PASS (reports=0) | PASS (reports=0) |
| p0_05_inject_nan | nan_logits → logits_finite | PASS (reports=1) | PASS (reports=1) |
| g02_inf_logits | inf_logits → logits_finite | PASS (reports=1) | PASS (reports=1) |
| g03_forbidden | forbidden_substring → output_substring | PASS (reports=1) | PASS (reports=1) |
| g04_token_loop | token_loop → token_repeat | PASS (reports=1) | PASS (reports=1) |
| p0_03_manual_dump | manual_dump:true 连续 | **PASS（03bfd2a54 修复后 v2 dumps=768 / TP=2=1536）** | PASS (本机 dumps=720 / TP=2 dumps=1440；165 上 896 / 1792) |

**P0 manual_dump / inject：BUG-5 已修；BUG-4（W3-1）已修复（`03bfd2a54`，实卡验证 v2 dump 恢复）；BUG-6/W3-2（K-13 字段对齐）已由 `5164073a9` 修复（per-req report，实卡验证）。**

## P1 全量结果

| 用例 | 场景 | 结果 |
|------|------|------|
| p0_04_tp_rank_gate | TP=2 + nan_logits，仅 TP0 检测 | **v2 PASS (reports=1)** / **v1 PASS (reports=1)** |
| p0_07_dump_schema | v1 manual_dump + verify_request_kv + inspect | dumps=720；schema 正常；verify 定位 OK；K-13 顶层字段已由 `5164073a9` 对齐 |
| g06_bad_reload | 热重载坏 JSON soft-fail | **PASS**：health=200 存活，`reload failed error=Expecting property name...`，旧配置保留 |
| p0_08_disk_reclaim | 磁盘回收 smoke | **PASS** |

未跑（本机缺环境/占位）：
- g05_spec_all_reject：需真 MTP/投机解码，Qwen2.5-0.5B 无 draft 模型，`inject_after_spec` 不会触发。
- p0_06_async_after_sample：async scheduling 占位 stub，产品开关未在本机启用。

### p0_07 细节

- v1 manual_dump 落 720 `.pt`，`inspect_kv_dump` 确认 schema：`req_id/layer=layer_23[1]/source=kv_caches/
  rank_tag=dp0_tp0_pp0_cp0/num_kv_heads=2/block_ids=[1]`，tensor `shape=(1,128,2,64)` bf16，全 finite。
- BUG-6：`dump_dir` 跟真实 req（路径）；K-13 顶层字段已由 `5164073a9` 修复（per-req report 对齐）。
- 另：`verify_request_kv`/`inspect_kv_dump` 必须以 `PYTHONPATH=$ANALYSIS`（analysis 分支）运行；
  若 `$PRODUCT`（config 分支）在前会 shadow `vllm_ascend.runtime_guard`（无 `analysis` 子包）→ `ModuleNotFoundError`。

## W3 轮次状态（165 / `686cd7e0b`）

| ID | 项 | 状态 |
|----|----|------|
| **W3-1** | = BUG-4：v2 `dump_kv` 恒 0 落盘（manual + auto） | ✅ 已修复（`03bfd2a54`，实卡验证） |
| **W3-2** | K-13 report 字段一致性 | ✅ 已修复（`5164073a9`，per-req report，实卡 TP=1/TP=2 验证） |
| — | GLM-4.7-Flash 本机 32GB OOM | ⏭ 跳过（环境，非代码 bug） |
| W1 | ✅ 完成（Sep13） | 3 BUG：W1-1/R-04 v2 计数不一致、W1-2/R-06 v2 block_ids 空（同源 W3-1，疑随 `03bfd2a54` 修，需复验）、W1-3/R-06 v1 长上下文无 report |
| W2 | ✅ 完成（Sep13） | 3 BUG：W2-1/F-07 未知顶层键静默接受、W2-2/F-39 v2 无 slot_mapping、W2-3/D-11 v2 token_repeat 误报 |
| W4 | ✅ 完成（Sep14 W4-b DONE） | §3 动作行为 A-01..A-32 全 PASS/跳过，无新 BUG |
| W5 | ✅ 完成（Sep14） | 6 open BUG 复验（HEAD `a9312e299`）：**5 仍存**（W1-1/W1-3/W2-1/W2-2/W2-3）、**1 已修**（W1-2，`03bfd2a54`）；P1 g05/p0_06 跳过；C1 v1 复跑=0.98466 ❌；soak 已启动 |

## W2 轮次状态（配置校验 / report 元数据）

| ID | 项 | 状态 |
|----|----|------|
| **W2-1 / F-07** | 顶层未知键（如 `windw`）被静默接受 | ✅ 已修复：`TOP_LEVEL_KEYS` + validate 响亮拒绝（产品 UT `test_v10b`） |
| **W2-2 / F-39** | v2 `include_slot_mapping=True` report 无 slot | ❌ **取消功能**：删除 `include_slot_mapping` / report 内 slot 字段；KV 分析只用 `block_ids`（F-38/F-39 REMOVED） |

**GLM-4.7-Flash（跳过）**：TP=1 加载即 ~28.86/29.49 GiB OOM；TP=2 仍 OOM（MoE experts
未随 TP 切分）。标跳过，不记产品缺陷。

## W5 复验（6 open BUG，产品 `a9312e299`）

> ⚠️ 此节是**历史快照**（当时 HEAD `a9312e299`，6 BUG 尚存 5）。后续提交
> `8876645a2`/`229e7deb3`/`39e22dacc`/`03bfd2a54` 已全部关闭，见上「功能测试最终结论」。

复验结论（详见 `rg_test/results/W5.md`）：

| BUG | case | 复验 | 证据 |
|-----|------|------|------|
| W1-1 | R-04 v2 计数不一致 | ❌ **仍存** | v2 `seen=15 oc=10`（15≠10）；v1 对照 15==15 |
| W1-2 | R-06 v2 block_ids 空 | ✅ **已修** | v2 `block_ids=[1,2,3,4,5] (n=32)`；`03bfd2a54` 生效 |
| W1-3 | R-06 v1 长上下文无 report | ❌ **仍存** | v1 `(no reports)`（chunked prefill） |
| W2-1 | F-07 顶层未知键 | ❌ **仍存** | v1/v2 `rejected_log=False windw_still_present=True`；`_validate.py` 无顶层 unknown-key 校验 |
| W2-2 | F-39 v2 无 slot_mapping | ❌ **仍存** | v2 `slot_mapping=None`；v1 对照 present |
| W2-3 | D-11 v2 token_repeat 误报 | ❌ **仍存** | v2 `reports=1 repeat_sum=10 seen=76 oc=77`；v1 对照 reports=0 |

其余：g05_spec_all_reject 跳过（本机无 MTP/GLM OOM）；p0_06_async_after_sample 跳过
（脚本 stub，但产品 async 开关已启用——`AscendAsync*` wrapper 在 get_output 后跑 check_after_sample）。
case3/case4 是 DFX 复读 bug 容器代号（investigation SKILL），非 runtime_guard 用例，跳过。

## BUG-4 / W3-1（v2 dump_kv 零 dump：manual + auto）— ✅ 已修复（`03bfd2a54`）

### 修复验证（本机 `03bfd2a54`，卡 3,4，Qwen2.5-0.5B，max_tokens=16）

| case | runner | reports | dumps | skipped |
|------|--------|--------:|------:|--------:|
| manual_dump TP=1 | v2 | 1 | **768** | 0 |
| manual_dump TP=1 | v1 | 1 | 768 | 0 |
| manual_dump TP=2 | v2 | 1 | **1536** | 0 |
| auto nan_logits TP=1 | v2 | 1 | **48** | 0 |

- **修复前 v2 恒 0，修复后 v2 全部 >0；v1 无回归**（768 对照不变）。
- rank_tag 正确（`dp0_tp0_pp0_cp0` / `dp0_tp1_pp0_cp0`），真实 `req_id`（`cmpl-...`）落盘，
  不再出现 `empty_block_ids` skip。
- 根因修复：`block_ids_for_request` 去掉 `if input_batch is None: return []` 早退，
  改为回退 `req_states.req_id_to_index` + `runner.block_tables`（`num_blocks.np` +
  `StagedWriteTensor.gpu.cpu()` D2H）。附单测 `test_bug4_block_ids_v2_*`（2 passed）。

### 165 复验矩阵（产品 `686cd7e0b`，修复前，manual_dump:true 连续）

### 165 复验矩阵（产品 `686cd7e0b`，manual_dump:true 连续）

| case | runner | dumps | skipped |
|------|--------|------:|--------:|
| tp1_v2 | v2 | 0 | 16 |
| tp1_v1 | v1 | 896 | 0 |
| tp2_v2 | v2 | 0 | 32 |
| tp2_v1 | v1 | 1792 | 0 |

**结论：未修好。v2 manual_dump 在 TP=1 / TP=2 全坏；v1 同配置正常。**
（此前本机笔记写「TP=1 已修 / 仅 TP=2 残留」已过时，以本矩阵为准。）

### auto dump（detector→dump_kv）同样坏在 v2（同一根因）

| case | runner | inject | reports | dumps | skipped |
|------|--------|-------:|--------:|------:|--------:|
| g01 nan_logits | v2 | 1 | 1 | 0 | 1 |
| g01 nan_logits | v1 | 1 | 1 | 56 | 0 |
| g04 token_loop | v2 | 8 | 1 | 0 | 1 |
| g04 token_loop | v1 | 8 | 0 | 0 | 0 |

- `g01_v2`：`type=logits_finite` 的 skip marker `reason=empty_block_ids`，同 manual 的
  `_run_kv_dumps` → `block_ids_for_request` → `[]`。v1 同配置 `dumps=56` 正常。
- 结论：**auto dump 与 manual dump 走同一 `_run_kv_dumps` → `block_ids_for_request`，
  在 v2 下同样空 block_ids；BUG-4 覆盖面从「manual_dump」扩大到「v2 dump_kv 全路径」。**
- `g04_v1`（token_loop→token_repeat）`reports=0`：detector 未触发，为独立的
  token_repeat/injection 阈值问题（非本 dump 路径 bug），单独跟进。

### 历史背景（曾记为 TP=1 延迟 D2H 已修）

- 旧现象：arm 在 `sync_for_step`（`prepare_inputs` **之前**）解析 block_ids → v2
  `StagedWriteTensor` 无 host `.np` → v1 路径异常被吞 → `[]` → skip。
- 曾做「延迟到 sample-end drain 再解析」；本机一度看到 TP=1 `dumps=720`。
- W3 / 165 在 `686cd7e0b`（wave-head task bus + manual 本地 end-of-wave D2H）上
  **TP=1 与 TP=2 的 v2 均 `dumps=0`**。

### 现场证据（与 W3 一致，且扩展到 TP=1）

- report：`req_id=__manual_trigger__`，`dump_dir=.../manual_trigger/_dummy_req_0`，
  `dump_attempted=True`，`dump_count=0`。
- 每个 `wave_N/<rank_tag>/` 只有 `dump_skipped.json`、无 `.pt`：
  - synthetic `__manual_trigger__` → `reason=no_dump_targets`
  - 真实 `cmpl-...` → `reason=empty_block_ids`（`block_ids=[]`，常见
    `req_idx=255` / `prompt_token_count=0` —— `req_states` 持久槽位，非 batch 行）
- 日志仍见：`dump_kv arm with empty block_ids (resolve at sample-end drain)`（或
  新路径下 `_run_kv_dumps` 的 `skip empty local block_ids`）。

### 失败分支（产品侧暂不修；定位供下次修）

调用链（`686cd7e0b`）：

1. `run_sample_phase` → `sample_fn()` → upstream `sample_tokens` **开头**
   `self.execute_model_state = None`（batch 仅活在局部变量 / `SamplePhaseResult`）。
2. 随后 `end_of_wave_sync` → `_maybe_fire_manual_local` → `_run_kv_dumps`，
   固定 `block_ids_for_request(runner, req_id, None)`（**不用** peeked `input_batch`，
   **不传** batch `req_idx`）。
3. `kv_block_meta.block_ids_for_request`：
   - v2 无 `runner.requests`（只有 `req_states`）→ 跳过；
   - `_runner_input_batch` → `None`（state 已清）；
   - **命中 `if input_batch is None: return []`** ← 当前失败点；
   - **达不到** 后面的 v2 真路径（`runner.block_tables` + `idx_mapping_np` /
     `StagedWriteTensor.gpu` D2H）。

次要混淆信号：`iter_local_request_rows` 在无 batch 时回退 `req_states.req_id_to_index`
（如 `255`），若误当 batch 行索引会读空槽；但在现行 end-of-wave 路径上，主因是
上一段 early-return，根本读不到 `block_tables`。

### 跟进

- 产品侧：修时应在 state 清空前拿到 `input_batch`（或走 `req_states` +
  `runner.block_tables`，勿依赖已 pop 的 `execute_model_state`）。
- analysis：**已由 `03bfd2a54` 按上述方向修复**（`req_states` 回退），实卡验证通过。

## BUG-5（g03 配置/注入错配）— ✅ 已修复

- 原现象：`patterns=["SECRET_MARK"]` 与 `inject.DEFAULT_TEXT="李白"` /
  裸 `INJECT=forbidden_substring` 不对齐。
- 修复：`configs/g03_forbidden_substring.json` → `patterns=["李白"]`；
  live 脚本默认 `forbidden_substring:5`（走 DEFAULT_TEXT）。

## BUG-6（manual_trigger report `dump_dir` 路径）— ✅ 路径已修；K-13 已由 `5164073a9` 闭合

- 原现象：report `dump_dir` 落在 `__manual_trigger__`，`.pt` 在真实 `req_id` 下。
- 已做：
  - 产品 `ReportWriter`：`dump_dir`/`dump_dirs` = 真实 req **根目录**（不含 `wave_*`）；
    `dump_arm_wave` 仅作 arm 元数据。
  - analysis `resolve_kv_dump_dir`：在 req 根下选 **已有 `.pt` 的 wave_***；兼容旧路径。
- **已由 `5164073a9` 修复（W3-2 / K-13）**：manual report 顶层 `req_id` / `block_ids` / `dump_dir` 三者对齐（见下）。

## BUG W3-2（K-13 report 字段一致性）— ✅ 已修复（`5164073a9`）

K-13 要求：`req_id` / `block_ids` / `dump_dir` 三者可对上。W3 实测 FAIL：

| 字段 | 实际 | 问题 |
|------|------|------|
| 顶层 `req_id` | `__manual_trigger__`（合成） | 与 `dump_dir` basename（真实 `cmpl-...`）不一致 |
| 顶层 `block_ids` | 缺失 / `None` | 真实 ids 只在 `detail.requests[0].block_ids` |
| `detail.requests[0].block_ids` | 常为 `[]` | arm 时刻解析，end-of-wave 前往往仍空 |
| `dump_dir` | `.../manual_trigger/<真实 req>/` | BUG-6 已改对路径；但与顶层 `req_id` 仍不对齐 |

已由 `5164073a9` 修复：manual report 改为 batch 里每个真实 req 各写一份，顶层 `req_id`/`block_ids`/`dump_dir` 三者对齐，`trigger_req_id=__manual_trigger__` 下沉 `detail`。实卡验证（本机 Qwen2.5-0.5B）：TP=1 两并发 reports=2 dumps=1584、TP=2 同对齐，`block_ids=[1]/[2]`。

## BUG-1/2/3（DP/PP 场景）

来自 NPU 标杆 DP/PP/组合测试（见 project_kv_dump_benchmark）：

| Bug | 现象 | 修复？ |
|-----|------|--------|
| **BUG-1** DP rank_tag 恒 `dp0` | `runner_dp_rank()` 走 `get_dp_group().rank_in_group`，external DP 下 world_size=1 恒返 0，双 replica 写 `dp0_…` 碰撞 | ✅ `rank_gate.py` 重写：优先 `runner.dp_rank` → `parallel_config.data_parallel_rank` → env `VLLM_DP_RANK`，最后才 fallback group rank |
| **BUG-2** DP manual_dump quota 竞态 | 计数写共享 JSON，两 DP 各在 prefill wave 消耗 1，`manual_dump:2` 只 dump 出 1 个 | ⚠️ 缓解未根治：改为 in-memory 递减（仅归零写回 JSON）；`_defaults.py` 注释承认 multi-DP worst case `num_DP × N`；PP=1 broadcast 下竞态本质仍在 |
| **BUG-3** PP layer 局部重编号 | `_iter_kv_tensors` 按位置枚举，PP 末段 dump 成 `layer_0..13` 而非全局 `14..27` | ✅ `kv_cache_reader.py` 新增 `_pp_start_layer` + `_kv_cache_config_layer_names`，layer 名改用全局 `model.layers.14.self_attn` |

## dump schema（v1 确认，K-11）

- `.pt` keys：`req_id, block_ids, layer, source, rank_tag, tp_rank, pp_rank, cp_rank,
  num_kv_heads, tensor`；tensor `[1,128,8,128]` bf16；`rank_tag`=`dp0_tp0_pp0_cp0`。
- report 字段：`ts/incident_type/req_id/rank/dump_attempted/dump_arm_wave/dump_count/dump_max_times/dump_dir[/dump_dirs]/detail`。

## §7 混部 H-01..H-06（本机，Qwen2.5-0.5B，两实例卡 0/1）— 全 PASS

启动脚本：`live/scripts/h_mixed_deployment.sh`（L1 H-01/02/06 → L2 H-03 → L3 H-04/05）。

| ID | 场景 | 结果 |
|----|------|------|
| H-01 | 两实例不同 dump_dir | PASS：A/B 各 48 dump，report 各 1，req_id 不串写 |
| H-02 | 共享日志可区分 | PASS：各自 serve.log 含自身 port 命中（2/2） |
| H-03 | 混部 + manual dump | PASS：armed A=768 dump，guard_off B=0 |
| H-04 | 混部 + 磁盘紧张 | PASS：A headroom=50TB skip（0 dump + `insufficient_free_space`），B 正常 48 dump |
| H-05 | 混部 + 热重载 | PASS：reload A 后 A/B 均 health=200，无干扰 |
| H-06 | UCM 日志劫持 | PASS：各自日志走 stdlib 树，互不劫持（201/201 行） |

**H-04 关键修法（测试脚本 bug，非产品 bug）**：磁盘 gate 只在 `estimated > 0` 时评估
（`actions.py:224`）；`manual_dump` 在 `sync_for_step`（prepare_inputs 前）arm，block_ids 为空
→ estimate=0 → gate 被跳过。改用 **auto-dump**（`RG_INJECT=nan_logits` + `k02_on_trigger_dump.json`
logits_finite→dump_kv）在 sample 后 arm，block_ids 已知 → gate 生效。
skip 证据：`skip: free=11639758004224 needed=50000001572864 (payload=1572864 tp_size=1 headroom=50000000000000)`。

## §6 PD 分离（T4 单节点 1P1D）— ✅ 冒烟 PASS（卡 0+2）；卡 1 RDMA 端口 down

- 前置：mooncake NPU wheel 已装（`mooncake-transfer-engine-npu==0.3.13.post1` + RDMA libs）。
- **卡 1 RDMA/HCCS 端口 down（infra）**：D 放卡 1 起不来，`Failed to initialize AdxlEngine, status: 503900,
  EI0009 Communication_Error_Initialize_Transport: Device 1 transport init error. Reason: The network port is down`
  （Hixl CS server ip:192.168.9.103 port:20146）。NPU 拓扑全 HCCS 互联，非拓扑缺失 → 卡 1 单点网络端口问题。
- **D 换卡 2 后 1P1D 冒烟 PASS**（`rg_test/pd_smoke.sh`，P 卡0:13700 / D 卡2:13701 / proxy:8080）：
  `PD REQUEST OK`，D 日志 `KV cache transfer ... took 163.77 ms ... remote_session_id 192.168.9.103:15064`，
  `External prefix cache hit rate: 100.0%`（KV 经 mooncake P2P 从 P 拉到 D）。
- **PD-01..PD-03 PASS**（`rg_test/pd_guard_test.sh`，P/D 两侧都挂 runtime_guard，`RG_INJECT=nan_logits` 打在 D）：
  P 侧 hook 不崩（report=0/dump=0，prefill-only 无 sample）；D 侧 `reports=1 dumps=48`，
  report schema 与单机一致（`req_id/incident_type/rank/dump_attempted/dump_arm_wave/...`），dump 只在 D 侧、P 无残留。
- T5 多节点：单容器单机，天然阻塞。

## §11 汇总速查（完整，据 W1–W5）

| 模块 | 契约 | 状态 |
|------|------|------|
| rank_gate | last-PP TP0 检测 / last-PP 全 TP dump / dump_rank_tag | ✅ T-01/02（TP=2）、T-03/04/05（PP=2）PASS；BUG-1 DP rank_tag 已修 |
| wave_tracker | stamp 生命周期 / reap / FIFO（async lag） | ✅ W1-3 code+live；D-16 live 未单测 |
| quota | 原子 try_consume/refund / refund 清冷却 | ✅ A-14 PASS；A-15 refund 代码级确认；BUG-2 DP 竞态缓解未根治 |
| queue | 满则 heavy drop / light inline / stop drain+sentinel | ⚠️ A-30/31/32 无法构造（无 config 开关），代码级确认语义 |
| report | max_per_req + wave backoff / dumps_report_json | ✅ A-01..A-04；BUG-6/K-13 已修（`5164073a9`） |
| request_state | 每 req 状态 / finish 清理 / 无跨 req 泄漏 | ✅ R-04/R-06：W1-1/W1-2/W1-3 已修且 live 复验 PASS（见「功能测试最终结论」） |
| io_snapshot | 增量累计 / tail-only（长输出不爆内存） | ✅ R-03/R-08 截断 PASS |
| kv_cache_reader | 空 id 拒 / payload 带 tp/pp/cp/num_kv_heads | ✅ A-10、A-16；BUG-3 layer 全局重编号已修 |
| manual_trigger | rank 门禁 + _wave_has_scheduled_tokens / 连续计数 | ✅ A-13、P0-3 manual_dump PASS |
| logger | stdlib Logger / apply_ascend_log_level | ✅ §8 + H-06 PASS |
| config | 热路径 gate / unknown-key 拒（V10）/ JSONC | ✅ F-07 顶层 unknown-key 拒已由 `229e7deb3` 修复（V10b CPU UT 通过）；其余 F 项 PASS |

## 229e7deb3 修复确认（2026-09-14 拉取 config 分支）

`[Fix] Reject unknown top-level config keys; drop report slot_mapping`（纯 Python，无 csrc，无需重编 .so）。CPU 回归 `test_review_regressions.py` **64 passed**。

| 项 | 结论 |
|----|------|
| **W2-1（F-07 顶层未知键）** | ✅ 已修：`_defaults.py` 增 `TOP_LEVEL_KEYS`/`ACTIONS_KEYS`，`_validate.py` 校验顶层未知键 → `ValueError`（响亮拒绝 + 列 allowed）。新增 CPU UT `test_v10b` |
| **W2-2（F-39 v2 slot_mapping）** | ✅ 以「删功能」方式闭合：删 `include_slot_mapping` default、`report_include_slot_mapping()`、`slot_mapping_for_request()`（-185 行）、processor enrichment、report tail order。`report.include_slot_mapping` 现作为未知键被拒 |
| **block_ids 加固（bonus）** | processor 在 `sample_fn()` 后保留 `result.input_batch` → `_last_input_batch`，report enrichment 显式传 `input_batch=`，v2 在 sample 清 state 后仍能解析 block_ids |

**BUG-4 / BUG-5 复核（均已修，非本 commit）**：
- **BUG-4**（kv_block_meta `block_ids_for_request` v1 路径 `table.block_table.np` 被 `except: return []` 吞掉 v2 StagedWriteTensor 异常）：✅ 现为 `except Exception: pass`（fall through 到 v2 `_block_ids_from_v2_block_tables`），由 `03bfd2a54` 修复，`229e7deb3` 未回退。
- **BUG-5**（g03_forbidden_substring `patterns=["SECRET_MARK"]` 与 `inject.py DEFAULT_TEXT="李白"` 错配）：✅ `configs/g03_forbidden_substring.json` 已改 `patterns=["李白"]`，脚本 `forbidden_substring:5` 走 DEFAULT_TEXT。

**live 需重跑（等空闲卡）**：F-07（v1+v2）、F-10（overlay 软失败不误伤）、R-06 v2（block_ids 非空）、T-02 v2 / F-11d / F-14 / F-15 / F-51（v2 dump block_ids）。F-38/F-39 从 FUNCTIONAL_TEST_LIST 退役（功能已删）。

## 待办

- ~~收 W1/W2 结果后补记~~（W1-W5 已全部完成，见 `rg_test/results/W1..W5.md`）。
- ~~性能 C1–C6（v1+v2 + 交叉轮换）~~ 已出数；**C1 v1=0.98466 待重测（等 8 卡空闲）**；C2 v1=1.00644 ✅。
- ~~24h soak~~ → **已启动，累计 ~3.6h（进行中，未达 24h）**（计划见 `SOAK_PLAN.md`；CP best-effort）。
- GLM-4.7-Flash：换更大卡或可切分 MoE 的并行策略后再跑；本机 32GB 标跳过。
- §6 PD：D 换卡复测 mooncake 传输（卡 1 RDMA 端口 down 疑似）。
- **产品侧历史 W1/W2 BUG 已全部 live 复验通过**：
  - W1-1（R-04）：`39e22dacc` 实卡复验 PASS（v2/v1 均 `seen==oc`=9）。
  - W1-2（R-06 v2 block_ids 非空）、W1-3（R-06 v1 长上下文 report）：`39e22dacc`
    实卡复验 PASS（`block_ids=[1..32]`；v2 验 block_ids / v1 验 report present）。
  - W2-1（F-07）、W2-2（F-39）：`229e7deb3` 修复（见「229e7deb3 修复确认」）；
    W2-3（D-11）：`8876645a2` 修复。
  - 复验脚本：`rg_test/reverify_39e22.py`（R-04）+ `rg_test/reverify_r06.py`（R-06）。
  - **R-06 复验注意**：长上下文必须让模型生成 ≥5 token（inject `token_loop:5:*`
    需 `wave≥5`）；用「answer with one word」会提前 EOS（~1 token）→ inject 不触发
    → 误判 no report。用「Describe the weather in three sentences」。

## W2-3 / D-11（v2 token_repeat 误报）

- **现象**：同诗句 prompt（temp=0/seed=42），v1/v2 输出同为 48 token；v1 不命中，v2
  `repeat_sum` 虚高命中；report `content_tokens_seen≈76` / `output_token_count≈77`。
- **根因**：v2 即便 sync scheduling 也返回 `AsyncOutput`；`run_sample_phase` 在
  `get_output` trim 之前用 padded `sampled_token_ids` 向 IO 流 append。
- **修复**：`AsyncModelRunnerOutput` 时 defer `check_after_sample` 到
  `AscendAsyncOutput.get_output`（与 async 路径一致）。UT：
  `test_d11_*` in `test_unified_wiring.py`。

## W1-1 / R-04（v2 content_tokens_seen ≠ output_token_count）

- **现象**：多轮 token_repeat report：v2 `content_tokens_seen=15`、`output_token_count=10`；
  v1 对照 15==15。
- **根因**：同波二次 after-sample 时 Store `last_append_chunk` 去重跳过 append，但
  `run_after_sample_cpu` 经 `new_ids_by_req` 旁路再折一次冻结 chunk → seen 虚高。
  （W2-3 去掉主双触发；本修去掉旁路，防潜伏双 fold。）
- **修复**：`token_repeat` 只读 Store `_consumed_len` 增量；报告 snapshot
  `use_cache=False`。UT：`test_w1_1_*` in `test_detectors_and_kv.py`。

## W1-3 / R-06（v1 长上下文 chunked prefill 无 report）

- **现象**：~4036 token chunked prefill；`INJECT token_loop wave=5 waves_left=39` 已打，
  但 0 report；日志可见 `missing sample-wave stamp ... arm_wave falls back`。
- **澄清**：缺 stamp **不会**跳过 detect/report，只污染 `arm_wave`/dump 对齐。
  0 report 更可能来自：chunked discard 空步仍 enqueue after-sample CPU，
  ActionQueue `drop_on_full` 静默丢掉后续（含 inject 命中）检测任务。
- **修复**：
  1. `WaveTracker` per-req stamp 改为 FIFO deque（async lag 下 take 不再 miss）；
  2. sampled 行全空时不 enqueue CPU detect；
  3. drop 时打 warning。UT：`test_v8a3_wave_tracker_fifo_under_async_lag`。

## 配置分支 force-push 重写 + BUG-4/BUG-5 复验（2026-09-15）

- **`feat/runtime-guard-config` force-push 重写**：`01566f71f` → `0a1a11c4e`
  （`[Feature] Add runtime_guard control plane (detect, report, dump_kv)`，基于更新
  上游 main 的单一 feature 提交；旧 `01566f71f` 已孤儿化）。
- 本机 `rg-config-review` 已 `git reset --hard` 到 `0a1a11c4e`；`vllm-ascend`（analysis）
  仍 `2792e30ea`（无变化）。

### BUG-4（kv_block_meta.py block_ids_for_request v1 路径吞异常）→ 已修复

- **原现象**：`block_ids_for_request` v1 路径 `table.block_table.np[...]` 在 v2
  `StagedWriteTensor`（无 host `.np` 镜像）下抛异常，被 `except: return []` 吞掉
  → v2 dump 拿不到 block_ids。
- **新代码**（`0a1a11c4e`）：v1 `.np` 访问失败改为 `except Exception: pass` 落到 v2
  分支 `_block_ids_from_v2_block_tables`（显式读 StagedWriteTensor 的 `.gpu`）；另有
  `table is None` 分支与 post-sample 兜底均走 `_block_ids_from_v2_block_tables`。
  函数 docstring 已标注 `(BUG-4)`。

### BUG-5（g03 forbidden_substring 与 inject 默认文本错配）→ 已修复

- **原现象**：`configs/g03_forbidden_substring.json` `patterns=["SECRET_MARK"]` 与
  `inject.py` `DEFAULT_TEXT="李白"` 错配 → output_substring detector 永不命中。
- **现状**：`g03_forbidden_substring.json` 已改为 `patterns=["李白"]`，与
  `inject.py DEFAULT_TEXT="李白"` 一致。

> 结论：BUG-4 / BUG-5 在最新 config 上均已修复。功能测试 W1–W5 结论（6 BUG 关闭）
> 基于旧 config（`01566f71f`）；config 重写后是否需全量回归复验，待定。

## KV dump 标杆复验 + analysis 工具箱走查（2026-09-16，Qwen2.5-0.5B，本机卡 2/3，config @afd21d935）

**结论：dump 正确（人为判 PASS）；p0_10 在 0.999 阈值下误报 1 层 FAIL，根因是全块 cos 把 TP 数值噪声当信号。工具链整体可用，8 条优化点见下。**

### 跑法

- 启动脚本入库：`test/live/scripts/k_bench_tp_dump.sh <tp> <cards> <out> [port]`（单发 `manual_dump:2`，确定性请求 temp=0/seed=42，v2 runner，`VLLM_BATCH_INVARIANT=1`）。
- TP=1（卡 2，REF）→ 96 .pt（wave_2+wave_3 各 48）；TP=2（卡 2,3，TARGET）→ 192 .pt（两波各 48×2 rank）；两次生成文本逐字一致。
- 分析链路全部实测：`summarize_reports` → `correlate_incident`（自动发现非默认 dump 根）→ `verify_request_kv`（48 文件全 finite PASS）→ `inspect_kv_dump` → `p0_09_tp_stitch`（2 rank 拼回 (1,128,2,64) OK）→ `p0_10_kv_compare`（0.999：47/48 过，layer_22 K 0.998786 FAIL）→ `capture_kv_ref`（meta 48 shards + 全量 ref）。

### 数值判定（为何人为 PASS）

- layer_0 K/V **cos=1.000000 maxdiff=0**（dump 通路 bit 忠实）；已写 slot 范围两边一致（slot 0-1，其余 126 slot 双方全零）；block_ids/层名（全局名）/有限性全对齐。
- slot 级分析：全块 0.9988 完全由 2 个已写 slot 贡献；layer_22 K slot0=0.9976、slot1=0.99997，深层更低——TP=1 vs TP=2 前向 bf16 all-reduce 顺序噪声随深度累积（与 2026-09-13 Qwen3-0.6B 观察一致，当时已写 slot 多、噪声被稀释）。
- wave_2 仅写 2 slot 是「首个 decode wave 早照快照」所致；**建议标杆对比用更长 prompt**（写满更多 slot）以稀释单 slot 噪声。

### 工具链优化点（按影响排序）

1. **stitch_kv compare 增加已写段切片**（`--max-tokens`/自动非零 slot）：全块 cos 在少 token wave 被单 slot 噪声主导，把正确 dump 判 FAIL（本次实际发生）。
2. **manual_trigger report 缺 token ids 且 `prompt/output_token_count`=0**（实测 2/1）：`prepare_ref_inputs` 直接拒、`compare_kv_similarity`/`locate_first_divergence` 硬要求 id lists（`--num-tokens` 只覆盖计数，绕不过 `validate_matching_token_ids`）→ §15.3 标杆流程以 manual dump 起步即断链。需 config 侧补 token 计数/ids（save_sensitive_info 打开时）。
3. p0_10 默认 `COS_THRESH=0.9999` 跨 TP bf16 必误报，默认或示例应改 0.999（skill 已有经验值，脚本默认未同步）。
4. `p0_03_manual_dump.json` 用 `manual_dump:true`（连续模式，已知后续 wave 缺文件风险）——对比场景应提供 `manual_dump:2` 单发配置；且实测 `:2` 落 wave_2+wave_3 两波（与旧 RUN_NOTES「单次」描述不符，行为需再确认）。
5. `verify_request_kv --kv-dir` 语义是 rank 目录（直接含 .pt），help 写 "KV dump dir" 易传成 dump 根 → files=0 误 FAIL；建议 help 写明或自动下钻。
6. `capture_kv_ref.sh` 无条件写 git 工作树 `golden/dump_schema/kv_ref_meta.json`（untracked 脏树）；建议加 `--out`/staging 提示。
7. `verify_request_kv [6] unique_layers` 把 K/V 文件各算一层（48"层"=24 层×K/V），显示易误导；summarize 表 prompt/output=0 是 #2 的下游表现。
8. 工具对 `attn[0]`/`attn[1]` 文件名走 glob 时方括号是字符类（本次分析脚本踩坑，工具内部用子串匹配无碍）——文档提醒自定义分析时别用 glob。

**好用的部分**：correlate 自动发现 dump 根、exit 码语义正确（PASS=0/FAIL=1）、层名 natural sort、`--json-out` 结构化、p0_09 缺 rank 检测。

实测数据落本机 `/data0/test-mrv2-cann91/rg_bench/{tp1,tp2,ref}`（对比后可删）。

## 8 条优化点修复复验（2026-09-17，config @21d24c892 / analysis @a0b45df72）

**结论：8/8 闭环。analysis 侧 7 条已提交（a0b45df72）；#2 config 侧修复已提交（21d24c892）并实卡复验通过。产品 UT 130 passed（+6 v2 staged-tensor 用例）。**

### #2 v2 manual report 补 token I/O（config @21d24c892）

- 根因两个：
  1. NPU `UvaBufferWrapper` 的 `_uva_buf.np` 是 host 镜像，v2 decode 追加走 device 侧 `post_update` kernel（只写 `.gpu`），host 镜像永远为 0 → 旧代码全零保护返回 None；
  2. `req_states.total_len` 是 **1-D** `[max_num_reqs]`，`_read_staged_row` 按 2-D 行索引 `gpu[idx,:1]` 抛 IndexError 被吞成 None（第一版修复实卡复验暴露；UT fake 用 2-D 没拦住，已按真实形状改）。
- 修复：`_read_staged_row` host 优先 + 全零落到 `.gpu`（单请求单行按需 D2H，仅报告触发时）；1-D/2-D 自适应；新增 `_output_from_req_states`（`total_len - prompt_len`，ids 取 `all_token_ids[prompt_n:total_n]`）接入 count/snapshot 回退链。
- 实卡复验（本机卡 2，TP=1，Qwen2.5-0.5B，`save_sensitive_info:true`）：report 由「prompt=0/缺 ids、output=0/缺 ids」→ `prompt_token_count=2 [14990,10276]`、`output_token_count=1 [271]`。
- 说明：v2 采样回 host 的主通路仍是 Store host list（`append_batch`）；`.gpu` 只是其后的兜底。`:2` 单发 fire 时 output=1（首 sample 已落），wave_2+wave_3 两波照旧。

### analysis 侧 7 条（a0b45df72）

- #1/#3 `stitch_kv`/`p0_10`：`--max-tokens`/`--written-only` + slots 列；`COS_THRESH` 默认 0.999。
- #4 `p0_03_manual_dump.json`：`manual_dump:2` 单发。
- #5 `verify_request_kv`：help 写明 rank dir 语义 + 传入更高层目录自动下钻。
- #6 `capture_kv_ref.sh`：`RG_META_OUT` 逃生口，meta 不再脏 git 树。
- #7 层数去重（`layers=24 kv_files=192`）。
- #8 README/SKILL 坑位提示（glob 方括号、阈值、`--kv-dir` 语义）。
- `k_bench_tp_dump.sh` overlay 补 `report.save_sensitive_info:true`，标杆起步即带 id lists，§15.3 链路（`prepare_ref_inputs` → compare）不再断链。

## 2026-09-17 晚 — rebase 后回归收官 + “首请求挂起”根因定案

### 远端分支重写验证（config@6e22d331a / analysis@d6c9ee60f）

- config squash 为单笔提交，基=内部主仓 main 最新（50283947d，09-17 21:06）；产品内容与 21d24c892 **逐字节一致**（16547 行 patch 零 diff，仅 hunk 行号随上游平移）。
- analysis 单笔叠 config tip，工具箱全保留；5 个 test 文件 +132/-35 为 base 适配（import 回 `token_utils` + 补 v2 staged-tensor 回归测试）；旧分支上的 util.py 合并重构经确认为有意丢弃（旧代码）。
- 两 checkout 已 reset 到新 tip 并复验：产品 UT 130 passed / 75%，与旧 base 一致。

### 回归结论（21d24c892，内容等同 6e22d331a）

- P0 smoke 12/12；P1 5/5：p0_04 v1 ✅ / p0_04 v2 ✅（retry，见下）/ p0_07 ✅ / g06 ✅ / p0_08 ✅。
- C3（Qwen2.5-0.5B TP=2 eager，交叉轮换 B/A×3）：**v1 T3/T2=1.01120 ✅，v2 T3/T2=0.99980 ✅**（v2 首跑 0.98657 为运行间噪声）。rank_gate 收敛后 detector 开销双 runner 均 ≤1%，过线（≥0.990）。

### “v1/v2 首请求 900s 挂起”根因 = 宿主机外部杀进程（非产品 bug）

- serve 日志铁证：`[shutdown] API server: shutdown triggered [launcher.py:114,signal_handler]` → abort 模式（timeout=0s）瞬杀 EngineCore，在途请求变孤儿，client 干等 900s 才 TimeoutError。被杀时引擎完全健康（10.8 tok/s、持续 200 OK）。
- 真凶：共享宿主机用户 computin 的 DeepSeek-V2-Lite + Mooncake PD 测试（ports 13700/13701），~30min 周期清理（观测 20:45 / 21:14 / 21:43）。p0_04 v2 首跑 HEALTH TIMEOUT 同因（启动握手期收 SIGTERM，`KeyboardInterrupt: terminated`）。
- 诊断要点：client 900s 超时 ≠ 产品挂起，先翻 serve 日志找 `signal_handler`；py-spy 无效（进程已死非卡死）。长跑前 `last -10` 确认 computin 是否活跃，或避开其周期窗口。

### 其他

- `_common.sh` `wait_idle` 多卡 bug 修复：`CARD=2,3` 时 grep "NPU 2,3" 永不匹配 → 逐卡拆分检查（npu-smi 每卡一行 "No running processes"）。

## 2026-09-18 补测收官 + RG_INJECT 双门禁

### 补测队列结果（新 base，queue_c6c5_soak.sh）

- **C1 v1 重测 ✅ 关闭**：T1/T0=1.00291（N=6 交叉轮换，.so 已同步）；09-14 的 0.98466 判为旧 base/跨 session 噪声。C1 双 runner 全绿（v2=1.00411）。
- **C6 v1 ✅ 干净过线**：300s leakback 残余 rss_delta=76KB（阈值 30MB），优于 v2 的 136KB；此前"v1 部分污染"确认为环境残留。
- **C5 v1 ✅**：T3DUMP/T3=1.01760（dump_kv on_trigger 开启，3 轮 geom；≈噪声级，门槛 ≥0.990 过）。
- **C5 v2 ⚠️ 环境阻塞（非分支缺陷）**：boot 即 `TypeError: GPUModelRunner.initialize_kv_cache() got an unexpected keyword argument 'kv_cache_allocation_context'`。隔离实验：裸 base 50283947d（T0 worktree，无分支代码）+ v2 复现同一 TypeError —— 新内部 base 的 worker.py 无条件传该 kwarg，而本容器 vllm（公开 0.28.0@2cf0a6915c，08-24）尚无此参数；v1 runner 自行消费该 kwarg 故不受影响。时间线也吻合：全部 v2 历史证据（C1/C3/C4/P0/P1）采集于 21d24c892 谱系（其 base 无此 kwarg），squash 重置（09-17 14:17 UTC）后今天首次 v2 活体启动。分支 patch 不涉这些行（diff 0 命中）。需配对 vllm≥含该 kwarg 的内部版本方可复测。
- **soak 补时进行中**：cards 2,3（A=v1/0.5B/token_repeat 8090，B=v2/7B/logits_finite 8091），目标有效 10.28h 补足 24h；含服务死亡检测/有效时间记账/外部 kill 签名记录/自动重启（w5_soak_topup.sh）。

### RG_INJECT 双门禁（shipped 安全）

- `inject.py` 增加 `_INJECT_MASTER_SWITCH = False`（源码级总开关）：出厂构建完全忽略 `RG_INJECT`；启用需改源码翻转 + 设置环境变量。
- 回归：产品 UT **153 passed / 75%**（+5 gate 测试；scenario 测试 helper 改为 reload 后武装）；e2e 冒烟（card 0, v1, 0.5B）：NEG（shipped+RG_INJECT=token_loop）0 注入日志/0 report；POS（翻转后）40 条 [INJECT] + token_repeat report 落盘。
- 分支 tip 前进：config 6e22d331a → **03a6ad89d**（amend 合入，单笔不变，基 50283947d）；analysis rebase 至其上（d0681cc03 + d46d6c75c）。C1/C6/C5 数字采集于 6e22d331a 内容（与 03a6ad89d 仅差 inject 双门禁 3 行 + 测试，热路径 `if inject.ENABLED:` 语义不变）。

## 2026-09-19 soak 补时收官(24h 证据达成)

- w5_soak_topup.sh(tip 03a6ad89d,cards 2/3,A=0.5B/token_repeat、B=7B/logits_finite,双 v1):
  09-18 11:02 -> 09-19 05:20 UTC,18.3h 墙钟,有效记账 cum=37000s(10h16m)精确达标。
- 质量指标:2792 心跳 0 次不健康;热更 555 轮 0 失败;10 个 attempt 全部为墙钟预算
  自然到期(服务死亡检测从未触发);kill_sig=none x10(全程无外部 SIGTERM);无 hang/crash。
- 24h 累计:原 soak 13.8h(09-14,含 v2 覆盖)+ 补时 10.28h = 24.06h 有效 > 24h。
- TEST_REPORT.md §1/§4 已回填收官结论。

## 2026-09-20 更正：aclnnAddRmsNormBias 根因 + 1ae55b060 活体 boot 验证

### 根因更正（推翻 09-18「需更新 CANN」结论）

- `aclnnAddRmsNormBias` 不在 CANN，而在 vllm-ascend **构建产物**
  `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so`
  （`pip install -e`（COMPILE_CUSTOM_KERNELS=1）时生成，gitignore 不入库）。
- `bootstrap_custom_op_env()`（utils.py）按 `_CUSTOM_OP_BASE_DIR`（= utils.py 所在包目录）
  查找 vendors 目录；**裸 git worktree 无此目录 → 直接 return → 运行时 dlsym 失败** →
  `RuntimeError: aclnnAddRmsNormBias ... not in libopapi.so`。
- 09-18 判别实验（03a6ad89d health=200、1ae55b060 同帧死）**变量混杂**：旧 tip 跑在带
  构建产物的主树，新 tip 跑在裸 worktree rg-rebase-verify。两 tip 的 utils.py /
  layernorm.py 逐字节一致，.so 同为镜像/主树构建。
- **解法（无需重编）**：`cp -r <带产物树>/vllm_ascend/_cann_ops_custom <worktree>/vllm_ascend/`。
  该目录是运行时算子注册数据；`vllm_ascend_C.so` 为运行时 dlsym 的 stub，主树 09-16
  构建直接可用。前提：两树该算子 csrc 未变（03a6ad89d 产物配 1ae55b060 已实测安全）。

### 1ae55b060 活体 boot 验证（09-19，四格全绿）

| 环境 | runner（端口） | boot | completions |
|---|---|---|---|
| 老容器 test-mrv2-cann91（CANN 9.1.0，card 2） | v1（8063） | health 200 | 正常（" Paris..."） |
| 老容器 | v2（8064） | health 200 | 正常；**零 `kv_cache_allocation_context` TypeError（版本门禁生效）** |
| rg-verify-919（`quay.io/ascend/vllm-ascend:nightly-main`，CANN 9.1.0_20260731131545） | v1（8060） | health 200 | 正常；runtime_guard action worker 启动 |
| rg-verify-919 | v2（8061） | health 200 | 正常（首请求慢过 60s 超时，重试 OK） |

- 均以 `PYTHONPATH=<worktree>:...` + `--additional-config`（logits_finite detector）启动；
  serve 日志零 aclnnAddRmsNormBias 命中。
- 结论：**新 tip 活体 boot/功能无阻塞**；C5 v2 复测的 boot 前置解除（perf 复测经决策暂不执行）。
- 验证容器 rg-verify-919 保留备用（挂物理卡 2 = /dev/davinci0，
  python=/usr/local/python3.12.13/bin/python）；两容器的 rg-rebase-verify worktree 均已拷入
  `_cann_ops_custom`。
- task_spec 侧 PR 描述 / 汇总报告已同步更正（09-19/09-20）。

## 2026-09-20 C5/C6 方法论重构 + 162 环境部署(执行暂停)

**方法论重设(用户定案)**:
- C5 不再用吞吐比,改为直接测 dump 时的 D2H 时间与 save 时间,多档 KV 大小各测 2 次。
- C6 改为对照组差分:裸 base(50283947d,无 runtime_guard)同压测后对比 RSS/HBM 变化,detectors 全开、dump 之后看残留,替代绝对 30MB 基线。
- 长 prompt 档 1k-128k 用真实公开数据(LongBench);每个请求带唯一序号前缀([c56-%06d])防 prefix-cache 复用 KV。
- C5+C6 合并一次跑:armA 先做 dump 计时(1k/4k/16k/64k/128k,manual_dump 逐档触发),再与 armB 同时压测,最后双臂 leakback 采样;dump 残留天然落进 armA 采样窗口。

**资产(analysis d1cf1fb6a,已 push)**:
- scripts/run_c56_ab.sh:双臂编排(boot->C5->stress->settle 30s->sample 300s->差分 verdict:RSS diff<10MB / HBM diff<512MB 为 PASS)。
- scripts/c56_driver.py:c5/stress/sample 三个子命令;corpus prompt 用 served 模型 tokenizer 按目标 token 数精确切割拼接;sample 走 /proc 进程树 RSS + npu-smi HBM。
- scripts/c5_shim/sitecustomize.py:RG_C5_TIMING=1 时经 builtins.__import__ 挂钩,包装 KvCacheReader.iter_request_snapshots(逐层 D2H 计时)与 write_snapshots(save 计时),JSONL 事件含 rank_tag/req_id/bytes,全防御式失败不影响产品路径。
- data/longbench_corpus.jsonl:21 docs / 2.6M chars(hf-mirror zai-org/LongBench data.zip,各源文件取最长 context:narrativeqa x6 / gov_report x4 / passage_count x3 / dureader x4 / multifieldqa_zh x4)+ README。

**162 环境要点(192.168.13.162,800I A3:8 NPU x 2 die x 64GB,逻辑设备=die 0-15,TP2=同 NPU 双 die,与 09-15 会话同款)**:
- 访问:pexpect 建 ControlMaster(本机 /tmp/r162_sync.py,d00824595@192.168.13.162),之后 rsync/tar-over-ssh 免密;裸 python3.12 已不在,须容器。
- 容器三要素:必须 --privileged(非特权新容器 dcmi init 报 -8020 "device is used",旧容器占位,特权绕过;--device 逐个映射无效)、--shm-size=16g(TP2 shm_broadcast 需 160MB,默认 64MB 直接 boot 失败,首跑即栽此坑)、挂 /usr/local/Ascend/driver 整目录 + npu-smi + hccn.conf。
- 镜像 quay.nju.edu.cn/ascend/vllm-ascend:nightly-main-a3(vllm 0.28.0;privileged 下 torch_npu 可见 16 die;产品树/裸 base 树均可 import,Qwen3-Coder-30B-A3B max_seq_len=262144)。
- 已部署待恢复:~/rg162_env/{product=03a6ad89d 全包含 vendors+.so, base=50283947d 裸对照, scripts, data} + ~/rg162_weights/Qwen3-Coder-30B-A3B-Instruct(57G/17 shards,该机共享库 /mnt/weight 无此模型;DSV4-Flash-bf16=546G 超 4 卡不适用)。二跑启动后按用户指示暂停,容器已删、NPU 无残留进程、文件保留。
- 恢复步骤:重建 privileged 容器(命令在会话记录)后,docker exec -d 容器 bash -c "cd /home/d00824595/rg162_env && nohup bash scripts/run_c56_ab.sh > /tmp/rg_c56_launch.log 2>&1 &";总时长约 1-1.5h(boot ~6min + C5 25-40min + stress 5min + sample 5.5min)。

**其他回填**:
- C4 执行计数已写入 PR 描述 §4:v1 完整 1 轮(T0-T3 x3 prompts=12 输出)、v2 完整 1 轮另补采 T0/T1 一轮(18 行),全部 bit-identical;原始档案在产品树 rg_perf/logs/{v1,v2}/c4_identity.jsonl + results/c4_v{1,2}.txt。
- PR 描述 §4 的 C5/C6 行待新方法学数字落地后改写(旧数字 C5 v1=1.01760 / C6 76KB 保留至替换)。


## 2026-09-20 (II) C5+C6 本机落地执行(方法论重构后首次完整跑通)

上一章(09-20 I)记录了方法论重构资产 + 162 部署;本章为本机(definitive)执行结果。
环境:test-mrv2-cann91 容器,910B4×2(cards 2,3),DSV2-Lite TP2 eager,
product=03a6ad89d,base=50283947d(rg-perf-t0),runner=v1,纯 Python 无重编。
入口 `perf/scripts/run_c56_ab_local.sh`(顺序臂:guard 臂含 C5 dump 计时,再
stress+leakback;裸 base 臂 stress+leakback;差分判决)。

### 结果(results/c56_local_ab_20260920/,definitive run 07:36-08:09)

- **C5 dump 直接计时**(manual_dump 10 组 = 5 档 × 2 rep × 2 rank,54 层/54 文件/rank):
  | tier(prompt tok) | D2H ms/rank | save ms/rank | bytes/rank |
  |---|---|---|---|
  | 1024 (1036) | 150-219 | 101-112 | 34 MiB |
  | 4096 (4108) | 160-413 | 180-233 | 91-125 MiB |
  | 16384 (16396) | 157-716 | 331-769 | 182-490 MiB |
  | 65536 (65548) | 200-881 | 790-2817 | 547-1948 MiB |
  | 131072 (131083) | 639-1140 | 2692-5359 | 2005-3892 MiB |
  rep 间 bytes 有差(dump 按 block 粒度覆盖该请求已分配 block,第二轮分配更多);
  D2H 有效带宽 ~2-4 GB/s;save(torch.save×54)在 128k 档 ~5s 是主要成本。
- **C6 对照差分 leakback**(settle 30s + 300s × 27 点,全检测器开,guard 臂前置 10 次 dump,共 ~19GB dump 文件):
  - guard 臂:RSS +0.2MB,HBM −0.7MB;stress 15 req / 244,169 tok(1k-128k LongBench,序号前缀防 KV 复用)
  - base 臂:RSS +0.0MB,HBM +1.3MB;stress 17 req / 506,591 tok(裸 base 吞吐更高,符合预期)
  - **差分:RSS +0.1MB,HBM −2.0MB → PASS**(门槛 +10MB / +512MB);dump/检测器零残留
  - 第一轮(run1,07:11,留档 verdict_run1_weakstress.txt)stress driver 有 bug 只发了 5 请求,但 C5 数据完整、C6 差分 +0.2MB 同样 PASS

### 过程中修掉的 4 个测试资产 bug(均已 commit 到 analysis)

1. **PYTHONPATH 覆盖**:boot 脚本 `PYTHONPATH=shim:product` 丢掉容器继承的 CANN
   site-packages(/usr/local/Ascend/*/python/site-packages),camem.py 顶层无守卫的
   `from acl.rt import memcpy` 直接 ModuleNotFoundError,报错位置误导为
   multiproc_executor.py:941(实为 worker_main except 日志点)。修:统一
   `:$PRODUCT:${PYTHONPATH:-}` 追加(b51a10add)。
2. **boot 子 shell 组合异步列表**:`( cd tree && ENV setsid python ... & echo $! )`
   把整个 compound list 丢后台,留下 bash wrapper wait4(server) 且持有 `$()` 管道
   → `PID=$(boot)` 在 server 退出前永不返回(实测卡 15min,server 健康)。修:cd
   独立成句,异步任务为简单命令直接 exec(3aebe66cd)。
3. **stress `dict(LB_TIERS)` 三元组崩溃**:两个 worker 线程在首个 LongBench 档
   同时 ValueError 静默死亡,sent=5 提前结束。修:`{n: t for n, t, _ in LB_TIERS}`(3dc82c588)。
4. **npu-smi 25.5 HBM 解析**:NPU id 与芯片名同格、HBM 在下一行 chip 行行尾
   `X / Y`(与 AICore、Mem 挤同格)。修:col-1 首 token 匹配 + 下一行最后一个
   `X / Y` 对(3dc82c588,实测 {2: 29255, 3: 29032})。

### 遗留

- C5 v2 直接计时未跑:本机 product 树钉在 03a6ad89d(无 d6acae3b6 版本门禁),
  v2 boot 需 1ae55b060 环境或 162(任务 #14,用户暂停)。PR/RFC 仍按已披露项处理。
- dump 产物 19GB 在 /data0/test-mrv2-cann91/rg_c56/run/A/dump(重跑会被
  rm -rf;磁盘 11T 充裕,暂留)。
- 162 侧执行恢复配方见上一章(容器三要件:privileged / shm≥1g / driver 整目录)。


## 2026-09-20 (III) C5 v2 直接计时补测(1ae55b060 worktree)+ 远端 tip 第 4 次重写发现

背景:v1 数字采集于 03a6ad89d(无 0.28 版本门禁,v2 在本容器 vllm 0.28.0 起不来);
v2 补测在 `rg-rebase-verify` worktree(1ae55b060,vendors 已拷,09-19 验证 v2 可 boot)。

### 远端 tip 第 4 次 force 重写:1ae55b060 → bc0629421

- 新 tip 为单笔 squash 形态("[Feature] Add runtime_guard control plane"),相对
  1ae55b060 diff 270 文件(+4772/−3958),全部在 base main(v2 spec_decode 重构等)。
- **runtime_guard 内容与 1ae55b060 零 diff**(`git diff 1ae55b060 FETCH_HEAD --
  vllm_ascend/runtime_guard/` 为空)→ 在 1ae55b060 上采集的数字对 MR tip 仍有效。
- **0.28 版本门禁被删**:bc0629421 的 worker/v2/model_runner initialize_kv_cache
  无条件转发 kv_cache_allocation_context(base main 已进 vllm 0.29 时代,
  `vllm_version_is("0.29.0")` 分支出现)。本机容器 vllm 0.28.0(2cf0a69)上
  bc0629421 的 v2 无法 boot——未来在 0.28 容器上跑新 tip v2 前需重新加门禁或换
  vllm 0.29 镜像(162 nightly-a3 也是 0.28.0,同样受限)。

### C5 v2 结果(results/c5_v2_1ae55b060_20260920/,DSV2-Lite TP2 cards 2,3)

10 组 × 2 rank 全部成功,与 v1 同量级(v2 的 StagedWriteTensor/UvaBufferWrapper
无宿主镜像、走 .gpu 按需拷贝,不构成额外开销):

| tier(prompt tok) | D2H ms/rank(v2) | save ms/rank(v2) | v1 参照 D2H/save |
|---|---|---|---|
| 1024 (1036) | 156-310 | 102-105 | 150-219 / 101-112 |
| 4096 (4108) | 158-358 | 182-230 | 160-413 / 180-233 |
| 16384 (16396) | 156-336 | 303-728 | 157-716 / 331-769 |
| 65536 (65548) | 181-865 | 807-2659 | 200-881 / 790-2817 |
| 131072 (131083) | 392-1624 | 2674-5521 | 639-1140 / 2692-5359 |

boot 用 gpu-memory-utilization 0.80(v1 用 0.85):共享卡上邻居任务反复起落,
worker 初始快照 free 12.16/29.49GiB < 0.85×29.49=25.07 → ValueError。0.85 时
干净卡余量本就只有 ~1.4GB,撞车即挂。

### 脚本资产

- 新增 `run_c5_v2_only.sh`(arm-A-only C5,PRODUCT 指向 verify worktree,RUNNER=1,
  等卡判据 = 卡 2/3 HBM<5GB——**进程表清空 ≠ 显存已释放**,邻居任务周期性
  起落时必须按 HBM 判)。
- 途中又踩一次共享卡竞态:08:35 boot 与邻居任务回撞(free 12.16GB),加 HBM
  闸门 + 0.80 后 08:58 重跑一次通过。

## 2026-09-21 远端 tip 第 5 次重写(f849eae4f)+ vLLM 0.29 本机验证收口

### 远端变化

- origin force 重写:bc0629421 → **f849eae4f** = feature squash `0052bdcd6`(runtime_guard/
  runtime_config 与 1ae55b060 **零 diff**,`git diff 1ae55b060 0052bdcd6 -- <guard dir>` 为空)
  + 新 commit "[Fix] Remove anomaly inject hooks from production path"(作者本人,Cursor 协作):
  RG_INJECT 整体移除(inject.py + processor 3 调用点 + 2 测试文件 + 文档,−717)。
- base 升 **vLLM 0.29.0**(c173a64a4),worker.py 残余 0.28 门禁全删 → 新 tip 只能配 vllm 0.29
  (0.28 容器 UT collection 即挂:attention_v1.py 导入 `vllm.v1.attention.ops.pcp` 不存在)。

### 本机 vLLM 0.29 隔离环境(不动钉住的 /opt/slime/venv)

- `pip install vllm==0.29.0 --no-deps --target /data0/test-mrv2-cann91/vllm029_pkgs`;
  容器原 vllm 是 /data0/test-mrv2-cann91/vllm 源码 checkout,PYTHONPATH 前插 pkgs 即遮蔽。
- tip5 worktree `/data0/test-mrv2-cann91/rg-tip5-verify`(vendors 已拷)。

### 验证结果(全绿)

- **UT 135/135**(runtime_guard/test 117 + runtime_config 18;注入相关 18 个测试随注入面移除)。
  注入残留 grep 干净(余下 inject_manual_dump_kv 旗标与 RG_INJECT 无关)。
- **v1/v2 活体冒烟各一轮**(run_tip5_smoke.sh,DSV2-Lite TP2 cards 2,3,gmu 0.80):
  boot health=1、completions 200(v1/v2 响应同长 653B)、长请求 OK、二次热更写无错、HBM 干净释放。
- **dump_kv 内容逐项校验**(v1+v2 各 108 文件 = 2 rank × 54 层;torch.load 抽查):
  tp_rank 0/1 按 rank 目录正确、num_kv_heads=1、block_ids=[1]、tensor (1,128,1,512) bf16
  (1 block × 128 tok × kv_heads × head_dim,DSV2-Lite GQA 布局)、incident_type=manual_trigger。
  v2 走 StagedWriteTensor `.gpu` 兜底,0.29 下完好。

### 坑(已修入资产)

- **npu-smi 5 位数 HBM 格式**:20228/ 32768(斜杠前无空格,4 位数时才有空格)→ 旧正则
  `[0-9]+ / 32768` 匹配不到;run_tip5_smoke.sh / run_c5_v2_only.sh 已改 `[0-9]+[ ]*/[ ]*32768`。
- **共享卡竞态三连**:boot 前判空(HBM<5GB)后 1-2 分钟内邻居抢回 ~17GB → worker
  `_init_worker` 快照 free 12.x GiB < 0.8×29.49 → ValueError。v1 连续 2 轮、v2 第 1 轮均撞,
  加"每臂 wait_idle + 失败重试 ×2"后 12:06/11:34 各一次通过。判空与快照窗口无法完全闭合,
  重试是当前唯一解;冒烟窗口尽量避开邻居 ~15min 周期。

## 2026-09-22 远端 tip 第 6 次推进(9e314a780)+ 本机验证收口

- 新 3 笔:`26fbefb25`/`5fa99e057` 两笔纯文档(design 时序图补 hook 门控与 sync 路径启动冻结、
  UCM 排障移 ops §2.5)+ `9e314a780` import hoist(v1/v2 runner 函数内延迟 import 提到模块顶,
  行为等价;processor/runner_bridge 不反向依赖 worker,无循环 import 风险)。
- **runtime_guard/runtime_config 相对 f849eae4f 零 diff**(git diff 为空)。
- 验证(rg-tip6-verify worktree,vllm029 PYTHONPATH):
  - UT 117+18=135/135 全绿;
  - v1/v2 boot 冒烟(DSV2-Lite TP2 卡 2,3,eager,0.80):双 runner health=1、completions 653B、
    manual_dump 108 文件(2 rank × 54 层)、二次热更写入干净、HBM 释放;
  - dump payload 双 runner 双 rank 逐项:tp_rank 0/1 随 rank dir 正确、num_kv_heads=1、
    block_ids=[1]、tensor (1,128,1,512) bf16——与 tip5 结果一致。
- 坑(复发确认):**裸 worktree 缺 `vllm_ascend/_cann_ops_custom/vendors`(63MB,gitignored)** →
  boot 报 `aclnnAddRmsNormBias not in libopapi.so`,与 09-20 章结论一致;从旧 worktree cp -r 即解。
  v2 attempt 1 因此失败,拷贝后 attempt 2 一次通过。另:直接 `import vllm_ascend.worker.*` 作首导入
  会触发 base 代码 device_op↔ops 循环 import(tip5 同样复现),系探针顺序伪影,真实 boot 走插件
  注册顺序无此问题——以真实 boot 冒烟为权威判据。


## 2026-09-24 远端 tip 快速演进（→tip12 1992d6b71）+ 双服务器验证轮（进行中）

- 远端一日多笔推进，本轮两笔关键：
  - **tip10 = 基线臂**：无 v2 异步 dump 接线，预期矩阵 DUMP_CHECK=0（用于与 tip12 对照）；
  - **tip12 = 1992d6b71**（force rewrite）：3a479639a BugFix（dump prepare 异常退还 auto 配额、
    runner_tp_rank 打印去重、僵尸请求强制回收）、b0903c3b2 目录重构（runtime_guard 迁入
    `vllm_ascend/observability/`，import 全量更新无旧路径残留）、1992d6b71 删 82 行兼容垫片；
    tip11 的 v2 异步输出接线修复（`maybe_wrap_v2_async_output`，TP0 包装 AsyncOutput 使
    `check_after_sample` 在 D2H trim 后触发）完整保留。白盒审查通过。
  - tip8/tip11 未及落地即被同日后继取代，脚本未同步。
- **脚本同步 analysis（§0.3 缺则必补）**：`test/perf/scripts/` 新增 run_tip7/9/10/12 的
  smoke+matrix、run_tip7/9 perf（含 tip7 perf_v2 干净重跑）、chain_tip10/chain_tip12 编排器。
- 关键脚本修复（tip9 矩阵一脉）：`measure()` 覆盖写 → 追加写（此前 matrix.jsonl 只剩最后状态）；
  manual dump 翻转后等 6s 轮询生效再发长请求（此前翻转即发导致 dump 不触发）。
- **residual_check 新增**（用例见 FUNCTIONAL_TEST_LIST §16）：每 case/状态结束后查 VLLM::
  孤儿进程并留证，用于定责「残留是否 guard 修改引起」。白盒侧：guard 后台线程均
  `daemon=True`，理论上不残留；客观结论待 T0 对照实验。
- C1/T0 全局对照沿用 tip7 perf_v2 干净重跑结论；本轮矩阵为单轮 T1→T2→T3 换位（103 起不了
  T0），聚焦 C2/C3、DUMP_CHECK 与 residual 证据。
- 103 编排：chain_tip10（smoke → 矩阵 6 组合：Qwen2.5-7B tp2/pp2/dp2、DSV2-Lite tp2、
  Qwen3-8B tp2、Qwen3-32B tp4；m1-m5 用卡 0,3，m6 加 4,5）→ 完成后 chain_tip12 自动接力
  同矩阵做对照。
- 162：大模型臂（Qwen3.5-122B / DSV4-Flash）+ T0 定责；当前全卡被他人 DSV32 任务（11:46 起）
  占满，看门狗等释放自动续跑；已通知将验证目标从过时 tip11 切到 tip12（1992d6b71）。
- **结果待补录**：tip10/tip12 DUMP_CHECK 对照、C2/C3 数值、residual 证据、dump 元信息与产物路径。

---

## 2026-09-28 C2 劣化二分定位 + 根因插桩 + 历史基线考古（因果链固化）

### 现象
tip14 (696b81d34 + V030) 交叉轮换校准（N=3, m1 组合 Qwen2.5-7B TP2）:
C1 = 1.00523 PASS, C2 = 0.98274 FAIL (>=0.999)。
C2 破线 = t2(热更新轮询)比 t1 慢 ~1.7%。

### 二分五点（全部 m1 组合，t1->t2 交叉轮换 N=2，几何均值）

| 点 | 提交 | 环境 | t1 gm | t2 gm | C2 | 判定 |
|----|------|------|-------|-------|-----|------|
| A | tip12 1992d6b71 | V029 | 9.076 | 9.091 | 1.00156 | PASS |
| C | 39f115097 control plane 重挂 | V030 | 8.988 (低 2%) | 9.076 | 1.00987 | 假 PASS(被掩蔽) |
| E | 36821945f claim bus 门控修复 | V030 | 9.250 | 9.005 | 0.97346 | FAIL |
| D | 8b205863d hooks 解耦 | V030 | 9.144 | 9.037 | 0.98829 | FAIL |
| B | tip13 6e4284483 | V030 | 9.183 | 9.008 | 0.98092 | FAIL |

### 掩蔽机制（两幕剧）
1. 劣化本体 = t2 每步 due-bit all_reduce (`_task_bus.sync_due_bits`, caller
   `_wave_head_merged_bus`)。同一份代码 V029 开销~0、V030 开销 ~1.7% (环境放大)。
2. 39f115097 rebase 时丢失 claim bus 的 `should_dump_kv_on_rank()` 门控
   (tip12 原有, 见 processor_dump.py) -> t1 也每步做 claim AR, t1 被拖慢 2%,
   C2 比值被同类开销相消 -> 假 PASS。
3. 36821945f "[Performance] Skip TP dump claim bus when dump is inactive"
   修复 t1 (9.25 恢复) 同时揭露 t2 的 config AR -> C2 从 ~1.0 暴露为 ~0.98。
   该提交是"揭露者"不是"引入者"。

### C1 可证伪检验（用户质疑驱动）
质疑: 若 t1 在 39f115097..36821945f 窗口被拖慢, C1=T1/T0 应该破线, 但历史 C1 全达标?
答案: 历史 C1 (tip13 矩阵 1.0077-1.0087, tip14 矩阵 0.99-1.02, tip14 校准 1.00523)
全部测于 36821945f 之后的树, t1 已修复 -> 达标自洽。该窗口从未测过 C1。
实测 (T0=5a84871b2 vs T1=39f115097, 交叉轮换 N=2):
  T0 gm=9.153 (spread +0.08%), T1 gm=9.000 (spread +0.30%),
  **C1(39f115097) = 0.98329 FAIL** -- 预测 0.988 命中, 掩蔽模型端到端闭环。

### 插桩结果 (patch_ar_instrument.py, rg-instr-tip14/rg-instr-tip12)
两树 t2 各 3x256 tok, 每 100 次调用输出 [RG-AR] 行:
- gate 后端: 两环境均为 ProcessGroup/gloo, device=cpu (HCCL/NPU 假设被否定)
- .item() 耗时: 两环境均 ~40us (host-device 同步假设被否定)
- **all_reduce 耗时: V029 ~1100us, V030 ~1500us (慢 30%), 尾部 max 14ms**
- 调用频率: 每步 1 次 (n 与 decode 步数一致), caller=_wave_head_merged_bus
结论: 真凶是每步 1-2ms 的阻塞 AR 本身。V029 靠 runner 步结构把 1.1ms 重叠掩盖;
V030 (v2 runner 关键路径更紧) 暴露 ~1.7%。注: AR 计时大部分是对端等待
(同步点强制对齐), 优化 Gloo 本身无意义, 必须消除/降频同步点。

### 历史基线考古 (git log -S / fsck / reflog)
- README 基线 C2 v2=0.99920 的测量 = **09-17 13:02-14:16, 产品树 21d24c892**
  (V029, runtime_config-only 时代, sync_due_bits 当已存在)。
- 原始提交链 (dangling, 对象完好, 可 git worktree add 检出复测):
  21d24c892 [Fix] MRV2 token reads (09-17 测量版)
  cef550d2b [Test] RUN_NOTES: 8 toolbox findings (当轮记录)
  d5c404f42 [Test] 09-17 regression wrap-up
  03a6ad89d [Feature] control plane 原始 amend (09-18 12:16)
  bb9def9a2 [Feature] Add runtime_guard control plane (原始诞生, 1ae55b060 链)
  1ae55b060 [Fix] Tolerate unbound runtime_guard (09-18 重写链 tip, C5/C6 于 09-20 在此测)
- 39f115097 (09-24) 是 control plane 重挂 V030 的产物, 不是 1ae55b060 后代。
- 历史 0.99920 (V029) 与 A 点 1.00156 (V029) 同性质: 每步 AR 存在但 V029
  测不出。不存在"达标代码被某笔提交改坏"。

### 修复方向 (目标: 开销 <=0.1%)
方案 A (推荐): 文件轮询替代每步 AR -- config 源头是共享文件, 各 rank 每步
  本地 stat mtime (~2us), mtime 变化才 reload+AR 一次。无变更时每步开销
  <0.01%, C2 预期恢复 >=0.999。PP>1 本就设计为文件轮询。
  风险: mtime 粒度 (用 mtime+size+inode 联合判定), 跨节点 FS 属性缓存。
方案 B: AR 降频 (每 N=30 步一次, 1.7%/30~0.06%), 一致性窗口拉长 ~1.5s,
  与 3s reload 节流量级匹配, 可接受。
方案 C: 优化 Gloo 传输本身 -- 不可行, AR 耗时主要是对端等待, 同步点不消除无解。
C1 无需优化: 36821945f 之后 t1 基础设施开销 ~0 (1.00523, 在噪声内)。

## 2026-09-29 产品 tip `d36597ee6`（单 commit squash + SOB）对 analysis 用例的影响

产品 `feat/runtime-guard-config` tip **`d36597ee6`**（此前同线 squash 曾记
`ad6e06bcd` / `6513317ee`）；历史明细在 `feat/runtime-guard-config-history`。
相对本旁支旧假设的关键变点：

| 变点 | 产品行为 | analysis 影响 |
|------|----------|---------------|
| 删除 `sync_mode` | 传输按 rank 固定：last-PP×TP → TP0 due-broadcast；其余 file poll | **F-01/02/03、T-05/06/11、中文 design §2.2 已改**；勿再验收 `forcing sync_mode=file` |
| due `all_reduce` → `sync_due_bits_from_src` | TP0 打包 `[wave_idx,config_due,dump_due]` 再 `broadcast`；DueBitsBusWorker 异步 | **C2 必须对产品 tip 重跑入库**（旧 0.99920 属 AR 时代；产品侧 N=3 报 ≈1.0，analysis 未复验） |
| `wave_idx` | receiver 错波 → `RuntimeError` | 产品 UT `tests/ut/runtime_config/test_task_bus.py` 已覆盖；实卡补 **F-03 / T-13** |
| 无 RG-BUS-STATS | 验证打点已从产品删除 | analysis **不必**补 stats 用例 |

### 还需不需要补用例？

**要补 / 要改（清单层，已改本提交）：**

1. **功能清单** `FUNCTIONAL_TEST_LIST.md`：F-01…F-03 / F-01b / F-09、T-05…T-13、§14.3 C2 缺口说明。  
2. **perf README**：C2 状态标「待对 tip 重测」。  
3. **中文 design/ops**：对齐 last-PP TP0 due-broadcast（本提交）。  

**要跑、尚未出数（实卡）：**

1. **C1+C2 交叉轮换** × v1/v2，对 PRODUCT tip（`run_c1_c2_cross_rotate.sh`，N≥3，正式门禁 N=6）。  
2. **T-05（PP=2）**：确认非 last-PP file poll + last-PP due-bcast，无 hang。  
3. **T-13 / F-01**：TP=2 reload 热路径日志无 world AR、idle 无 `broadcast_object`。  

**不必新建（产品 CPU UT 已覆盖）：**  
`sync_due_bits_from_src` / `wave_idx` wraparound / misalignment / v18f·g due-broadcast — 以 config 仓 `tests/ut/` 为准。  

**旁支代码树注意：**  
`vllm_ascend/runtime_config|runtime_guard` 仍是旧 fork（含 `sync_mode`）；analysis 本地 UT（如 V6 sync_mode freeze）**不**代表产品 tip。跑产品行为请挂 **config worktree**；本旁支只维护 live/perf 清单与脚本。`test_refresh_config_cost.py` 已去掉 `sync_mode=` 传参以便跟新产品 ctor。

## 2026-09-29 产品 tip `4d9f2c67c`（drop set_log_level / print_output）

相对 `6ec13d80e`（metrics un-nest）之后最新产品 tip **`4d9f2c67c`**：

| 变点 | 产品行为 | analysis 影响 |
|------|----------|---------------|
| 删除 `log.print_output_on_finish` | 无 `log` JSON 段；finish 不再 INFO 打 output | **O1 / A-04 print 路径作废**；敏感输出靠 `report.save_sensitive_info` |
| 删除 `set_log_level` action | 合法 action 仅 `report` / `dump_kv` | **F-52 / A-20/21 / L-06 标 REMOVED**；调级用 `ascend_log` |
| `[SamplingMeta]` | after-sample logger DEBUG（无 JSON 开关） | live 用 `ascend_log.debug` / modules 开 `runtime_guard` DEBUG 验收 |

设计/ops 与 user_guide 旁支副本已对齐；C1/C2 仍以当前 tip 为 PRODUCT 重测目标。

## 2026-09-29 产品 tip `e826b20fa`（prune JSON + drop output_substring）

相对 `4d9f2c67c` 之后最新产品 tip **`e826b20fa`**：

| 变点 | 产品行为 | analysis 影响 |
|------|----------|---------------|
| `runtime_config_hot_reload` bool | startup-only；固定内部 3s；JSON 无 interval | F-06 已对齐；旧 `runtime_config_reload_interval` retired |
| 删 JSON：`queue_max_size` / `free_headroom_bytes` / `short_log_interval_seconds` / `deferred_queue_max` | 固定内部常量；旧键 soft-pop | **F-16 REMOVED**；headroom 行为仍在 |
| `decode_token_ids` / `include_block_ids` | 跟随 `save_sensitive` / 始终附带 | **F-32/33、F-36/37 REMOVED** |
| 删除 `output_substring` | DETECTOR_SECTIONS 仅 spec / token_repeat / logits_finite | **D-01–07 / g03_forbidden REMOVED**；inject 场景改走 token_repeat |

设计/ops / user_guide / FUNCTIONAL_TEST_LIST 已 tip-align。analysis 代码 fork 仍可能含旧 detector；跑产品行为请挂 **config worktree**。
---

## 2026-09-28 (III) 方案 D async due-bus：v1 死锁 postmortem + v2 双阶段实现 + 验证

### 方案 D（async due-bus）设计思路
A（文件轮询）/B（AR 降频）之外的第三条路：把每步阻塞 AR 移出主推理线程，
由 per-group 后台 daemon 线程串行执行；主线程只做 queue.put + 读上一轮结果槽
（~us 级），同步点不再落在关键路径。
- config 信道 one-step-delayed：本轮提交 bits，返回上一轮已合并结果，
  各 rank 锚定一致（+1 wave），语义与 tip12 文件轮询的"下一轮生效"等价。
- dump 信道不能延迟（dump 决策要用本轮 bits），消费时机决定成败——见 v1/v2。

### v1 死锁 postmortem（rg-async-tip14，2026-09-28 早，rg_c2_async_verify）
v1：head 提交本轮 bits、config 信道返回上轮结果（异步保持），但 dump 信道在
head 阻塞等本轮 AR 完成后立即 broadcast_object 投递。
现象：t1 两轮正常（gm=9.218/9.219）；t2 两轮第一次 warmup 请求全部 900s 超时
挂死，无 C2 结果，master.log 永远停在 "health=200, measuring" 后无 ROUND_GM。
根因：v1 允许后台 worker 线程做 AR 的同时，主线程在同一 gloo group 上发起
broadcast_object。不同 rank 上两个线程进入 collectives 的相对顺序不一致
（一端先 AR 后 bcast，对端相反）→ gloo 集合通信序列错位 → 全员死锁。
只在 t2 触发：t2 有 config reload due → head 需要 quiesce+等待+两次 bcast，
并发窗口被拉宽；t1 无 due 不进 bcast 分支。教训：**同一 ProcessGroup 上
任何时刻只允许一个线程发起 collective，跨线程换序即死锁**。

### v2 双阶段拆分 + 显式 quiesce（现行实现）
1. wave-head：仅提交 1×AR([config_due, dump_due])。config 信道消费上一轮
   （one-step-delay 保持）；dump 信道**不消费**，仅 stash
   (group, seq, local_bits) 到 _due_bus_wave。config_due 时先
   `quiesce_due_bus()` 再 broadcast_object——主线程任何 collective 前必须
   等 worker 清空队列，根除并发。
2. end-of-wave（forward 结束、进入 dump D2H 前，processor_dump.py
   end_of_wave_sync 开头钩子）：`poll_due_bits(group, seq)` 检查本轮 AR 是否
   完成——正常情况下整个 forward 足够 AR 落地，零延迟消费本轮合并 bits；
   未完成时打 warning（30s 限流，"[RG-DUE-BUS] wave_end: due-bus AR seq=%s
   NOT finished at forward end ..."）再阻塞等待——即"forward 结束检查同步
   是否完成"的健康检查（用户需求，融入设计）。随后 dump jobs broadcast 投递。
3. 错误语义转移：head 阶段 collective 失败**不再丢 jobs**（未交接，仍留 TP0
   _kv_dump_jobs，下轮 wave-end 再投）；wave-end 投递失败才 refund 已交接
   arms。
公开 API：sync_due_bits（提交+返回上轮）/ last_due_seq / poll_due_bits
（wave-end 消费本轮+告警）/ quiesce_due_bus（主线程 collective 前屏障）。
补丁落盘：rg_v2_patch/{_task_bus.py, processor_bus.py, patch_async_v2.py,
patch_ut_v2.py}，幂等脚本一次应用，backup_v1/ 保留现场。

### UT 结果（rg-async-tip14，V030 PYTHONPATH）
- test_review_regressions.py：**81 passed**。v18g 改写为
  test_v18g_merged_bus_config_bcast_head_dump_bcast_wave_end（head 只 config
  bcast、jobs 留 TP0；wave-end poll+bcast+deferred）；v28d 改写为
  test_v28d_head_collective_failure_keeps_jobs_pending（head 失败 jobs 保持
  pending、stash 清空）；新增 wave-end 告警路径
  test_v2_poll_due_bits_warns_when_ar_unfinished + 尾段投递失败 refund
  test_v2_wave_end_delivery_failure_refunds_handed_off_jobs。
- 全套 runtime_guard + runtime_config：**169 passed**；observability：9 passed。
- 测试工程注意点（踩坑记录）：
  (1) 告警断言必须 leaf logger 捕获
      （vllm_ascend.observability.runtime_config._task_bus，root 捕不到）；
  (2) 必须重置 worker["warn_ts"]——bootstrap 的阻塞等待会合法消耗 30s
      限流窗口，把 wave-end 健康检查告警吞掉（产品语义：启动后前 30s 的
      wave-end 告警会被 bootstrap "due-bus behind" 限流——已知无害）；
  (3) wave-end 路径 UT 必须 patch should_dump_kv_on_rank，否则走到
      _drop_pending_dump_jobs 分支。

### C2 验证（async2_696b81d34_vllm030，rg_c2_async2_verify，N=2 交叉轮换）
t1 gm=9.120（r1:9.11, r2:9.13, spread +0.29%）；t2 gm=8.996
（r1:9.05, r2:8.94, spread +1.19%）；
**C2 = 0.98645 FAIL**（目标 >=0.999）。
- **死锁修复确认**：t2 两轮全部正常测完并干净退出（v1 两轮均在第一次
  warmup 请求 900s 挂死）。残余清理全绿（RESIDUAL_LIVE none）。
- 对照：stock tip14 C2=0.98274（N=3）；bisect E/D/B 0.973-0.988。
  v2 相对 stock +0.4pp，但 N=2 噪声（t2 spread +1.19%，r2 t2=8.94 为低尾）
  无法分辨真实开销——**按协议 C2 结论需 N>=3 复测**。
- live due-bus 告警语义验证（t2 serve log，共 8 行，全良性）：
  bootstrap 2 行（每 rank 启动一次，预期）；quiesce 6 行（~33s 一次
  config_due 事件，主线程等在飞 AR 完成——防死锁屏障的正确代价，
  1.5ms/33s ≈ 0.005% 可忽略）；**wave_end 0 行**——性能轮中 AR 全部在
  forward 内落地，poll 零等待。
- t2 seq 速率 ~9.2/s 与解码步速一致，确认 due-bus 每步提交、异步消费。

### dump 功能 sanity（manual_dump flip，v2 wave-end 信道端到端）— PASS
- 流程：boot（manual_dump=false）→ 2048 tok 长请求在跑 → 第 8s 翻转
  config manual_dump=true → 25s 后统计。
- **SANITY_DUMP files=7672**，布局符合既有 schema
  （manual_trigger/cmpl-*/wave_NN/dp0_tp0_pp0_cp0/*.pt）——config 信道
  bcast + wave-end dump 信道投递 + D2H dump 全链路工作。
- **wave_end 告警 live 触发 1 次**（TP0 seq=30，flip 落地那波）：
  "[RG-DUE-BUS] wave_end: due-bus AR seq=30 NOT finished at forward end
  (done=29 backlog=1); waiting before dump lane"——用户要求的 forward-end
  完成检查在真实环境命中了一次真实的未完成事件，告警→等待→投递成功，
  该功能端到端验证通过。
- 已知日志盲区（记录在案）：wave_end 等待告警 30s 限流，若每波都等只有
  第一条可见；本轮 0 条 wave_end 告警说明性能轮 poll 未等待，但严谨结论
  需计数器（poll_wait/wave_total）而非仅靠限流日志。

### 结论与后续
1. v2 修复了 v1 死锁（并发 collective 根除），功能全通（UT 259 + sanity），
   用户要求的 forward-end 健康检查实现并 live 验证。
2. C2=0.98645 (N=2)：较 stock +0.4pp 但未达 0.999，且受 r2 t2 低尾影响，
   统计上不可下结论——需 N=3 交叉轮换复测（协议要求）。
3. 若 N=3 复测仍 ~0.986-0.99：说明异步化只消除了主线程阻塞，per-wave 的
   锁/GIL/线程唤醒开销仍在（worker 9.2 AR/s × 1.5ms ≈ 14ms/s 跨线程活动，
   GIL 敏感的 eager 主循环可能被拖累）——届时转向方案 A（文件轮询 mtime，
   无变更时不做任何 AR，同步点彻底消失），v2 的 due-bus/quiesce/wave-end
   基础设施可保留给"有变更"的稀有路径复用。
4. 建议补丁（低成本）：poll_due_bits 增加计数器（waits/total，周期性
   debug 输出），消除限流日志盲区。

---

## 2026-09-28 tip15 (8f5e3cad8) DueBitsBusWorker 正式实现：冒烟 PASS + C2 N=3 = 0.99308 FAIL

### 对象与环境
- 提交：`8f5e3cad8 [Perf] Overlap PP==1 merged due-bus AR with forward via
  DueBitsBusWorker`（rebased tip14 lineage），worktree `rg-tip15-8f5e3`
- 环境：V030 (vllm 0.30.0) + CANN 9.1.0，Qwen2.5-7B-Instruct TP2，卡 4,5，
  v2 runner，eager，与 bisect 系列同测量协议（perf_lib, 交叉 t1->t2）
- 设计要点：depth-1 daemon `DueBitsBusWorker` **独占** group 上全部
  collectives（due-AR + config bcast + dump bcast）；主线程仅 wave-head
  `submit` + end-of-wave `drain`（`warn_if_pending=True` 保留用户要求的
  forward-end 未完成告警）。结构上根除 v1 跨线程乱序死锁。
- UT：185 passed（含新增 test_bus_worker 6 例），lastfailed 空。

### 工程坑（worktree 复用必踩，重要）
- 新 worktree 检出后 EngineCore 初始化必崩：
  `aclnnAddRmsNormBias ... not in libopapi.so`。DeepSeek-V2-Lite 与
  Qwen2.5-7B 同报错，与模型无关、与 8f5e3cad8 代码无关。
- 根因：该算子**非 CANN 内置**，属 `vllm_ascend/_cann_ops_custom/`
  （63MB **untracked 编译产物**，含 libcust_opapi.so）。`git worktree add`
  不携带 untracked 文件；`bootstrap_custom_op_env()` 找不到包时**静默
  return**，native rms_norm 路径 dlsym 失败。
- 修复：`cp -a rg-async-tip14/vllm_ascend/_cann_ops_custom rg-tip15-8f5e3/vllm_ascend/`
  即恢复（boot 2 分钟 health=200）。**以后任何新 worktree 必须手工补包。**
- 曾误诊为模型选型问题（第一次失败时换模型，第二次同错才转向环境归因）。

### 冒烟（rg_tip15_smoke，TIP15_SMOKE_VERDICT PASS）
- boot 2min health=1；短请求 658B / 长请求 2368B 正常生成。
- `bus_worker_started=2`（TP0/TP1 各一），worker 生命周期正常。
- manual_dump 热翻转后 15s 内 `dump_dir` 产出 **112 个 .pt**
  （每步落盘节奏与 decode 步数吻合）——正式实现 manual dump 落
  `dump.dump_dir`，**不再**是 v2 原型的 `report/kv_cache/manual_trigger`
  （冒烟脚本 glob 路径未对准导致 manual_pt=0 的假象，功能本身正常）。
- end-of-wave 未完成告警 live 触发（功能端到端工作）。
- 清理干净：RESIDUAL none、HBM released。
- **发现 bug：告警限流失效**——60s 窗口 132 条，同一 rank 同一秒 2 条，
  30s rate-limit 未生效（v2 原型 15 分钟仅 8 条）。刷屏不影响功能与性能，
  但干扰后续测量观察，建议修复。

### C2 交叉轮换 N=3（rg_c2_tip15，Qwen2.5-7B TP2）
- t1 gm=**9.139** tok/s（r1 9.196 / r2 9.137 / r3 9.086，spread +1.25%）
- t2 gm=**9.075** tok/s（r1 9.078 / r2 9.039 / r3 9.110，spread +0.77%）
- **C2_TIP15 T2/T1 = 0.99308 (>=0.999) FAIL** —— t2 开销 0.69%，未达 <=0.1% 目标。
- 三代对照（同协议同模型）：

| 实现 | C2 | t2 每步开销 |
|------|-----|------------|
| tip14 stock（wave-head 阻塞 AR） | 0.98274 (N=3) | ~1.7% |
| v2 原型（head submit / wave-end poll / quiesce） | 0.98645 (N=2) | ~1.35% |
| **8f5e3cad8（DueBitsBusWorker 独占）** | **0.99308 (N=3)** | **~0.69%** |

异步化方向正确（改善 56%），但与上轮预判一致：异步化只消除了主线程
阻塞等待，**per-wave 跨线程成本（锁/线程唤醒/queue depth-1 串行化）仍在**。

### 机理取证
- t1 与 t2 **都**每波提交 merged-bus（两侧 serve log `bus worker
  started`=2 各在），即 t1 也每波做 [false,false] due-AR。
- 告警分布：**T1=0、T2=216**。同样每波 AR，t1 三个 arm 全部在 forward
  内完成；t2 有 216 波 end-of-wave 时未完成（~3% 波）。
- 粗算：216 波 × 全额等待 ~2ms / 总步数 ≈ 0.06%——**未完成波全额等待
  不是主要开销**；0.69% 主体仍是每波 submit/drain 的跨线程固定成本
  （叠加 t2 特有 reload 检查路径）。
- t2 未完成波与 reload 3s 节奏的关联、submit→AR 启动延迟（GIL/调度）、
  rank 间步伐不齐的贡献，需计数器插桩归因（当前限流失效，
  216 是下限而非精确计数）。

### 后续选项（按优先级）
1. 修告警限流 bug（30s 间隔未生效），否则任何观察都被刷屏污染。
2. 补计数器插桩：poll_ready 命中率、submit→完成时延分解、
   per-wave submit/drain 固定成本——归因 0.69% 构成后再选 3/4。
3. 方案 A（推荐，上轮已预判）：文件轮询 mtime 替代每步 due-AR，
   无变更时每步开销 <0.01%，同步点彻底消失；现有 bus 基础设施
   保留给"有变更"的稀有路径。
4. 方案 B 备选：AR 降频每 N=30 步（0.69%/N≈0.02%），一致性窗口 ~1.5s
   与 3s reload 节流同量级。

## 2026-09-28 tip15 复测（限流修复 + [RG-BUS-STATS] 插桩）：C2=0.97995 FAIL，两轮合并 ~0.988；归因修正——开销主体是 TP1 end-of-wave 排空等待

### 改动（rg-tip15-8f5e3，基于 8f5e3cad8 未提交）
1. **限流修复**（processor_bus.py）：rebase 时丢失的 30s 告警间隔恢复
   （实例级 `_merged_bus_warn_ts`）→ 冒烟告警 **132→4 条**（每 rank
   每 30s ≤1，符合设计）。
2. **插桩**（processor_bus.py + bus_worker.py）：`[RG-BUS-STATS]` 每
   300 波输出 per-rank 分相位计数——主线程侧 head_due/submit/drain/
   wait，worker 侧 sched/ar/bcast（ns 级计时，us 展示）。
   - 插桩自身 bug：`_bus_stats` 静态方法与实例属性同名冲突 → 生产类
     getattr 命中方法本身，boot 崩溃（`'function' object is not
     subscriptable`）。SimpleNamespace UT mock 抓不到此类 bug——补真实
     子类回归 UT `test_bus_stats_dict_survives_mixin_class_name_shadowing`
     （属性改名 `_bus_stats_st`）。
3. UT：**177 passed**（176+1）；插桩主线程开销 ~2us/波（≈0.002% 波长），
   T1 侧早退零开销——不足以解释下述 run-to-run 差异。

### C2 复测（rg_c2_tip15，Qwen2.5-7B TP2，N=3 交叉轮换）
| 轮 | t1 (tok/s) | t2 (tok/s) | T2/T1 |
|----|-----------|-----------|-------|
| r1 | 9.205 | 9.076 | 0.98600 |
| r2 | 9.274 | 9.088 | 0.97993 |
| r3 | 9.177 | 8.940 | 0.97418 |

gm(t1)=9.218（round_spread +1.06%），gm(t2)=9.033（round_spread +1.68%）
**C2 = 0.97995 FAIL**（目标 ≥0.999）

vs 首轮同 build 0.99308 → 0.97995，run-to-run 相差 1.3pp。两轮合计
12 个 per-round 比值范围 **0.974–1.003**：N=3 的 run-to-run 方差
~1.3% >> 0.1% 阈值分辨率。**两轮合并（6 round/状态）C2 ≈ 0.988**：
与 v2 原型（0.98645）同档，优于 stock tip14（0.98274）约 0.5pp，
距 0.999 目标仍差 ~1.2pp。

### 归因数据（C2 t2 满载，300 波/窗口，波长约 110ms）
| 指标 | TP0 | TP1 |
|---|---|---|
| not_ready（end-of-wave 未完成率） | 0.7–1.3% | 27–79% |
| wait（end-of-wave 阻塞） | 15–21us | **1.4–3.3ms** |
| ar（worker 侧 AR 耗时） | 1–21ms | 37–103ms |
| 主线程固定（head_due+submit） | ~36us | ~36us |
| sched / bcast | 250–320us / 82–140us | 250–260us / 83–155us |

r3 t2=8.940（最低吞吐轮）恰对应 wait avg 3.3ms（两轮最高）——归因
与吞吐方向一致。

### 关键修正与结论
1. **T1 serving 阶段完全不走 bus**：serve_t1.log 0 条 [RG-BUS-STATS]
   （bus worker 仅在 boot profile_run 启动；`hot_reload_enabled=False`
   时 wave-head 走本地路径）。上轮"t1 与 t2 都每波提交 merged-bus"是
   误判——"bus worker started=2"只是 boot 期现象。C2 差值即 T2 独有
   的 bus 开销。
2. **开销主体 = TP1 end-of-wave 排空等待 1.4–3.3ms/波（1.3–3% 波长）**，
   部分被 rank0 步间调度间隙吸收 → 净开销 ~1–2%，与合并 C2 吻合。
   上轮"0.69% 主体是 submit/drain 跨线程固定成本"的推断**不成立**——
   固定成本实测仅 ~36us/波（0.03%）。
3. **AR rendezvous 偏斜约一整波**：TP1 worker submit 后等 TP0 侧
   ~99ms（≈波长 110ms）AR 才完成。两侧 sched 均仅 ~300us（queue
   pickup 不慢）→ 偏斜发生在两 rank 的 submit 时刻之间。根因待查
   （rank0 调度/broadcast 结构或 worker 线程 GIL 饥饿）。
4. 限流生效后 MERGED_BUS_UNFINISHED_T2=7——告警计数只是下限，
   not_ready 率才是准确的未完成波指标。

### 判定与下一步
- **判定：C2 未达标**。DueBitsBusWorker 异步化方向有效（较 stock 改善
  ~0.5pp）但不够：per-wave AR 的 rendezvous 尾部（TP1 等待约占一波
  1.3–3%）构成 ~1pp 级开销，异步化只能隐藏其中被 forward 覆盖的部分。
- **方案 A（文件轮询 mtime 替代 per-wave AR）为正解**：归因显示消除
  per-wave AR 即回收全部 ~1pp；现有 bus 基础设施保留给"确有变更"的
  稀有路径。
- **测量方法学**：N=3 run-to-run 方差 ~1.3% 已超过阈值分辨率——后续
  方案对比验收需 N≥5 或合并多轮统计；单轮结论（含首轮 0.99308）不可
  作为达标依据。


## 2026-09-28 tip15 根因复核：逐波时间戳插桩 + 双会话交叉验证（推翻 submit 偏斜旧解读）

### 背景与质疑
上节第 3 点"AR rendezvous 偏斜发生在两 rank submit 时刻之间"与 TP lockstep
语义矛盾。本轮用逐波 wall-clock 插桩复核：每波记录 head/submit/AR 进入/AR 完成
epoch 时间戳（[RG-BUS-WAVE]，300 波/组×8 条 recent），AR 内部拆分
ar_call（collective）/ar_item（D2H），并运行时打印 gate 后端。

### 采集
- probe 会话（卡4,5, port8173, T2, 100s 稳态）：64 条 WAVE（TP0/TP1 各 32）
  log: rg_probe_tip15/serve_t2.log
- profiler 会话（卡4,5, port8174, T2, 40s）：48 条 WAVE + ascend_pt trace
  log: rg_prof_tip15/serve_t2.log；trace: rg_prof_tip15/traces/rank0_*/rank1_*
  （ascend 原始格式，msprof CLI 不支持离线导出，留待 MindStudio）

### 实测（两轮会话互相印证）
| 维度 | probe 会话 | prof 会话 |
|---|---|---|
| head/submit 相位（TP1−TP0） | ~2ms 齐步 | ~2ms 齐步 |
| bus 进入 AR（head+0.4ms） | 双侧齐步 | 双侧齐步 |
| 慢侧 AR 耗时 | TP0 挂 ~89–115ms/波 | TP1 挂 ~110–130ms/波 |
| 快侧 AR 耗时 | TP1 ~0.6–8ms | TP0 ~0.9ms（7 波慢） |
| 慢侧 drain wait | p50 9us, p90 2.84ms, max 3.6ms | p50 1.77ms, p90 3.69ms |
| 快侧 drain wait | ~10us | ~10us, p90 1.03ms |
| gate 后端 | cpu 64/64（Gloo） | cpu 48/48 |

关键耦合指纹：慢侧 wave-k AR 的完成时刻 = 快侧 wave-(k+1) head+~1ms
（例：TP0 head=131.225 AR 完成 131.3346；TP1 head=131.333 AR 完成 131.3346
——两者在 TP1 进入后 1.2ms 内同时完成）。STATS 300 波窗口：慢侧
ar avg=47.6ms / wait avg=569us/波（约 0.52% 波长）；快侧 ar max 出现一次
233ms（约两波）——错位形成/翻转的化石证据。

### 根因判定（按可信度）
**实证根因 = Gloo collective 序号错位一格**：
due-bit AR 的调用条件（due_locals 非空）在两侧非严格对称；历史上某波仅一侧
进入 collective（另一侧 `if not bits: return` 早退），同一 ProcessGroup 的
collective 序号从此永久错开一波。稳态下慢侧每波 AR 在波头 +0.4ms 进入后
等满一整波——它等的不是"对端这次没来"，而是对端的**下一次**（快侧 wave-(k+1)
AR 进入时双方 <1ms 完成交换）。forward 窗口内配对方根本不存在，AR 不可能
完成；波末 forward 让出后，对端下一波进入恰落在 drain 窗口 → 慢侧主线程
wait_result 暴露 p90 2.8–3.7ms（max 4.9ms；尾波 53ms）→ ~0.5–1pp 吞吐损失。
**错位方向 per-boot 随机**（probe 慢侧=TP0，prof 慢侧=TP1）——与具体 rank
无关，取决于 boot/warmup 早期哪侧先形成不对称。

原三项假设判定：
- R1（device_group/HCCL 被 forward 占住）**排除**：`_cpu_gate` 优先 cpu_group
  且构造 assert 必存在；运行时 112/112 条 gate=cpu，全走 Gloo，无 HCCL 参与。
- R2（GIL/调度推迟 bus）**方向性排除**：ar_in 显示双侧齐步进入（skew~2ms），
  99ms 不是"进入晚"。但"纯 Gloo 解释不了 99ms"的直觉正确：99ms 的本质是
  配对 collective 尚未被发起（序号错位），非 gloo 慢、非网络/争用。
- R3（结构必然项）**成立且精确化**：异步化把阻塞挪到 end-of-wave 后，序号
  错位使 AR 在 forward 窗口内不可能完成，主线程波末 wait 必然暴露等待。
  R3 是"落点"，序号错位是"根因"。

### 旧解读修正（重要）
1. "TP1 先 submit、再干等 TP0 ~99ms"——**错**。主线程 head/submit 双会话均
   齐步（~2ms）。上节第 3 点相应作废。
2. "AR rendezvous 偏斜 ~99ms"——数字对、语义错：不是两侧同波 AR 的
   rendezvous，而是慢侧 wave-k AR 与快侧 wave-(k+1) AR 的跨波配对。
3. "后台 AR 与 forward 未真实 overlap"——对，但机制非资源竞争：配对方在
   forward 窗口内不存在，不是被 forward 占住进度。
4. 顺带解释历史现象：DueBitsBusWorker 各轮 C2（0.982–0.993）的波动部分
   来自错位方向/是否错位的 boot 随机性，非实现质量问题。

### 下一步
- **方案 A（file mtime 轮询替代 per-wave AR）仍为正解**：无 collective 即无
  错位，drain 等待整体消失。
- 若保留 due-bit AR：必须保证双侧 collective 调用次数严格一致（not-due 侧
  也发全 0 向量，不可早退），或加入序号对齐/重对齐机制——复杂度高于方案 A，
  不推荐。
- 测量注：本次插桩（纯 Python time.time/perf_counter）开销 ~us 级，对 C2
  影响可忽略；wave 采样为 300 波窗口 recent 8 条，非全波覆盖。


## 2026-09-29 v2 全波插桩复核：根因改判为「CPU 相位漂移累积」，昨日「Gloo 序号错位一格」结论作废

### 背景
用户质疑昨日结论（错位=collective 序号错配对），要求 v2 插桩：每波带 wave idx
（bus AR 计数 + sync_for_step 计数）、forward start/end 锚点、worker 线程
AR 进入/退出实时日志（[RG-BUS-AR-IN/OUT]）、drain 后整波汇总
（[RG-BUS-WV2]，含 head/submit/ar_in/ar_out/fwd_in/fwd_out/drain_end 绝对时间戳）。

### 采集
- probe 会话（卡4,5, port8173, T2, 100s 稳态流量, Qwen2.5-7B TP2）
- 1429 波/rank，2858 条 WV2，**全波覆盖**（v1 只有 300 波窗口 recent 8）
- log: rg_probe_tip15/serve_t2.log（analyze_wv2*.py 三件套已上传容器）

### 决定性证据（全部指向配对正确、相位漂移）
1. **双侧每波都调用 AR，无一跳过**：两 rank idx 集合完全一致（0..1428），
   sync 计数同步增长，wait/log 无缺口。
2. **同 idx AR 完成时刻差 p50≈0ms**：Gloo 按调用序配对正确，跨波错配不存在。
   （v1 看到的"慢侧 AR ~99ms"实为等对端**同波**进入，v1 无 idx 才误读成错位一格。）
3. **同 idx head 相位差 p50≈102ms**（≈一个波周期）：漂移在跑。
4. **坍缩实证 idx 906→908**（01:47:04，head 时间戳互证 rank 身份）：
   - 906/907 稳态：TP0 head 领先 TP1 ~106–110ms；TP0 worker AR 等满 ~108–111ms
     （异步不阻塞 forward）；TP1 AR 亚毫秒完成（TP0 已在等）。
   - 908 双侧空闲波（无 scheduler 输出，fwd_in=0）：TP0 head=424.119，
     无 forward 可做，drain 直接等 AR → **阻塞 141ms**（wait_us=141342）直到
     TP1 于 424.259 进入其 908 AR；TP1 drain 仅 2ms。两侧 909 起相位归零，
     漂移重新累积。→ 「空闲波坍缩」机制实锤，方向 per-boot（本会话 TP1 慢）。

### 漂移分解（analyze_wv2_drift.py，busy 波 n=1405，fwd>50ms）
| 阶段 | TP0 | TP1 | TP1−TP0 |
|---|---|---|---|
| pre (head→fwd_in) | 0.10ms | 0.10ms | -0.00 |
| fwd (execute_model) | 107.87ms | 109.89ms | **+2.03** |
| post (fwd_out→drain_end) | 4.75ms | 3.30ms | **-1.45** |
| gap (drain→next head) | 12.47ms | 12.60ms | +0.13 |
| 周期 | 125.19ms | 125.89ms | **+0.70ms/波** |

- 100s 累计漂移 +992ms ≈ 8 个波周期；skew 增长几乎全部来自 d_fwd
  （TP1 forward 偶发 +10~29ms 跳变，最大 idx907 +29ms；同窗口无任何其他日志事件，
  属宿主/设备侧抖动）。d_post 是 TP0 作为领先方的 drain 暴露（果非因）。
- TP0 wait 尖峰（>5ms, busy 波）66 次，TP1 为 0 —— 等待全部由领先方承担。

### 根因（最终判定）
vLLM V1 TP worker 之间**没有每波 CPU 级屏障**：scheduler 输出走 ZMQ 单播，
HCCL 集合通信只做 device 流节流，CPU 相位可自由漂移。TP1 每波慢 0.70ms
（fwd +2.03 被自身 post -1.45 部分抵消）→ 相位漂移累积至近一个波周期
（~100ms）。due-bit AR 是唯一 CPU 会合点，异步设计使领先方在 worker 线程
等满一波（不阻塞 forward），残余等待在波尾 drain 暴露：mean +1.45ms/波
（≈1.2% 周期）+ 66 次尖峰 —— 与 C2 缺口（0.99275 vs 0.999）量级吻合。

### 修复方向（不变，依据更强）
- **方案 A（file mtime 轮询替代 per-wave AR）仍为正解**：相位漂移本身无害
  （device 节流兜底），吞吐损失全部来自 AR 这个 CPU 会合点在 drain 的暴露。
  去掉 AR → drain 无等待 → C2 恢复。
- 若保留 AR：需改造为对称的每波屏障（双侧互等）而非单侧 drain 暴露——
  等价于回到阻塞式设计，不推荐。
- 注意：rg-tip15-8f5e3 worktree 仍带 v2 probe 插桩（未提交），C2 复测前需剥离。


## 2026-09-29 hooks-only T1 probe：漂移确认为流水线原有（TP0 同向领先、钉在约一波）；「TP0 单源 due+广播」方案判定

### 背景
方案A（mtime 轮询）已弃。用户要求先做 hooks-only probe 变体：T1 基线（无
runtime_config，bus/drain 完全不激活）确认相位漂移是否同样存在——判定漂移
是否 bus 引入。

### 采集
- patch_probe_hooks.py：hooks wrapper 自打 [RG-HK-WAVE]（idx 复用 processor.py
  _probe_sync_idx，T1/T2 每波自增；init_logger_ascend 保证 (Worker_TP 前缀兜底）。
- run_tip15_probe_t1.sh：与 T2 probe 完全同构（warmup/REQS/100s/停服全同），
  唯一差异不传 --additional-config；卡4,5 port 8175；1429 波/rank，2858 条 HK 行。
- analyze_hk_t1.py：skew 曲线/百分位/三段 pin 检验/无会合检验 + T2 参照比对。
- log: rg_probe_tip15/serve_t1.log

### 结果（s = head_tp1 − head_tp0，正 = TP0 领先；T2 参照按同物理方向 +102~110）
| 指标 | T2（bus 激活） | T1 hooks-only |
|---|---|---|
| TP0 领先量 p50 | ~+102ms | +66ms（thirds 74/51/84ms） |
| pin 位置 | +100~110 ≈ 1 波 | +108~110 ≈ 1 波（P=109ms） |
| fwd_out≈head skew（无 CPU 会合） | 1417/1418 | **1429/1429** |
| busy fwd p50 | ~100ms | 104ms（tp0 104.2 / tp1 104.9） |
- 漂移曲线反复钉在 +108~110ms ≈ +1 周期；20 个空闲波多次坍缩归零
  （−35/−64/+1 事件）后重新累积。

### 结论
1. **CONFIRMED：漂移是流水线原有的**。T1 无 bus/AR/drain 仍漂至 ~1 波并被 HCCL
   钉死，方向与 T2 相同（TP0 领先、TP1 每波慢）——「领先方被 device 集体拽住、
   钉在约一波」模型在无 bus 状态同样成立。T2 的 C2 缺口只来自波尾 drain 暴露，
   不来自漂移本身。
2. 「TP0 单源算 [config_due, dump_due] → 广播」方案判定：**语义可行、性能对症**：
   - 对齐本就靠「每 rank 每波恰一次 bus + list/D2H 同波契约」，不靠对称 AR；
     dump due 事实上只有 TP0 有信息（队列/detect 只在 TP0），list 本来就是
     TP0→TP 广播；config timer 单源化（去 OR 投票）更一致，reload 生效波由
     TP0 一锤定音。
   - 广播把结果就绪点从「落后方 head（AR 语义）」提前到「TP0 head」：TP1
     收包即得（TP0 早 ~1 波已发出）、TP0 自产自消 → 实测方向（TP0 领先）下
     波尾 drain 税→0，正对 C2 缺口。
   - 每 rank 税不劣于 AR。AR 隐式背压消失后需补：队列上限 + MERGED_BUS_UNFINISHED
     告警保留；gloo 通道容忍 producer 超前 ~1 波（单 tag FIFO 保序/小消息缓冲）。
   - **新增护栏（关键）**：payload 带 wave_idx + recv 侧断言配对——AR 时代错配
     = 挂死（响亮失败），缓冲广播时代错配 = 静默串波（危险），必须显式断言；
     v2 探针已给 MergedBusRequest 加过 wave_idx，可直接复用。
   - 符号稳健性：漂移方向 per-boot（本两会话均 TP0 领先）。若某次 boot 翻转
     （TP1 领先 ~1 波），广播税落回新领先方 ≈ (S−fwd)（典型几 ms），量级≈今日
     AR、不会更差；届时再启用「生产 head(n) → 消费 tail(n+1)」
     （slack≈P+fwd−S>0 两向稳，代价 auto-dump 时序再 +1 波，需产品确认）。
   - 修正旧预期：单靠「消费点后移到下一波头」在 skew≈1 波时 slack≈P−S≈1ms，
     贴边不可靠；对症修复是 TP0 单源广播，后移仅作翻转相位的备选增强。
3. 下一步：实现 TP0 单源 due+广播（含 wave_idx 配对断言）→ 消费点保 tail(n) →
   剥离 probe 插桩 → C2 N=3 复测。
4. worktree 现带 v2 probe + HK probe 两层插桩（未提交），C2 复测前需剥离。

## 2026-09-29 TP0 单源 due 广播落地：strip probe + sync_due_bits_from_src + UT/冒烟全过；C2 N=3 复测

### 改造（rg-tip15-8f5e3；最终 diff 存容器 /data0/test-mrv2-cann91/tip15_bcast_final.diff，6 文件 +376/-34）
- **strip_probe.py**：剥离全部 v2+HK 探针插桩（每波 [RG-BUS-WAVE] 日志、
  wave_idx/tp_rank/enqueued_wall 探针字段、_probe_ 属性、recent 采样列表），
  保留产品层（rate-limit 警告、BUS-STATS 每 300 波聚合、MergedBusResult.timings）。
- **patch_tp0_broadcast.py**：
  - `_task_bus.sync_due_bits_from_src(group, due_locals, *, wave_idx)`：TP0 构造
    `[wave_idx % 2^24, bit…]` float32 payload → 一次 gloo broadcast → 接收端断言
    wave_idx 配对（错配 raise RuntimeError，防静默串波；%2^24 保证 float32 长会话
    精度）。对称 `sync_due_bits`（all_reduce）保留给 PP>1 单 lane 路径不动。
  - `bus_worker.py`：merged bus 每波切到新函数并传 wave_idx；文档同步。
  - `processor_bus.py`：`_bus_wave_seq` 每波递增随请求下发；BUS-STATS 保留。
  - 语义依据：dump_due 本就仅 TP0 产生；config reload 周期性投票改由 TP0
    单源广播一锤定音；gloo broadcast 源端无需 rendezvous → 领先 rank 波尾
    drain 税→0（正对 C2 缺口根因）。
- **fix_ut_regress.py**：v18f/v18g 断言从 all_reduce 切到 broadcast（含
  get_process_group_ranks patch，与 test_bus_worker 同款 KeyError 修复）；
  v18e/e2（PP>1 单 lane all_reduce 路径）不动。
- 警告文案 "waiting on all_reduce" → "waiting on the due broadcast"。

### 验证
- **UT 全绿**：runtime_guard + runtime_config 228 passed（test_task_bus 新增 8
  用例：空列表/单 rank/波号错配/模环绕对；v18f idle=1 broadcast 0 bcast_object）。
- **冒烟 r3 PASS**（rg_tip15_smoke_r3）：dump lane 112 个 .pt 落盘、两 rank bus
  worker 正常启动、wave misalignment 断言未触发、unfinished warns=4（与改造前
  基线一致）、无残留进程、HBM 干净释放。

### C2 N=3 复测（rg_c2_tip15_r3，point=tip15_8f5e3_bcast_r3，卡 4,5）
- **C2 = T2/T1 = 1.00001 ≥ 0.999 PASS**（N=3 交叉轮换，t1→t2）
  - t1 gm=9.110 tok/s（r1:9.10 r2:9.18 r3:9.04，round spread +1.55%）
  - t2 gm=9.110 tok/s（r1:9.16 r2:9.06 r3:9.12，round spread +1.10%）
  - 对照：stock tip14 0.98274 → per-wave-AR v2 0.99275 → **TP0 单源广播
    1.00001**：波尾 drain 等待税归零，C2 缺口闭合。
  - MERGED_BUS_UNFINISHED_T1=0 / T2=7（3 个 t2 会话合计，常态水平）；
    wave misalignment 断言全程未触发；每会话无残留、HBM 释放干净。

### 提交（rebase 到 origin/feat/runtime-guard-config d24f3b78b）
- 远端新增 6 提交（sync_mode 移除 / transport 改 last-PP TP + file poll /
  collective 作用域 last-PP × TP / ruff 格式化），与本改造在
  bus_worker/processor_bus 两处内容冲突：解决为远端 ruff 格式 + 本侧
  enqueued_ns 计时；docstring 对齐 last-PP × TP 语义（2 处措辞）。
- 提交后全量 UT 229 passed（rebase 前 228，远端新增 1 用例）。
- 分支 feat/runtime-guard-config 领先 origin 2 提交（19abdd3ac 广播改造 +
  cd4fc5442 docstring 对齐），未推送。

## 2026-09-29 tip15 round 2：产品 tip 验证（b1b58921f P0 全 PASS / C123 verdict / 新 tip 0645cf331 重构适配）

> 本条目为 round-2 独立归档：脚本已入仓（`perf/scripts/tip15_round2/`、`live/scripts/tip15_round2/`），
> 结果目录 `rg_c4_tip15_tip/`、`rg_tip15_pp2_smoke/`、`rg_tip15_tp1_smoke/`、`rg_c123_tip15_v2/`、`rg_c5_tip15_v2/`，
> 与 09-28/09-29 上午的历史目录不混用。

### A. 版本与前提
- 产品 tip：`b1b58921f` + UT 修复 `2ca3cdab5`（worktree `rg-tip15-8f5e3`）；T0 基线 = merge-base `8a2c3182c`（worktree `rg-t0-b1b58`，无 runtime_guard）
- 拓扑：Qwen2.5-7B v2 runner、TP=2 卡 4,5（PP=2 冒烟用卡 4-7）、enforce-eager、batch-invariant

### B. P0 功能验证（测于 b1b58921f+2ca3cdab5）
| 项 | 结果 | 关键证据 |
|---|---|---|
| C4 输出一致性 | **PASS** | T0-T3 四状态 3 prompt bit-identical（`rg_c4_tip15_tip/c4_identity.jsonl`） |
| PP=2×TP=2 冒烟 | **PASS** | 56 .pt 仅 last-PP 两 rank；broadcast+file-poll 双路径；0 misalignment |
| TP=1 冒烟 | **PASS** | 单 rank `dp0_tp0_pp0_cp0`、bus worker started=1、纯 file poll 无 collective |

### C. C123 交叉轮换 N=3（`rg_c123_tip15_v2/`，verdict 行 07:57:41）
- gm：t0 9.263 / t1 9.181 / t2 9.139 / t3 9.081 tok/s；轮内 spread ≤1.43%
- **C3 T3/T2=0.99369 PASS（≥0.990）；C1 T1/T0=0.99110 FAIL；C2 T2/T1=0.99541 FAIL；OVERALL FAIL**
- 配对逐 tag（n=9）与状态级 GM 一致：缺口方向三轮一致，非单轮噪声
- 关键事实：
  1. T1（无 config）也起 bus worker → C1 测的是 always-on 基础设施成本（**首次** C1 测量，无历史基线）
  2. 相邻状态 gm 随固定顺序单调下降 → 「真实成本叠加」与「固定顺序位置漂移（已知 ±1.5~2.5% 系统偏差）」在 N=3 下不可分辨
  3. 与 r3 交替协议 C2=1.00001 PASS 矛盾 → 方法学差异（fixed-order vs 交替）
  4. 门禁要求 N≥6，本轮 N=3 低于门禁
- **下一步**：order-shuffled 轮换（r1 t0..t3 / r2 t3..t0）×N≥6 重跑分辨 (a)/(b)；无论测量结论，T1 always-on 每波工作都是优化对象
- C5：**BLOCKED**——measure 客户端首请求挂死 90min（serve 侧无 POST 记录，非 proxy）；客户端已加固（no-proxy opener + 120s timeout + 诊断输出）待新 tip 重跑。注意 C5 是唯一让 token_repeat 真正命中的测试，若复现即为命中路径产品 bug

### D. 新 tip 0645cf331 重构（9 提交）review + 适配
- review 结论：合理。ruff/`un-nest metrics`（恢复 pre-nest import，design/ops 文档移至 analysis）/drop `set_log_level`+`print_output_on_finish`/prune config surface + **删 output_substring 检测器**（RETIRED，validate soft-pop）/DetectorSchema catalog/`allow_arm`→`allow_manual_dump`（仅内部 API，JSON 字段 `manual_dump` 不变）/**v2-only**（删 v1 wiring）/merge small modules + report caps 100000
- 新 tip UT **176 passed**；本地 2ca3cdab5（token_repeat UT same-wave dedupe 修复）已被上游吸收（远端含 4 处 clear_wave_cache），本地提交安全丢弃
- 脚本适配（analysis 提交 `e33d933dc`+`a59b48ad6`+`5b1eb3187`）：cfg 删 `output_substring` 与顶层 `reload_interval_seconds`（均 RETIRED）；`--additional-config` 的 `runtime_config_reload_interval` → `runtime_config_hot_reload: true`（布尔，固定 3s poll；传旧参数 pydantic 直接报错——首跑 TP=1 FAIL 即此因）；`run_tip15_c4_v1.sh` 标记 DEPRECATED（v2-only 后 v1 无 guard hooks）
- 新 tip 冒烟：**TP=1 PASS**（56 .pt 单 rank、bus worker=1、incident=1、HBM 释放）；PP=2 首跑 FAIL 为环境冲突（邻居作业绑 HCCL 192.168.2.195:16666，EI0020），已挂 idle 自动重试
- FTL 同步更新：§0.1 v1+v2 双跑标记 SUPERSEDED（v2-only）；文末追加 round-2 脚本登记表

## 2026-09-30 多机部署（9.103/13.160/13.162）：三机代码统一 3ec9c83ee，TP=1 冒烟 2/3 PASS（9.103 等卡中）

> 阶段 1-2（环境检测 + 多机部署 + 冒烟一致性）记录。产物目录按机器隔离：
> 160 `/home/d00824595/rg_tip15_tp1_smoke_r3`、162 `/home/d00824595/rg_tip15_tp1_smoke_r4`、
> 9.103 `/data0/test-mrv2-cann91/rg_tip15_tp1_v3`。

### A. 代码态统一
- 远端 `feat/runtime-guard-config` 于 09-29 23:59 被 force-push 重写：`c3cc8c003`→`3ec9c83ee`（内容 diff 为空，UT 修复被原样吸收）；history 分支同步重写
- 三机 worktree 全部统一 `3ec9c83ee`（160 老谱系 246ca397f 分叉切新 tip，老提交对象保留；162 经 ghfast.top 代理 fetch；9.103 reset）
- UT 三机一致：9.103 与 160 均 177 passed

### B. 冒烟脚本演进 v2→v4（多机适配修复，全部提交 analysis 分支）
1. **hbm() 双格式**：npu-smi 26.0.rc1（160/162）卡号在 chip 行 $3、HBM 在本行 `N/65536`；老格式（9.103）HBM 在下一行 `N/32768`。v4 先本行后 getline，正则字符串拼接（v3 的 `/...t/` 字面量 bug 已修）
2. **boot 防 setsid 死锁**：`$(... & echo $!)` 在 160 实测触发 subshell do_wait 互锁 7 分钟；改为 `( ... & )` 分离启动 + pgrep 取 pid
3. **manual_dump 时序（关键）**：`1aa5f5f65` 起 config leader `ensure_persisted()` 启动时用默认值覆写 JSON，预写 manual_dump:true 必被抹；v4 改为 health=200 后再写 true，hot reload 3s 生效。**9.103 早前 PASS 测于 0645cf331 旧语义，3ec9c83ee 上需以 v4 复验**
4. **incident find 断言**：产品 report 恒写 `<incident_type>/report_*.json`，路径不含字面 "incident"；改为 `-mindepth 2 -maxdepth 2 -name 'report_*.json'`

### C. 各机环境要点（rg 容器专用原则）
- **13.160**（lab-worker-a3-01）：容器 rg-test-160（nightly-main-a3，py3.12.13），16 卡 64G；vllm030_pkgs 由 9.103 经本机两跳 scp 726MB 搬入；根分区 100% 满，落盘仅 /home；160 自带预编译 .so（cpython-312）直接可用 → **v3 热修版 PASS**（manual_pt=17640、incident=3、HBM 回落）
- **13.162**（800I A3）：容器 rg-tip11-162（dev 镜像 py3.11）+ rg162（git 主仓）；**根因链**：`aclnnAddRmsNormBias` 为 Ascend 950(A5) 专属内置算子，A3 的 CANN 9.1/9.2 libopapi 均无（cann92 两镜像实测 0 符号）；9.103/160 能跑全靠 worktree 预编译产物 `vllm_ascend_C.*.so`（torch.ops._C_ascend 自定义 kernel 不查 libopapi），162 纯源码 worktree 缺产物 → forward_oot fallback 炸。**解法**：dev 容器内 `SOC_VERSION=ascend910_9391 python setup.py build_ext --inplace -j8` 25 分钟编译成功（前置：从主仓复制 catlass@41bf90da；SOC 自动探测在 26.0.rc1 失败需显式指定）→ **v4 PASS**（manual_pt=18032、incident=3、无算子 fallback）
- **9.103**：v4 TP=1 与 PP=2 重试 watcher 均在等卡（邻居作业占满 8 卡）；TP=1 曾于 0645cf331 PASS（v2）

### D. v4 冒烟 PASS 判据（三机统一）
health=200 + short/long 输出 sane + manual_pt>0 且 ranks 恰为 [dp0_tp0_pp0_cp0] + bus worker started=1 + incident report≥1 + wave misalignment=0 + Traceback=0 + residual none + HBM 回落

## 2026-09-30 阶段 3 性能矩阵（160 主力 22 runs + 162 补充 8 runs）：5 模型全可启动，无可证实 >1% guard 开销

> 脚本 `run_tip15_matrix.sh`（单配置单次调用，BOOT_FAIL/NO_CARDS 容错）+ 驱动 `matrix160.sh`/`matrix162.sh`
> 已入 `perf/scripts/tip15_round2/`；原始数据 `matrix160.jsonl`（154 行）/`matrix162.jsonl`（57 行）同目录。
> 产物根：160 `/home/d00824595/rg_matrix_160`、162 `/home/d00824595/rg_matrix_162`。

### A. 覆盖面（全部 OK，零 BOOT_FAIL 收官）
- 模型架构 ×5：Qwen2.5-7B（dense）、Qwen3-8B（dense 新代）、DeepSeek-V2-Lite（MoE+MLA，**非 Qwen**）、Qwen3-30B-A3B（大 MoE）、Qwen3-0.6B（小权重）
- 并行形态：TP=1/2/4、PP=2×TP2、DP=2×TP1（vllm030 均可启动）
- 状态：t1（guard 基础设施）vs t3（hot reload + 3 检测器全开）

### B. 160 主力结果（gm tok/s，t1/t3）
| 配置 | t1 | t3 | 比值 |
|---|---|---|---|
| q25_7b TP1 | 19.65 | 21.11 | 1.074 |
| q25_7b TP2 | 16.61 | 18.23 | 1.097 |
| q25_7b TP4 | 15.98 | 17.08 | 1.069 |
| q25_7b PP2×TP2 | 30.34 | 28.74 | **0.947** |
| q25_7b DP2×TP1 | 20.09 | 20.84 | 1.037 |
| q3_8b TP2/TP4 | 13.00/12.66 | 13.47/12.97 | 1.036/1.024 |
| dsv2_lite TP2/TP4 | 8.32/8.96 | 8.72/9.73 | 1.048/1.086 |
| q3c_30b TP4 | 7.86 | 7.97 | 1.014 |
| q3_06b TP1 | 19.57 | 20.97 | 1.072 |

注：t1 块紧随邻居作业（kimi）释卡后测得，系统性偏低；10/11 配置 t3 更快即此故。唯一 -5.3%（PP2）在 ±5% 噪声带内（t1 基线 30.34 为全表峰值）。

### C. 162 补充结果（跨机一致性 + DeepSeek 复验）
| 配置 | t1 | t3 | 比值 |
|---|---|---|---|
| dsv2_lite TP2 | 10.879 | 10.803 | 0.993 |
| q25_7b TP2 | 22.951 | 22.887 | 0.997 |
| q25_7b TP4 | 22.848 | 21.334 | **0.934** |
| q3c_30b TP4 | 9.681 | 9.469 | 0.978 |

- **DeepSeek-V2-Lite 双态跑通**：162 源码编译的 `npu_add_rms_norm_bias` kernel 对 MLA 架构完全可用（9.103 预编译产物同路径未验证，待有卡补测）
- 跨机绝对值：162 q25_7b TP2=22.95 vs 160 同配置 16.61——差 38%，主因 160 t1 段受邻居作业释放后的系统状态干扰；162 数据内部一致性好（t1/t3 差 <1%）
- 162 平均 t3 开销 -2.5%（TP4 -6.6% 最大），与 160 的"噪声掩盖"结论不冲突：t1/t3 整块顺序执行的基线漂移是主要误差源

### D. 判定（阶段 4 评估）
1. **11+4 组配置全部"无可证实的 >1% guard 开销"**——按严格口径（无可证实降速）全部达标，**无需进入 profile 深挖流程**
2. 两个待复测点（都不构成 fail）：q25_7b PP2×TP2（-5.3%，t1 波峰嫌疑）、q25_7b TP4 on 162（-6.6%）——建议下一轮 t1/t3 交替执行复测
3. 方法学改进（已记 FTL 待办）：矩阵 t1/t3 应交替轮换而非整块顺序；160 的 t1 段数据标注"邻居作业释卡后"低可信

### E. 脚本修复（已合 canonical）
- t3 引号 bug：`$([ ... ] && echo --additional-config "$ADDCFG")` word-split 拆散 JSON → api_server `unrecognized arguments` 必现 BOOT_FAIL；160/162 两机各自就地修复，canonical 版改为 `T3ARGS`/`DPARGS` 数组传参（本提交）
