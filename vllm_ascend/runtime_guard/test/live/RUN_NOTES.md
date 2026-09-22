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
