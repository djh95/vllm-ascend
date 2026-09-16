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

perf T0 基线语义：**T0 = 当前产品分支去掉 runtime_guard 的基座** = config 分支与 main 的 merge-base
（派生值，随产品分支 rebase 而变，不是固定 commit；本次为 baec54437 的父提交 714dd1d1b）。
重建命令（每次重跑 C1-C6 前执行，用派生值而非写死 hash）：
  BASE=$(git -C /data0/test-mrv2-cann91/vllm-ascend merge-base origin/main feat/runtime-guard-config)
  git -C /data0/test-mrv2-cann91/vllm-ascend worktree add /data0/test-mrv2-cann91/rg-perf-t0 $BASE
然后在该树内编译出 .so（--no-build-isolation --no-deps），编完把 venv editable 指回产品树。

## 环境要点（本机，非 165）

- 源码：`/data0/test-mrv2-cann91/`；产品代码 `rg-config-review`（config 分支），
  工具/测试 `vllm-ascend`（analysis 分支）。
- python：`/opt/slime/venv/bin/python`（容器内 root）；跑服务必须 `docker exec test-mrv2-cann91`。
- 模型：功能测试用 `Qwen2.5-0.5B-Instruct`（快）。
- **必须 `export VLLM_BATCH_INVARIANT=1`**：本机 CANN 缺 `aclnnAddRmsNormBias`，
  拷入的 `vllm_ascend_C.so` 编译自更老/不同 CANN；不设则 EngineCore 起不来
  （`RuntimeError: aclnnAddRmsNormBias ... not in libopapi.so`）。该开关使
  `enable_custom_op()` 返回 False，layernorm 回退 `torch_npu.npu_add_rms_norm`。

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
