# runtime_guard 实卡功能测试清单（细粒度）

> **归属**：`feat/runtime-guard-analysis`（临时工具旁支）。  
> **不在** `feat/runtime-guard-config` 上维护（config 只保留产品代码 + CPU UT）。
>
> 分工（均在本旁支，**需 NPU** 的项标出）：
> - **`test/live/`** — 实卡功能行为清单 + 启动脚本/配置（`scripts/`、`configs/`）
> - **`test/perf/`** — 吞吐/开销对比（T0–T3、C1–C6）；**强制 ModelRunner v1+v2**；脚本在 `perf/scripts/`
> - **`test/analysis/`** — 后处理脚本 CPU UT（可不需 NPU）
>
> config 侧 `TEST_MATRIX.md` 只描述产品 CPU UT；perf/live 以本旁支为准。
>
> **脚本落点（强制）**：凡清单/矩阵里提到、但仓库里**还没有**的  
> **启动脚本、执行脚本、起服封装、验收 curl/驱动、机房 configs**，最终都只补进  
> **本 analysis 旁支**（`test/live/scripts|configs`、`test/perf/scripts`），  
> **禁止**写回 `feat/runtime-guard-config`。现有占位 README 不算完成，须落到可跑文件。

功能测试回答「功能对不对」，不是「跑得快不快」。  
**强制双跑 ModelRunner v1 + v2**（先 v2 后 v1，见 §0.1）；拓扑/模型：M1/M2/T4/T5/T6。  
启动脚本与配置路径约定见 §0.3（正文脚本后续按用例补齐）。  
**磁盘 / dump 结构 / 标杆 / 注入定位** 见 §0.4、§10、§15（强制）。

**落盘约定（config 产品）**：`kv_cache/<type>/<req_id>/wave_N/<rank_tag>/*.pt` +
`request_info.json`；report 字段用 `dump_attempted` / `dump_dir` / `rank`（非旧 `dump_armed`）。

---

## 0. 阅读约定

### 0.0 优先级（P0 / P1 / P2）

| 级别 | 含义 | 何时跑 |
|------|------|--------|
| **P0** | **冒烟门禁**：最短路径证明「服务能起、guard 主路径可用、不打崩引擎」 | 每次改动后 / PR 前 / 上机第一轮；不过则不进全量 |
| **P1** | **全量功能**：字段、拓扑、模型、PD、多轮等正文用例 | P0 通过后跑；发版或大改前必齐 |
| **P2** | **环境依赖 / 长耗时**：多节点、稀缺机型、空闲卡才跑 | 有卡再补；不阻塞日常冒烟 |

下文「P0-x」指 §0.2 冒烟子集里的条目；正文各表若未标优先级，默认按章节进 **P1**（§13 环境门槛另限）。

### 0.1 ModelRunner v1 / v2（强制双跑）

runtime_guard 挂在 model runner 路径上，**v1 与 v2 都是正式验收对象**，不能只测其中一个。

| Runner | 启用方式（示意） | 顺序 |
|--------|------------------|------|
| **v2** | `VLLM_USE_V2_MODEL_RUNNER=1`（或产品等价开关） | **先跑** |
| **v1** | 关闭 / 不设上述开关（默认 v1，以当前产品为准） | **后跑** |

规则：

1. **每条功能用例 = 至少 2 次执行**（v2 一次 + v1 一次）；两个都过才算该用例 done。
2. **结果按 runner 分列记录**，禁止合成一个「过/不过」。
3. P0 冒烟子集同样双跑（见 §0.2 结果列）。
4. 启动命令 / JSON 配置见 §0.3；**脚本正文后续按用例补齐**，先留占位路径。

记录模板（建议贴到执行笔记或 CI 表）：

| 用例 ID | runner | 结果 | 日志/report 要点 | 启动脚本 |
|---------|--------|------|------------------|----------|
| P0-2 | v2 | ☐ | | `scripts/p0_02_token_repeat.sh`（待补） |
| P0-2 | v1 | ☐ | | 同上 + `VLLM_USE_V2_MODEL_RUNNER=0` |

### 0.2 P0 实卡冒烟子集（每次必跑 · 须 v1+v2）

| ID | 场景 | 预期 | 启动脚本（待补） |
|----|------|------|------------------|
| P0-1 | reload=0、detector 全关 | 输出与无 guard 一致（temp=0） | `live/scripts/p0_01_guard_off.{v2,v1}.sh` |
| P0-2 | 单卡 + `token_repeat` + report | 命中写 report；HTTP body 不变 | `live/scripts/p0_02_token_repeat.{v2,v1}.sh` |
| P0-3 | `manual_dump` | 能 dump；结构见 §15；`dump_attempted`；事后删盘 | `live/scripts/p0_03_manual_dump.{v2,v1}.sh` |
| P0-4 | TP≥2 时 detector | 仅 last-PP TP0 检测 | `live/scripts/p0_04_tp_rank_gate.{v2,v1}.sh` |
| P0-5 | `RG_INJECT=nan_logits` | detector 命中；report 对标杆；可初步定位（§10/§15） | `live/scripts/p0_05_inject_nan.{v2,v1}.sh` |
| P0-6 | async scheduling（若开） | `check_after_sample` 在 `get_output` 后仍执行 | `live/scripts/p0_06_async_after_sample.{v2,v1}.sh` |
| P0-7 | dump 结构 + `verify_request_kv` | §15.2 PASS；与标杆 schema 一致 | `live/scripts/p0_07_dump_schema.{v2,v1}.sh` |
| P0-8 | 磁盘回收 | 用例结束删 `kv_cache` 临时 dump；盘不涨满 | 嵌入各 dump 脚本 `trap`/`finally` |

### 0.3 启动 / 执行脚本与配置（全部落 analysis · 缺则必补）

**规则**：本清单任意用例（含 §14 缺口升格后的新 ID、P0–P2 全文）若尚无对应  
**启动脚本 / 执行脚本 / 配置**，一律在本旁支补齐，目录如下；**不**往 config 产品树加。

```
vllm_ascend/runtime_guard/test/live/
  FUNCTIONAL_TEST_LIST.md          # 本文件（要测什么）
  scripts/                         # 起服 + 发请求 + 验收（现在多数待补）
    p0_01_guard_off.v2.sh
    p0_01_guard_off.v1.sh
    …
  configs/                         # runtime_guard JSON / CLI 片段（现在多数待补）
    p0_02_token_repeat.json
    …

vllm_ascend/runtime_guard/test/perf/scripts/   # 性能起服与交叉轮换（同样：缺则补这里）
```

完成定义：用例 ID 有可执行入口（`.sh` 或统一 runner + `configs/<id>.json`），且 **v1/v2 都能跑**；  
仅文档占位 / README 不算 done。

每条用例脚本建议固定提供：

| 项 | 内容 |
|----|------|
| 环境 | `ASCEND_*`、卡数、`VLLM_USE_V2_MODEL_RUNNER` |
| 服务启动 | `vllm serve …` / 内部 launcher 参数 |
| guard 配置 | `configs/<case_id>.json`（或 `--additional-config`） |
| 请求 / 执行 | curl / 客户端：prompt、temp、是否 inject |
| 验收 | grep 日志字段、检查 `report_*.json` / `kv_cache/.../wave_*/` |
| runner | **同一用例两份脚本**（`.v2.sh` / `.v1.sh`），或一份脚本接受 `RUNNER=v1\|v2` |

正文各节表格后续可加「启动脚本」列；未填前以用例 ID 对应 `live/scripts/<id>.*` 为准，**缺文件 = 待办**。

### 0.4 磁盘空间与临时 dump（强制运维约定）

KV dump 体积大，**实卡测试必须管磁盘**，否则会把共享盘打满、拖垮同机其他任务。

| 规则 | 要求 |
|------|------|
| **测前** | `df -h` 看 dump 所在盘；`free_headroom` 不够则先清旧目录或换 `dump_dir` |
| **测中** | 只保留当前用例所需 wave；`auto_max_times` / quota 设小；禁止无上限连环 dump |
| **测后** | **及时删除**本次产生的 `kv_cache/**` 与调试用大 `.pt`（脚本 `trap EXIT` 清理） |
| **标杆** | 仅把**裁剪后的标杆**（小样本 / schema 快照 / 摘要 JSON）检入 `test/live/golden/` 或机房只读区；**禁止**把全量多层 KV 提交进 git |
| **失败保留** | 失败用例可暂留一份现场，标 `KEEP_DUMP=1`，排查完立刻删；超 N 天自动清 |
| **混部** | 多实例必须分 `dump_dir`，各自清理，互不踩盘 |

启动/执行脚本验收项须含：**跑完后盘占用回落**（或显式 `KEEP_DUMP` 说明）。

- **验收口径**：每条用例给「观察点」（看什么日志/HTTP body/report 字段）与「预期」
  （必须满足的行为契约）。不满足即回归。
- **模型代号**：`M1` MoE（DSV2-Lite、Qwen3-Coder-30B-A3B）、
  `M2` DeepSeek-V4（MoE+MLA+MTP，≥4 卡）、`T4` PD 分离单节点、`T5` 多节点 PD、`T6` MTP。
- **可注入故障**：`RG_INJECT=scenario[:step][:param]`，场景见 §10（与产品 `inject.py` 对齐）。

---

## 1. 配置字段级覆盖（每个字段都要证明「生效」且「默认值安全」）

> 这是「测试每个 config 字段的作用」的落地。默认 schema 见
> `vllm_ascend/runtime_config/_defaults.py`。每个字段列三条：默认值、验证方法、预期。

### 1.1 顶层字段

| ID | 字段 | 默认 | 验证方法 | 预期 |
|----|------|------|----------|------|
| F-01 | `sync_mode` | `broadcast` | 单 DP 起服务，日志看同步组选择 | `broadcast` 走 in-DP 广播组，无全量 world all_reduce |
| F-02 | `sync_mode=file` | — | 显式设 `file` | 每 rank 独立轮询配置文件，无集合通信；多 rank 各自 reload |
| F-03 | `sync_mode` PP>1 | — | PP=2 起服务（不显式设） | 自动回退 `file`，日志有 fallback 提示；无死锁 |
| F-04 | `reload_interval_seconds` | `0` | 默认不起 `--additional-config` | `refresh_config` 走 early-return，无轮询，detector 全关 |
| F-05 | `reload_interval_seconds>0` | — | 设 3，改配置内容 | 热重载在 ≤interval 内生效（detector 开关/阈值变化） |
| F-06 | `reload_interval_seconds` 非法 | — | 写 `"abc"` / `-1` | 软警告，回退默认，服务不崩 |
| F-07 | 未知顶层键 | — | 写 `{"windw": 10}`（typo） | 重载被**响亮拒绝**（V10），旧配置保留，日志报 unknown key |

### 1.2 `dump` 子块

| ID | 字段 | 默认 | 验证方法 | 预期 |
|----|------|------|----------|------|
| F-10 | `auto_max_times` | `0` | 默认 | 自动 dump 关闭，命中 detector 不落盘 |
| F-11 | `auto_max_times=N` | — | 设 N，重复触发 detector | 最多落 N 次 dump，之后 `try_consume` 耗尽不再 dump |
| F-12 | `auto_cooldown_seconds` | `300` | 触发一次后立刻再触发 | 冷却期内不再 dump；到期后可再次 dump |
| F-13 | `manual_dump` | `False` | `False` 时发 manual_trigger | 不落盘（manual 未启用） |
| F-14 | `manual_dump=True` | — | 发 manual_trigger | 落盘，scope 强制 `all_requests`（覆盖 request 默认） |
| F-15 | `dump_dir` | `None` | 不设 / 设自定义路径 | 默认 `kv_cache/`；自定义路径下文件正确写入 |
| F-16 | `free_headroom_bytes` | `5GiB` | 调大到超过磁盘空闲 | dump `prepare` 因 free_headroom 不足拒 arm（无写满盘风险） |
| F-17 | `dump` 手动/自动互斥 | — | 同时给非法组合 | 校验拒绝（V4 关联），日志报错，不 arm |

### 1.3 `ascend_log` 子块（详见 §8）

| ID | 字段 | 默认 | 验证方法 | 预期 |
|----|------|------|----------|------|
| F-20 | `level` | `INFO` | 起服务看日志量 | INFO 级生效 |
| F-21 | `level=WARNING` | — | 热重载改 WARNING | INFO 被抑制（此前 UCM 锁死 INFO，L3） |
| F-22 | `debug` | `[]` | 设 `["detector"]` | 命中模块 DEBUG 日志出现，未命中模块不出现 |
| F-23 | `modules` | `{}` | 设 `{"x": "DEBUG"}` | 分模块级别覆盖全局 level（L5） |

### 1.4 `report` 子块

| ID | 字段 | 默认 | 验证方法 | 预期 |
|----|------|------|----------|------|
| F-30 | `save_sensitive_info` | `False` | 默认触发 token_repeat | report 里 token-id 列表被 `sanitize_report_detail` 丢弃，只留 count |
| F-31 | `save_sensitive_info=True` | — | 同上 | token-id 列表保留 |
| F-32 | `decode_token_ids` | `True` | 命中 detector | report 带 `token_text` 解码文本 |
| F-33 | `decode_token_ids=False` | — | 同上 | 无 `token_text`（隐私/成本） |
| F-34 | `max_prompt_token_ids` | `1000` | 长 prompt 触发 | prompt 被截断到上限（0=不限制），不超 |
| F-35 | `max_output_token_ids` | `1000` | 长输出触发 | output 截断到上限（0=不限制） |
| F-36 | `include_block_ids` | `True` | dump 或 report | 含 `block_ids` |
| F-37 | `include_block_ids=False` | — | 同上 | 不含 `block_ids` |
| F-38 | `include_slot_mapping` | `False` | 默认 | 不含 `slot_mapping`（D2H 省） |
| F-39 | `include_slot_mapping=True` | — | 触发 dump | 含 `slot_mapping` |
| F-40 | `max_per_req` | `1` | 同 req 反复命中 | 该 req 只写 1 条 report，之后 `_stop_detect_for_req`（V15） |

### 1.5 `actions.defaults.on_trigger`

| ID | 字段 | 默认 | 验证方法 | 预期 |
|----|------|------|----------|------|
| F-50 | `on_trigger=["report"]` | 默认 | 命中 detector | 只上报不 dump |
| F-51 | `on_trigger=["report","dump_kv"]` | — | 命中 detector | report + dump 都执行，report 先于 dump（executor 排序） |
| F-52 | `on_trigger` 含 `set_log_level` | — | 命中 | sync_only 内联执行，立即调级 |

### 1.6 detector 子块（阈值/窗口级，详见 §2）

> `output_substring` / `token_repeat` / `logits_finite` / `spec_acceptance` 各自字段
> 在 §2 表内逐条列，此处不重复。

---

## 2. 检测器行为（4 个 detector，逐字段 + 边界）

> 统一前提：detector 只在 **last-PP TP0** 做检测（`anomaly_check_rank_skip_reason`）。
> 非该 rank 静默跳过，日志 `skip_reason` 可观测。

### 2.1 output_substring（风险字串）

| ID | 配置/场景 | 预期 |
|----|-----------|------|
| D-01 | pattern 为 str，命中输出 | 告警一次（该 req 不再重复告警，`_alerted`） |
| D-02 | pattern 为 `list[int]` token id，命中 | 同上 |
| D-03 | `match_prefix=True` | 只匹配输出前缀；中段出现不告警 |
| D-04 | `match_prefix=False`（默认） | 任意位置命中即告警 |
| D-05 | 解码漂移（token 前导空格） | text 回退路径仍能命中，不因再 tokenize 漂移漏报 |
| D-06 | 不命中 | 不告警，无 report |
| D-07 | `add_special_tokens=True` | pattern 编码时追加特殊 token，命中含特殊 token 序列 |

### 2.2 token_repeat（重复 token）

| ID | 配置/场景 | 预期 |
|----|-----------|------|
| D-10 | 合成重复序列 | 超过 `repeat_sum_threshold` 且过 `min_tokens` 预热后命中 |
| D-11 | 唯一 id 序列 | 不命中 |
| D-12 | `min_tokens` 未达 | 预热期内不告警 |
| D-13 | `consecutive_hits` 未连续 | 不告警（需连续命中） |
| D-14 | `ignore_token_ids` | 被忽略 token 不计入评分 |
| D-15 | 窗口大小热重载变更 | 清空 `_states`/`_alerted`（B12），不串旧状态 |
| D-16 | 多轮对话同一 req | 游标 `_consumed_len` 只消费增量，不重复评分 |

### 2.3 logits_finite（非有限 logits）

| ID | 配置/场景 | 预期 |
|----|-----------|------|
| D-20 | 注入 NaN logits | 命中，`ill_type=nan`；kind 分类为 `nan` |
| D-21 | 注入 +inf / -inf | `ill_type` 仍 `nan`，kind 分类为 `pos_inf` / `neg_inf` |
| D-22 | 正常 logits | 不命中 |
| D-23 | 无法归因到 req 的行 | 仅 warning，不误归因、不产 null-req report（V12） |
| D-24 | decode 阶段行 | 逐 req 归因（per-request） |
| D-25 | 大批量非有限 | `_deferred` 上限 256，不丢 report |

### 2.4 spec_acceptance（投机解码接受率）

| ID | 配置/场景 | 预期 |
|----|-----------|------|
| D-30 | 接受率 < `low_threshold` 且长度 < `len_low_threshold` | 告警（低接受率） |
| D-31 | 接受率 > `high_threshold` 且长度 > `len_high_threshold` | 告警（高接受率异常） |
| D-32 | 正常区间 | 不告警 |
| D-33 | 非 last-PP rank | `get_pp_group().is_last_rank` 门禁，跳过 |
| D-34 | accepted 列表短于 req_ids | 不 IndexError（B9），continue |
| D-35 | 无 MTP 时启用 | 不误报 / 优雅 no-op |

---

## 3. 动作行为（report / dump_kv / set_log_level）

### 3.1 report（异步上报）

| ID | 场景 | 预期 |
|----|------|------|
| A-01 | 命中 detector | 异步写 report，不阻塞推理线程 |
| A-02 | `dumps_report_json` 遇 np.int64/torch scalar/NaN | report 不丢，`.item()`/`.tolist()`/`repr()` 兜底（V2） |
| A-03 | `max_per_req=1` 连续命中 | 去重，只写 1 条；写满后 wave backoff（64,×2） |
| A-04 | report 不改变采样输出 | HTTP body 与无 guard 完全一致（I7/O1） |

### 3.2 dump_kv（KV 落盘）

| ID | 场景 | 预期 |
|----|------|------|
| A-10 | 空 `block_ids` | 拒绝全缓存 D2H，跳过（I6） |
| A-11 | 正常 arm | TP0 arm，下一 step last-PP 全 TP drain，文件写入 `dump_dir` |
| A-12 | scope=`request` | 只 dump armed req 的 block |
| A-13 | scope=`all_requests`（manual_trigger） | 全请求 dump |
| A-14 | 配额耗尽 | `try_consume` 返回 false，不 arm（V14） |
| A-15 | 排队失败 | `refund` 配额 + 清冷却，可重试 |
| A-16 | 跨 TP 对齐 | payload 带 `tp_rank`/`num_kv_heads`/`rank_tag`，可跨 TP 对比（V18） |
| A-17 | `free_bytes_at` 不足 | 拒绝 arm（free_headroom 保护） |

### 3.3 set_log_level（sync_only）

| ID | 场景 | 预期 |
|----|------|------|
| A-20 | 触发 | 内联同步执行，`apply_ascend_log_level` 立即生效 |
| A-21 | 非 sync_only 环境 | 不进入异步队列（sync_only 语义） |

### 3.4 ActionQueue

| ID | 场景 | 预期 |
|----|------|------|
| A-30 | 队列满 + 重任务（dump） | drop，绝不内联（V9a） |
| A-31 | 队列满 + 轻任务（report） | 内联回退执行 |
| A-32 | `stop` + 队列满 | drain + sentinel，正常退出（V9b） |

---

## 4. 并行拓扑（TP / PP / DP / CP）

> 目标：证明 runtime_guard 在任意并行组合下 rank 门禁、同步、dump 覆盖都正确。

| ID | 拓扑 | 场景 | 预期 |
|----|------|------|------|
| T-01 | TP=2 | 触发 detector | 只在 TP0 检测；TP1 静默 skip |
| T-02 | TP=2 | dump_kv | last-PP 全 TP 各落一份，`tp_rank` 正确标记 |
| T-03 | PP=2 | detector | last-PP 才检测；非 last-PP skip |
| T-04 | PP=2 | dump | 只在 last-PP dump；`pp_rank` 标记正确 |
| T-05 | PP=2 | sync_mode | 强制回退 file，无集合通信死锁 |
| T-06 | DP=2 | 配置同步 | broadcast 走 in-DP 组，不 full-world |
| T-07 | DP=2 | dump | DP 独立；每个 DP 副本独立 arm/drain |
| T-08 | DP=2 | report 去重 | 各 DP 独立 `max_per_req`，不跨 DP 去重 |
| T-09 | TP×PP×DP 组合 | 全模块 | 门禁/同步/dump/report 全符合；无 hang |
| T-10 | CP>1 | dump | `cp_rank` 正确进 `rank_tag`；`dp{D}_tp{T}_pp{P}_cp{C}` |
| T-11 | 多 DP + 多 PP | sync_mode | 永不 full-world collective（多 DP 安全） |
| T-12 | TP0 崩 / 非 TP0 触发 | rank gate | 非 leader 不重复 arm/detect |

---

## 5. 模型大小 / 架构覆盖（M1 / M2）

| ID | 模型 | 场景 | 预期 |
|----|------|------|------|
| M-01 | DSV2-Lite（M1 MoE） | 冒烟 + detector 全开 | routed-expert logits 下 detector 正常；无 OOM |
| M-02 | Qwen3-Coder-30B-A3B（M1 MoE） | 同上 | 同上（另一 MoE 结构） |
| M-03 | DeepSeek-V4（M2） | 冒烟 | 275G 权重加载；guard bind 不崩 |
| M-04 | DeepSeek-V4（M2） | C3/C5（detector 开销） | 大模型下 detector 开销仍达标 |
| M-05 | DeepSeek-V4（M2） | MLA logits | spec_acceptance 在 MLA logits 上工作 |
| M-06 | 小模型（如 0.5B/1.5B） | 冒烟 | 快速验证功能，无架构特化假设崩溃 |
| M-07 | 不同 hidden/layer 规模 | dump payload | `num_kv_heads`/shape 随模型正确，不硬编码 |

---

## 6. PD 分离（T4 / T5）

| ID | 场景 | 预期 |
|----|------|------|
| PD-01 | T4 单节点 1P1D | prefill worker 与 decode worker 均挂 hook | hook 在 P/D 两侧都不崩、不重复 |
| PD-02 | T4 | detector 命中 | P 侧与 D 侧 report 字段**完全一致**（同 schema） |
| PD-03 | T4 | dump_kv | D 侧 arm/drain 正常；P 侧无残留 |
| PD-04 | T5 多节点 P+D | `sync_mode=broadcast` | 跨节点配置同步（无 full-world 死锁） |
| PD-05 | T5 | TP0-only rank gate | 跨节点仍只在 TP0 检测 |
| PD-06 | T5 | report merge | 跨节点 report 字段一致，可聚合 |
| PD-07 | PD | 连接断开/重连 | soft-fail，服务存活 |

---

## 7. 混部（mixed deployment）

> 定义：同一节点/卡上同时跑多个服务，或多个 vllm-ascend 实例共享 NPU 与日志/目录。

| ID | 场景 | 预期 |
|----|------|------|
| H-01 | 两实例不同 `dump_dir` | dump 互不覆盖、互不串写 |
| H-02 | 两实例共享日志 | ascend_log 分模块，两实例日志可区分 |
| H-03 | 混部 + 手动 dump | 只 dump 本实例 armed 请求 |
| H-04 | 混部 + 磁盘紧张 | `free_bytes_at`/free_headroom 按实例独立判定，不互相拖垮 |
| H-05 | 混部 + 热重载 | 各实例独立 reload，互不干扰 |
| H-06 | 混部 + UCM 日志劫持 | 各实例日志都走 stdlib 树，互不劫持 |

---

## 8. 日志级别调整（ascend_log + set_log_level + UCM 绕过）

> 与 TEST_MATRIX L1–L5（CPU UT）对应，这里是 **live 侧** 验证。

| ID | 场景 | 预期 |
|----|------|------|
| L-01 | 起服务 | `vllm_ascend.*` logger 是 `logging.Logger`（非 `ucm.logger.Logger`） |
| L-02 | `once` 方法 | `info_once`/`debug_once`/`warning_once` 去重有效 |
| L-03 | 热改 `level=WARNING` | INFO 被抑制（UCM 锁死被 bypass） |
| L-04 | `debug=["detector"]` | 命中模块 DEBUG 出现，其余不出现 |
| L-05 | `modules={"x":"DEBUG"}` | 分模块调级生效 |
| L-06 | set_log_level 动作触发 | 立即调级，不等异步 |
| L-07 | 日志无 `ucm` 前缀污染 | 输出回 stdlib handler，capsys/caplog 都能抓 |

---

## 9. 请求场景：长 prompt / 多轮对话 / 长输出

| ID | 场景 | 预期 |
|----|------|------|
| R-01 | 长 prompt（> max_prompt_token_ids） | report prompt 截断到 `max_prompt_token_ids`，不 OOM、不超 |
| R-02 | 长 prompt 触发 token_repeat | 游标从 prompt 后开始，不误报 prompt 内重复 |
| R-03 | 长输出（> max_output_token_ids） | output 截断；detector 仍能命中 |
| R-04 | 多轮对话同 req_id | 增量消费，不重复评分；`append_output_ids` 去重 |
| R-05 | 多轮对话跨轮命中 detector | 每轮独立判定，`max_per_req` 跨轮去重 |
| R-06 | 多轮对话长上下文 | block_ids 跨轮正确，dump 覆盖正确 block |
| R-07 | 空 prompt / 空输出 | 优雅 no-op，不崩 |
| R-08 | prompt 与 output 混合截断 | `max_prompt_token_ids`/`max_output_token_ids` 各自独立截断 |

---

## 10. 故障注入 → 检出 → report 标杆 → 初步定位（强制闭环）

> 格式：`RG_INJECT=scenario[:step][:param]`（step 默认 5）。  
> 每条除「命中预期」外，还要求：**注入/detector 异常不得打崩 engine / async copy / sampler**。  
> **仅「服务没崩」不算过**：必须走完下面五步，且 **v1/v2 分列**。

### 10.1 闭环步骤（每个 inject 场景）

| 步 | 做什么 | 判据 |
|----|--------|------|
| 1 | 注入异常（§10.2） | 场景按表生效 |
| 2 | detector 是否抓到 | 对应 `incident_type` 有 `report_*.json`；非目标 detector 不误报 |
| 3 | report 内容是否正确 | 与 **report 标杆**（§15.3）字段/关键 detail 一致（允许 `req_id`/时间戳等浮动） |
| 4 | 能否 dump（若 `on_trigger` 含 `dump_kv`） | 有 `wave_N/<rank_tag>/*.pt`；结构见 §15.2；`verify_request_kv` PASS |
| 5 | 能否**初步定位** | 按 analysis **skill** 走完 triage→correlate→（可选）ref 对比；输出「嫌疑模块/阶段」一句话结论 |

### 10.2 注入场景表

| ID | scenario | detector | report 关键点（须进标杆） |
|----|----------|----------|---------------------------|
| G-01 | `nan_logits` | `logits_finite` | `ill_type`/kind 含 nan；req 可归因 |
| G-02 | `inf_logits` | `logits_finite` | kind 含 pos_inf/neg_inf |
| G-03 | `forbidden_substring[:…]` | `output_substring` | pattern 命中信息；output 相关字段 |
| G-04 | `token_loop[:step[:N]]` | `token_repeat` | 重复评分/窗口相关 detail |
| G-05 | `spec_all_reject` | `spec_acceptance` | 低接受率类字段（真 MTP 环境） |
| G-06 | 热重载坏 JSON | （配置） | 无新误报；旧配置保留；服务存活 |

### 10.3 初步定位与 skill（强制）

定位过程中 **必须使用**本旁支 skills（勿另起一套口头流程）：

| Skill | 路径 | 用途 |
|-------|------|------|
| `runtime-guard-investigation` | `analysis/skill/investigation/SKILL.md` | 端到端排查主流程 |
| `runtime-guard-analysis` | `analysis/skill/analysis/SKILL.md` | report/KV 离线汇总与对账 |
| `runtime-guard-ref-kv-dump` | `analysis/skill/ref-kv-dump/SKILL.md` | 标杆/ref dump + 相似度/首分歧 |
| `runtime-guard-detector-sweep` | `analysis/skill/detector-sweep/SKILL.md` | 选 detector / 注入对齐 |
| `runtime-guard-config-recommender` | `analysis/skill/config-recommender/SKILL.md` | 配置建议 |

**Skill 更新规则**：实卡跑完发现文档过时（路径、字段、步骤漏项、误导）→ **当场改对应 SKILL.md** 并随 analysis 旁支提交；  
在用例笔记里记「更新了哪些 skill、因何案例」。禁止只改个人备忘、不回写 skill。

定位最低交付物（写入执行笔记）：

1. 命中的 `incident_type` + `req_id` + report 路径  
2. dump 是否有、结构是否正常（或明确未 arm）  
3. 与标杆 report/KV 摘要对比结果（同/差在哪）  
4. 一句话初步结论（如「logits 非有限，采样前，非 KV 串扰」）  
5. 用过的 skill 名 + 是否已更新 skill  

测后按 §0.4 **删除**临时 dump（保留标杆摘要即可）。

---

## 11. 各模块「符合预期」检查清单（汇总速查）

| 模块 | 关键契约 | 对应用例 |
|------|----------|----------|
| `rank_gate` | last-PP TP0 检测；last-PP 全 TP dump；`dump_rank_tag` | T-01..T-12 |
| `wave_tracker` | stamp 生命周期；reap 时 `discard_many`，`_sample_waves` 不无限增长（V8） | 附 D-16 |
| `quota` | 原子 `try_consume`/`refund`；refund 清冷却（V14） | A-14, A-15 |
| `queue` | 满则 heavy drop / light inline；stop drain+sentinel（V9） | A-30..A-32 |
| `report` | `max_per_req` + wave backoff；`dumps_report_json` 兜底（V2/V15） | A-01..A-04 |
| `request_state` | 每 req 状态；finish 后清理；无跨 req 泄漏（V16） | R-04..R-06 |
| `io_snapshot` | 增量累计；tail-only（长输出不爆内存） | R-03 |
| `kv_cache_reader` | `_slice_blocks` 空 id 拒绝；payload 带 tp/pp/cp/num_kv_heads（V18） | A-10, A-16 |
| `manual_trigger` | rank 门禁 + `_wave_has_scheduled_tokens`；连续/递减计数 | A-13 |
| `logger` | stdlib Logger；`apply_ascend_log_level`（L1–L5） | §8 |
| `config` | 热路径 gate；unknown-key 拒绝（V10）；JSONC 注释+尾逗号（V13） | F-07, §1 |

---

## 12. 与 TEST_MATRIX / perf 的关系（避免重复造轮子）

- 本文档**不重复** config `TEST_MATRIX.md` 已覆盖的 CPU 侧 UT。
- **perf** 已迁到本旁支 `../perf/README.md`（C1–C6，**须 v1+v2**，需 NPU）；
  本文档只管「功能对不对」，perf 管「开销可不可接受」。
- 共同强制约束：**ModelRunner v1+v2 双跑（先 v2）**、**M1/M2/T4/T5/T6**；
  perf 另要求 **交叉轮换**。

## 13. 执行门槛（环境依赖）

| 用例组 | 需要 | 备注 |
|--------|------|------|
| §0.2 P0 | 单卡（P0-4 需 TP≥2）+ 1 模型；**v1 与 v2 各跑一遍** | 冒烟门禁；脚本见 §0.3 |
| §1–§4, §8–§11 | 单卡 live NPU + 1 个模型 | P0 通过后再跑；同样双 runner |
| §4 TP/PP/DP、§5 M1 | 多卡（≥2） | TP/PP/DP 组合需 ≥4 卡 |
| §5 M2、§6 T5 | ≥4 卡或多节点 | DeepSeek-V4 级 |
| §6 T4 | PD 分离 pool | 机房 PD 模板 |
| §7 混部 | 同节点多实例 | 可降级为并发脚本模拟 |
| §14 缺口项 | 视场景 | 未建用例前勿标 done |
| `../perf` C1–C6 | 实卡 NPU + 可起 T0 worktree | **分 runner 出数**；见 perf README |

## 14. 覆盖缺口（相对产品能力 / 实卡必测）

下列在正文里**缺失、过薄、或仅 skills 提及**，后续应升格为正式 ID（仍须 v1+v2）。

### 14.1 产品已有、live 未单列或过薄

| 缺口 | 说明 | 建议优先级 |
|------|------|------------|
| **T6 / 真 MTP·投机** | 有 D-30.. 与 inject `spec_all_reject`，缺「真实开启 MTP/draft」端到端 | P1 |
| **manual_trigger 控制面** | F-13/14、A-13 有，缺连续/递减计数、`_wave_has_scheduled_tokens`、非调度 wave 不 arm | P0/P1 |
| **dump 落盘契约** | 约定写在文首，缺独立用例：`wave_N/<rank_tag>/*.pt` + `request_info.json` + `dump_attempted`/`dump_dir` | P0 |
| **多请求并发** | 缺 batch/并发下 `all_requests` dump、quota 竞态、report `max_per_req` 隔离 | P1 |
| **Streaming / chat API** | 隐含 completions；缺 SSE/`/v1/chat/completions` 下 detector/report | P1 |
| **Preemption / recompute** | 抢占重算后 block_ids / dump 是否仍一致 | P1 |
| **进程生命周期** | 缺：起服中热挂载失败、推理中 SIGTERM、dump 中途杀进程的残留与 soft-fail | P1 |
| **磁盘中写失败** | 有 free_headroom 拒 arm；缺 ENOSPC/权限错误中途写失败 | P2 |
| **量化 / Graph** | 缺 W8A8 等 + logits_finite；ACL graph / 编译路径 hook 是否仍触发 | P2 |
| **EP / 大规模并行** | MoE 有 M-01/02，缺 EP 维与 rank_tag | P2 |
| **Prefix cache / APC / chunked prefill** | PD 有，缺前缀缓存命中下的 token 游标与 dump | P2 |
| **n>1 / logprobs / beam** | 多样本与 logprobs 行归因（logits_finite V12 相关） | P2 |
| **多模态** | 无 VL/音频请求路径 | P2（若产品不支持可永久剔除） |
| **实卡 → 后处理闭环** | 缺：live dump 后立刻跑 `verify_request_kv` / `correlate`（脚本在 analysis） | P1 |
| **v1↔v2 字段级对拍** | 有双跑规则，缺「同请求同配置 report schema/命中字段」对拍表 | P1 |
| **golden 标杆入库** | §15 已定义目录；`reports/` / `dump_schema/` 实体与 diff 脚本待补 | P0 |
| **注入五步闭环脚本** | §10.1 已定义；G-01.. 执行脚本与定位笔记模板待补 | P0 |

### 14.2 Skills / 文档提及、当前产品 `_defaults` 无的检测器

`slot_consistency` / `kv_slot_token` / `kv_slot_order` / `kv_state` 出现在 analysis skills 与
`summarize_reports.KNOWN_DETECTORS`，但 **config 产品 `DETECTOR_SECTIONS` 仅 4 个**。  
→ 不进 live 正式表，直到产品合入；skills 需与产品对齐，避免假覆盖。

### 14.3 Perf 侧缺口（详见 `../perf/README.md`）

| 缺口 | 说明 |
|------|------|
| C1/C2 交叉轮换正式数 | 历史 pending；**须分 v1/v2** |
| C5 dump_kv 开销 | 未建 |
| C6 leak-back 门禁 | `perf_lib` 有采样，缺正式验收条 |
| 起服脚本未入库 | `perf/scripts/` 仅占位 |
| 机房路径写死 | `perf_lib` 默认 `/data0/...` 需环境变量化（勿当通用默认） |

---

## 15. Dump 可用性 · 数据结构 · 标杆生成与对比

本节与 §0.4（磁盘）、§10（注入定位）配套；**无 dump / 结构错 / 无标杆对比 = 用例未完成**。

### 15.1 能否 dump 下来

| ID | 场景 | 预期 |
|----|------|------|
| K-01 | `manual_dump` / `manual_trigger` | `dump_attempted=true`；`dump_dir` 下有 `wave_N/<rank_tag>/` |
| K-02 | detector `on_trigger` 含 `dump_kv` | 命中后同样落盘；配额未耗尽 |
| K-03 | 配额/冷却/headroom 拒绝 | **不**落盘且日志可解释；不静默成功 |
| K-04 | 多 TP | last-PP 各 TP 有文件；`rank_tag` 含 `tp` |

### 15.2 数据结构是否正常

目录契约：

```
kv_cache/<incident_type>/<req_id>/wave_N/<rank_tag>/*.pt
kv_cache/<incident_type>/<req_id>/wave_N/request_info.json   # 若产品写出
```

每个 `.pt`（dict）至少含：`req_id`、`block_ids`、`layer`、`rank_tag`、`tensor`；  
`tensor` shape 符合 `[n_blocks, block_size, H, D]`（或产品当前约定），finite（除非用例专门注入坏 KV）。

验收命令（analysis 脚本，须进执行脚本）：

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.verify_request_kv \
  --report <report.json> --report-dir <root> --block-size <BS>
python -m vllm_ascend.runtime_guard.analysis.scripts.inspect_kv_dump --path <one.pt>
python -m vllm_ascend.runtime_guard.analysis.scripts.correlate_incident \
  --report-dir <root> --req-id <id>
```

| ID | 检查 | 预期 |
|----|------|------|
| K-10 | 目录/rank 布局 | 与文首落盘约定一致 |
| K-11 | `.pt` schema | 关键键齐全；`verify_request_kv` PASS |
| K-12 | block 容量 | `len(block_ids)*block_size >= N_tokens`（有 ids 时） |
| K-13 | 与 report 一致 | `req_id` / `block_ids` / `dump_dir` 可对上 |

### 15.3 生成标杆并对比

标杆目录（analysis 旁支，**只存小文件**）：

```
vllm_ascend/runtime_guard/test/live/golden/
  reports/          # 注入场景的 report 摘要 JSON（脱敏、去时间戳）
  dump_schema/      # 单层小 .pt 或 shape/meta JSON（非全量权重 KV）
  README.md         # 如何刷新标杆、浮动字段白名单
```

| ID | 动作 | 预期 |
|----|------|------|
| K-20 | 首次绿跑 | 生成/更新 golden（人工 review 后入库） |
| K-21 | 回归 | 新 report **关键字段** vs golden 一致；脚本 diff 或 `jq` 白名单 |
| K-22 | KV 对比 | 同请求 ref：`compare_kv_similarity` / `locate_first_divergence`；  
| | | 健康路径：与自身/ref 高余弦；注入坏路径：首分歧落在预期阶段 |
| K-23 | 清理 | 对比结束后删临时全量 dump（§0.4）；golden 保留 |

### 15.4 与「初步定位」的衔接

1. inject / 真实故障 → detector report（对 K-21）  
2. dump（对 K-01..K-13）→ skill `analysis` / `investigation`  
3. 需要因果时：`prepare_ref_inputs` → 清洁环境再 dump → `ref-kv-dump` skill 对比（对 K-22）  
4. 结论写入笔记；**更新 skill**（§10.3）；删盘  

脚本占位（缺则补 analysis）：`live/scripts/k_*_*.sh`、`live/golden/`。
