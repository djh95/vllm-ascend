# runtime_guard 实卡功能测试清单（细粒度）

> **归属**：`feat/runtime-guard-analysis`（临时工具旁支）。  
> **不在** `feat/runtime-guard-config` 上维护（config 只保留产品代码 + CPU UT）。
>
> 分工：
> - **config** `TEST_MATRIX.md` / 产品 UT — CPU 侧；
> - **config** `test/perf/` — 吞吐对比（若仍挂在产品树）；
> - **本文档** — live NPU 功能行为（拓扑 / PD / 长 prompt / 多轮等）。
>
> 后处理脚本 UT 见 `../analysis/MATRIX.md`，与本文档分开。

功能测试回答「功能对不对」，不是「跑得快不快」。拓扑/模型覆盖：v2 优先再 v1、M1/M2/T4/T5/T6。

---

## 0. 阅读约定

- **Runner 覆盖（强制）**：每个功能用例都必须在 **两个 runner** 上跑，**v2 先再 v1**
  （`VLLM_USE_V2_MODEL_RUNNER=1` 先，然后 v1）。一条用例只有两个 runner 都过才算 done，
  结果按 runner 分列记录，禁止合并成一个数。
- **优先级**：`P0` = 每次 PR / 冒烟必跑；`P1` = 全量；`P2` = 依赖环境/空闲卡。
- **验收口径**：每条用例给「观察点」（看什么日志/HTTP body/report 字段）与「预期」
  （必须满足的行为契约）。不满足即回归。
- **模型代号**（同 TEST_MATRIX）：`M1` MoE（DSV2-Lite、Qwen3-Coder-30B-A3B）、
  `M2` DeepSeek-V4（MoE+MLA+MTP，≥4 卡）、`T4` PD 分离单节点、`T5` 多节点 PD、`T6` MTP。
- **可注入故障**：`RG_INJECT=scenario[:step][:param]`（5 场景），用于在不改源码的前提下
  强制触发 detector/action 命中路径，见 §10。

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

## 10. 故障注入（RG_INJECT，5 场景）

> 用 `RG_INJECT=scenario[:step][:param]` 强制命中，不改源码。每条都验证
> 「detector/hook 异常绝不进入 engine loop / async copy thread / sampler」（V3a–d）。

| ID | 场景 | 预期 |
|----|------|------|
| G-01 | 注入 detector 异常 | soft-fail，服务存活，不影响采样 |
| G-02 | 注入 hook 异常 | 同上 |
| G-03 | 注入 async copy 线程异常 | 同上 |
| G-04 | 注入 action 异常 | 同上 |
| G-05 | 注入 JSON 解析失败 | 保留旧配置，服务存活（I4） |

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

## 12. 与 TEST_MATRIX / perf README 的关系（避免重复造轮子）

- 本文档**不重复** TEST_MATRIX 已覆盖的 CPU 侧 UT（V2–V25、L1–L5），也不重复
  perf README 的 T0–T5/C1–C5 吞吐对比。
- 本文档新增的是 **live 功能行为**：拓扑正确性、模型覆盖、PD/混部、长 prompt/多轮、
  日志级别——这些只能起真实服务验证，属于「功能对不对」。
- 三者共同的强制约束：**v2 先再 v1**、**M1/M2/T4/T5/T6 拓扑覆盖**、**交叉轮换**
  （仅 perf 需要）。

## 13. 执行门槛（环境依赖）

| 用例组 | 需要 | 备注 |
|--------|------|------|
| §1–§4, §8–§11 | 单卡 live NPU + 1 个模型 | P0 冒烟即可 |
| §4 TP/PP/DP、§5 M1 | 多卡（≥2） | TP/PP/DP 组合需 ≥4 卡 |
| §5 M2、§6 T5 | ≥4 卡或多节点 | DeepSeek-V4 275G |
| §6 T4 | PD 分离 pool | `deepseek_v4_1p1d1node_pool_apc` |
| §7 混部 | 同节点多实例 | 可降级为并发脚本模拟 |
