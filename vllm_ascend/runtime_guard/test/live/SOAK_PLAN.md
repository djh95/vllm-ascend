# runtime_guard 24h soak 压测计划（拓扑 × runner × 配置）

> 归属：`feat/runtime-guard-analysis`。
> 目标：长时间高并发 + 多拓扑 + 双 runner + 多配置，**累积 ≥24h**，重点暴露
> 卡死（hang）/ 崩溃（crash）/ 内存泄漏。多实例并行、时间累积计。
> **CP 为 best-effort**：起服失败只记录、不阻塞其余拓扑。

## 覆盖矩阵

| 维度 | 取值 |
|------|------|
| 拓扑 | TP=2、PP=2、DP=2、组合 TP=2+PP=2、**CP=2（best-effort）** |
| runner | v2（`VLLM_USE_V2_MODEL_RUNNER=1`）与 v1（`=0`），每条拓扑双跑 |
| 配置 | `token_repeat`、`logits_finite`、`output_substring` 各一；`token_repeat`+`dump_kv` 一；热改 `ascend_log.level` |

## 拓扑 × runner 实例

| 实例 | 拓扑 | 卡数 | runner | detector 配置 |
|------|------|------|--------|---------------|
| S1 | TP=2 | 2 | v2 | token_repeat + report |
| S2 | TP=2 | 2 | v1 | token_repeat + report |
| S3 | PP=2 | 2 | v2 | logits_finite + report |
| S4 | PP=2 | 2 | v1 | logits_finite + report |
| S5 | DP=2 | 2 | v2 | output_substring + report |
| S6 | DP=2 | 2 | v1 | output_substring + report |
| S7 | TP=2+PP=2 | 4 | v2 | token_repeat + report + dump_kv |
| S8 | TP=2+PP=2 | 4 | v1 | token_repeat + report + dump_kv |
| S9 | CP=2 | 2 | v2 | token_repeat + report（best-effort） |
| S10 | CP=2 | 2 | v1 | token_repeat + report（best-effort） |

> 每个 detector 配置的字段：`on_trigger=["report"]`（dump 实例加 `"dump_kv"` 且
> `dump.auto_max_times` 限小），`report.save_sensitive_info` 视隐私需求；均
> `reload_interval_seconds=3` 走热改路径。

## 每实例驱动（复用 `w5_soak.sh` 模式）

1. 起服（多卡：`--tensor-parallel-size` / `--pipeline-parallel-size` /
   `--data-parallel-size` / `--context-parallel-size` 对应）。
2. 8 并发 completions 循环。
3. 每 5min 热改一次：detector 阈值/窗口翻转，或 `ascend_log.level` INFO↔WARNING。
4. **看门狗**：`/health` 心跳；连续无响应 > N min → 记 `hang`；进程退出 → 记 `crash`。

## 卡分配

- 本机 910B4 可用卡 `{0,2,3,6}`（**卡 1 RDMA 端口 down 避用**；卡 4/5 现有单卡 soak；
  卡 7 他人 GLM 训练，勿动）。
- 165 Ascend910 卡 4-7（4 卡空闲）可另起一个不同模型实例。

## CP 说明（best-effort，不阻塞）

- 启动参数 `--context-parallel-size 2`（或 prefill/decode CP 分开的等价参数）。
- runtime_guard 已支持 CP：`rank_gate.runner_cp_rank`、rank_tag 含 `cpN`、dump
  分片按 `pcp_rank*dcp_size+dcp_rank` 标记。
- **若 CP 起服失败**（模型/环境/block-size 不支持，如 DCP 对多 block size 不兼容）：
  记下 serve.log 报错到 RUN_NOTES，跳过该实例，**不阻塞 TP/PP/DP 的 soak**。

## 内存泄漏门禁（soak 并行）

> 除 perf C6 短时 leak-back 外，新增长时运行内存监测：soak 期间被动采样 + 主动回落检查，
> 与压测并行执行（不占用额外卡）。

- **被动采样**：`rg_test/mem_monitor.sh` 每 10min 记录所有 vllm 进程总 host RSS + 每卡 HBM，
  追加到 `/tmp/rg_mem_monitor.log`。用于检测长时运行单调增长（泄漏）。
- **主动回落检查**：`rg_test/leakback_check.py <served> <port> [N] [idle_sec]` 对指定实例
  burst N 请求 → 采峰值 RSS → 空闲采样 → 判回落。
- **门禁**：
  1. **空闲回落**：burst 后空闲期 RSS 回落至基线 **+30MB** 内（对齐 C6 leak-back）。
  2. **长时增长**：24h 内 host RSS 相对首采样基线增长 **≤10%**（约 2GB）；HBM 应恒定（KV cache 预分配）。
- 结果并入下方 RESULT 行：`mem_rss_delta_mb` / `mem_hbm_delta_mb`。

## 记录

每实例一行：`RESULT|<实例>|<拓扑>|<runner>|<config>|health=<0/1>|hang=<0/1>|crash=<0/1>|elapsed=<s>|rss_delta_mb=<n>|hbm_delta_mb=<n>`；
结束写回 RUN_NOTES.md 与 `rg_test/results/`。
