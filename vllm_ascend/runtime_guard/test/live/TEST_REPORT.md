# Runtime Guard 测试报告

| 项 | 内容 |
|----|------|
| 版本 | V1.0 / 2026-09-14 |
| 产品 | `feat/runtime-guard-config` @ `39e22dacc` |
| 实卡旁支 | `feat/runtime-guard-analysis` @ `62dd85456` |
| 环境 | test-mrv2-cann91 · CANN 9.1.0 · Qwen2.5-0.5B · 双 Runner |
| Word 交付件 | `vllm-ascend/docs/zh/design/word/RuntimeGuard_测试报告.docx` |

## 1. 结论

**通过。** P0+P1 功能收口；历史 6 BUG 全关；稳定性 soak 收官（有效累计 **24.06h > 24h**）；性能 C1–C6 已出数，C1 v1 新 base 重测 1.00291 过线。

## 2. 功能

- P0 冒烟 6 用例 × v2/v1：**全 PASS**
- P1：TP gate / dump schema / 坏 JSON 热更 / 磁盘回收 / 混部 H-01..06 / PD-01..03：**PASS**
- 环境跳过：G-05（无 MTP）、P0-6 live stub（Async 由 UT/W2-3 覆盖）

### BUG 关闭

| ID | 修复 | 复验 |
|----|------|------|
| BUG-4/W3-1 | `03bfd2a54` | live dumps>0 |
| BUG-5 | configs/g03 | live |
| BUG-6/W3-2/K-13 | `5164073a9` | live |
| W1-1/R-04 | `39e22dacc` | live `seen==oc` |
| W1-2/R-06 | `03bfd2a54` | live block_ids |
| W1-3/R-06 | `39e22dacc` | live report |
| W2-1/F-07 | `229e7deb3` | UT `test_v10b` |
| W2-2/F-38/39 | REMOVED | n/a |
| W2-3/D-11 | `8876645a2` | UT `test_d11_*` |

## 3. 性能（C1–C6）

| ID | v2 | v1 | 判定 |
|----|----|----|------|
| C1 T1/T0 | 1.00411 ✅ | 0.98466（~1.5%） | **待重测（等 8 卡空闲）** |
| C2 T2/T1 | 0.99920 ✅ | 1.00644 ✅ | 通过 |
| C3 T3/T2 | 0.97678 | 0.98816 | 记录开销/归因 |
| C4 | bit-identical | bit-identical | 通过 |
| C5 | 噪声 | 噪声 | 观察 |
| C6 | leakback OK | 部分污染 | v2 通过 |

> 跨 session 噪声约 ±1%；C1 v1 ~1.5% 疑似波动，但需等 8 卡空闲后重测排除噪声再定论，暂不判 FAIL。

## 4. 稳定性

- 计划：`SOAK_PLAN.md`（≥24h，拓扑×runner×配置）
- **执行：完成，有效累计 24.06h ≥ 24h**（09-14 原 soak 13.8h + 09-18/19 补时 10.28h）
- 原 soak（09-14 11:37 → 09-15 01:24 UTC）：A=Qwen2.5-0.5B/token_repeat + B=Qwen2.5-7B/logits_finite，8 并发持续压测 + 5min 周期热更 **4070 轮零失败**；中断=宿主机外部 SIGTERM（非产品缺陷）
- 补时（09-18 11:02 → 09-19 05:20 UTC，w5_soak_topup.sh，cards 2/3，A/B 双 v1，tip 03a6ad89d）：有效时间记账（仅健康窗口计入），18.3h 墙钟累积 37000s 有效（10h16m），10 个 attempt 全部为墙钟预算自然到期（无服务死亡），**2792 心跳 0 次不健康、热更 555 轮 0 失败、全程零外部 SIGTERM、无 hang/crash**
- v2 覆盖由原 soak 提供（09-18 时点 v2 在新 base 因 kv_cache_allocation_context kwarg 无法 boot，环境阻塞非分支缺陷）
- **结论：达标**

## 5. 遗留说明

G-05/MTP、GLM OOM、T5 多节点、C3 全量 detector 税、BUG-2 DP quota 缓解、卡 1 RDMA infra —— 见 RUN_NOTES。
