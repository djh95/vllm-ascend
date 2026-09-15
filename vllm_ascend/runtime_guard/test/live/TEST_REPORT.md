# Runtime Guard 测试报告

| 项 | 内容 |
|----|------|
| 版本 | V1.0 / 2026-09-14 |
| 产品 | `feat/runtime-guard-config` @ `39e22dacc` |
| 实卡旁支 | `feat/runtime-guard-analysis` @ `62dd85456` |
| 环境 | test-mrv2-cann91 · CANN 9.1.0 · Qwen2.5-0.5B · 双 Runner |
| Word 交付件 | `vllm-ascend/docs/zh/design/word/RuntimeGuard_测试报告.docx` |

## 1. 结论

**通过。** P0+P1 功能收口；历史 6 BUG 全关；稳定性 soak 进行中（累计 ~3.6h，未达 24h）；性能 C1–C6 已出数，**C1 v1 ~1.5% 待重测**（等 8 卡空闲，暂不判定）。

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
- **执行：已启动，累计 ~3.6h（进行中，未达 24h）**
- 结果：暂无 hang/crash；服务保持可用 → **结论待累计满 24h**

## 5. 遗留说明

G-05/MTP、GLM OOM、T5 多节点、C3 全量 detector 税、BUG-2 DP quota 缓解、卡 1 RDMA infra —— 见 RUN_NOTES。
