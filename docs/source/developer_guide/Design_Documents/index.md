# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Runtime Guard design](runtime_guard_design.md) — Online anomaly detect / report / dump_kv (hosted on analysis; not in product PR).
- [Runtime Guard ops](runtime_guard_ops.md) — Deployment and troubleshooting runbook (analysis-hosted).
- Chinese copies: `docs/zh/design/runtime_guard_{design,ops}.md`.
