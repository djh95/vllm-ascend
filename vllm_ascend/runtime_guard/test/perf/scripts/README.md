# perf / scripts

| 脚本 | 作用 |
|------|------|
| `_common.sh` | `RUNNER`、`RG_PERF_ROOT` |
| `serve_t0`…`serve_t3.sh` | T-label 起服/配置骨架（`START_CMD` 可注入） |
| `run_c1_c2_cross_rotate.sh` | 交叉轮换步骤说明（`EXECUTE=1` 待机房全自动） |
| `run_c3_ab.sh` | `perf_ab_quick` |
| `run_c5_dump.sh` / `run_c6_leakback.sh` | dump 开销 / leak-back |

```bash
export RG_PRODUCT_ROOT=... MODEL=... RUNNER=v2 RG_PERF_ROOT=./rg_perf
bash …/perf/scripts/serve_t2.sh
bash …/perf/scripts/run_c3_ab.sh
RUNNER=v1 bash …/perf/scripts/run_c3_ab.sh
```
