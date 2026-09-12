# perf / scripts（NPU 起服与执行）

**归属**：仅 `feat/runtime-guard-analysis`。默认路径经 `RG_PERF_ROOT`（见 `perf_lib.py`），勿写死 `/data0`。

| 脚本 | 作用 |
|------|------|
| `serve_t1.sh` | T1 起服骨架（`RUNNER=v1\|v2`） |
| `run_c3_ab.sh` | 调 `perf_ab_quick`（C3） |
| （待补）`serve_t0/t2/t3`、`run_c1_c2_cross_rotate`、`run_c5_dump`、`run_c6_leakback` | 见上级 README |

```bash
export MODEL=... RG_PRODUCT_ROOT=... RUNNER=v2 RG_PERF_ROOT=./rg_perf
bash vllm_ascend/runtime_guard/test/perf/scripts/serve_t1.sh
# server up:
bash vllm_ascend/runtime_guard/test/perf/scripts/run_c3_ab.sh
```
