# perf / scripts（NPU 起服与执行）

**归属**：仅 `feat/runtime-guard-analysis`。  
perf README 中 C1–C6 / 起服 / 交叉轮换等**尚缺的启动与执行脚本，最终都补进本目录**，  
不写回 config 产品分支。

见上级 `README.md`：C1–C6 **必须**分别在 ModelRunner **v2、v1** 上跑。

待补文件（命名建议；缺则 = 本旁支待办）：

| 脚本 | 作用 |
|------|------|
| `serve_t0.sh` | merge-base worktree 起服（无 guard bind） |
| `serve_t1.{v2,v1}.sh` | 产品 HEAD，无 additional-config |
| `serve_t2.{v2,v1}.sh` | reload=3，detectors off |
| `serve_t3.{v2,v1}.sh` | reload=3，4 detectors on |
| `run_c1_c2_cross_rotate.sh` | `RUNNER=v2\|v1` 三态交叉轮换 |
| `run_c3_ab.sh` | 调 `perf_ab_quick.py` |
| `run_c5_dump.sh` | dump_kv 开销对比（待定配置） |
| `run_c6_leakback.sh` | idle leak-back 门禁 |

公共环境变量与 `perf_lib.py` 对齐：`RG_PERF_URL`、`RG_PERF_CFG`、`RG_PERF_NPU`、`RG_PERF_OUT_*`。
