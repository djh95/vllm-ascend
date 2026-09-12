# live / scripts（启动 + 执行脚本）

**归属**：仅 `feat/runtime-guard-analysis`。

| 文件 | 用途 |
|------|------|
| `_common.sh` | `RUNNER`、清盘 `trap`、`df`、路径 |
| `diff_report_golden.py` | report 关键字段 vs `../golden/reports/` |
| `p0_01_guard_off.sh` … `p0_08_disk_reclaim.sh` | P0 冒烟（部分仍需机房补起服） |
| `p0_05_inject_nan.sh` | G-01 注入闭环 |

用法：

```bash
export MODEL=/path/to/weights
export RG_PRODUCT_ROOT=/path/to/product/checkout   # config 产品树
export RUNNER=v2   # or v1
# 服务已起且挂上 ../configs/<case>.json 后：
bash vllm_ascend/runtime_guard/test/live/scripts/p0_05_inject_nan.sh
KEEP_DUMP=1 bash …/p0_03_manual_dump.sh   # 保留 dump 排查
```

测后默认删 `kv_cache`（§0.4）。配置在 `../configs/`，标杆在 `../golden/`。
