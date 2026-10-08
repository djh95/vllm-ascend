# live / scripts

| 文件 | 用途 |
|------|------|
| `_common.sh` | RUNNER、清盘、`df` |
| `run_both_runners.sh` | 先 v2 再 v1 跑同一脚本 |
| `g_inject.sh` + `g01`…`g06` | §10 注入闭环 |
| `p0_01`…`p0_08` | P0 冒烟 |
| `diff_report_golden.py` / `refresh_golden.py` | 标杆对比 / 从实卡 report 刷新 |

```bash
export RG_PRODUCT_ROOT=/path/to/config-checkout
export MODEL=... SERVED_MODEL_NAME=...
# 服务已挂对应 configs/*.json 后：
bash …/live/scripts/run_both_runners.sh g01_nan_logits.sh
KEEP_DUMP=1 bash …/p0_03_manual_dump.sh
python3 …/refresh_golden.py --report path/report.json --out …/golden/reports/g01_nan_logits.json
```
