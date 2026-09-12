# live / scripts（启动 + 执行脚本）

**归属**：仅 `feat/runtime-guard-analysis`。  
清单里写到、仓库里还没有的启动/执行脚本，**最终都要加到本目录**（配置进 `../configs/`），  
不要写到 `feat/runtime-guard-config`。

约定：

- `RUNNER=v2|v1` 或成对文件 `*.v2.sh` / `*.v1.sh`
- 配置 JSON 放 `../configs/`；标杆放 `../golden/`（小文件）
- 补齐顺序：先 P0（含 P0-7/P0-8）、§10 注入闭环、§15 dump/标杆，再按清单各节 ID
- 凡产生 `kv_cache` 的脚本：**`trap` 测后删盘**（除非 `KEEP_DUMP=1`）
- 注入类脚本须：命中 detector → report 对 golden →（可选）dump 结构检查 → skill 初步定位 → 更新 skill（若有偏差）
- README / 占位 ≠ 完成；须可跑脚本

示例骨架（尚未接真实模型路径）：

```bash
#!/usr/bin/env bash
# live/scripts/p0_02_token_repeat.sh
set -euo pipefail
RUNNER="${RUNNER:-v2}"   # v2|v1
export VLLM_USE_V2_MODEL_RUNNER=$([ "$RUNNER" = v2 ] && echo 1 || echo 0)
DUMP_ROOT="${DUMP_ROOT:-./runtime/report/kv_cache}"
cleanup() { [[ "${KEEP_DUMP:-0}" = 1 ]] || rm -rf "$DUMP_ROOT"; }
trap cleanup EXIT
# TODO: ASCEND_RT_VISIBLE_DEVICES, MODEL, PORT, additional-config
# TODO: curl + 验收 report/kv_cache + golden diff + verify_request_kv
```
