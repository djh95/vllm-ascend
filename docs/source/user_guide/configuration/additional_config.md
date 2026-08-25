# Additional Configuration

Additional configuration is a mechanism provided by vLLM to allow plugins to control internal behavior by themselves. VLLM Ascend uses this mechanism to make the project more flexible.

## Migration Guide

Starting from PR #9064, vLLM Ascend is migrating **10 environment variables** to `--additional-config`.

### Important Notice

- **Current Support**: Both environment variables and `--additional-config` are supported during the transition period
- **Recommendation**: Use `--additional-config` for new deployments and migrate existing configurations
- **Future Plan**: Environment variables will be **removed** in a future release; only `--additional-config` will be supported

### Quick Reference

| Environment Variable | Config Key | Type Conversion |
|---------------------|------------|-----------------|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `enable_balance_scheduling` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `enable_flashcomm1` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `enable_matmul_allreduce` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `enable_flashcomm2_parallel_size` | Integer (unchanged) |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | Integer (unchanged, field name changed) |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | Integer (unchanged) |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | `"1"` → `true`, `"0"` → `false` |

### Example Migration

**Before (environment variable):**

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
vllm serve Qwen/Qwen3-8B
```

**After (additional-config):**

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"enable_flashcomm1": true}'
```

**dfx_config_path / dfx-config**

Path to the DFX runtime JSON controlling dump, `ascend_log`, report, and anomaly detectors
(nested `detector.<name>.enabled`, shared `detector.stop_after_alert`,
`detector.output_substring.match_prefix`, report `max_*` truncation, etc.).
If omitted, vLLM-Ascend uses `<cwd>/dfx/config/dfx_config.json` (created with defaults on first start).

**dfx_config**

Inline overlay with the **same object schema** as the DFX JSON file. Merge order:
`defaults ← dfx_config_path (if explicit) ← additional_config.dfx_config`.
Typical use: enable detectors without editing a file. Hot-reload still reads the JSON file (overlay is bootstrap-only unless persisted by the leader).

**dfx_config_reload_interval**

Hot-reload period in seconds for the DFX JSON. Default `0` (disabled; config
loaded once at startup only). Set `> 0` to enable periodic refresh. The same
value is persisted into the DFX JSON as `reload_interval_seconds` for visibility;
changing only the JSON field does not override the startup setting.
This startup setting is authoritative; it is not turned back on by fields inside the JSON
after the process has started with `0`.

On **API / EngineCore** (processes without `RANK`), the same interval also starts a daemon
thread that file-polls the JSON and applies `ascend_log` (`level` + `debug`) via
`apply_ascend_log_level` — it does **not** join the worker world broadcast and does not
write the file. Initial levels are applied at AscendConfig construction; the thread
re-applies after subsequent file changes. Workers keep step-driven sync only.

Inside the DFX JSON, `dump.manual_dump: true` keeps arming an msprobe dump on every nonempty
real `execute_model` wave until you set it back to `false` (not auto-cleared). A positive int `N`
arms the next `N` waves then clears. Skips `max_times` / cooldown; still requires dump active (`auto_max_times>0` or `manual_dump`)
and an initialized debugger.
**Requires `dfx_config_reload_interval > 0`** — with interval `0`, editing `manual_dump` in the
JSON has no effect.

Optional detect-time input filters via top-level `"input_filter": { "filters": [...] }`
(empty = no filter). Owned by the process-wide `InputFilterManager` singleton; detectors
call it before checking a request. Each entry has `type`, `mode` (`include`|`exclude`),
and type-specific fields: `input_token_id_prefix` (`prefixes`), `prompt_length`
(`op` + `value` / `min`/`max`), `prompt_contains_token_ids` (`token_ids`,
`match`=`any`|`subsequence`). Detect only when all includes match and no exclude matches.
Prefix matching is only via `type: input_token_id_prefix` (no separate prefixes field).
`dump.auto_max_times>0` and an active `manual_dump` are **mutually exclusive** (startup/reload error).

Legacy DFX dump keys (`enabled`, `max_times`, `manual_trigger`, `cooldown_seconds`) are rejected.

Manual `dump.manual_dump` bypasses filters.

To capture prompt token ids for writing filters, set
`input_filter.print_input_token_ids_once: true` (requires reload interval `> 0`). On the
next `execute_model` with requests, TP0 logs `[DFX print_input]` (`length=` + full ids +
prefix hint) and clears the flag. Design: `docs/zh/design/dfx_design.md` §2.6; ops:
`docs/zh/design/dfx_ops.md`. Annotated example:
`vllm_ascend/dfx/templates/dfx_config.example.jsonc`.

Default `sync_mode` is `broadcast`: **one JSON leader per EngineCore / DP** monitors the file and
broadcasts **inside that DP only** (`inner_dp_world`). Multi-DP never uses the full EP world for
config sync (avoids idle-DP deadlock). If `inner_dp_world` is unavailable, workers fall back to
local file poll — place a readable `dfx_config_path` on each EngineCore (per-node copy is fine;
DPs do not sync config to each other). Set `"sync_mode": "file"` for explicit per-process mtime
polling on a shared path. See `docs/zh/design/dfx_design.md` and
`docs/zh/design/dfx_ops.md` (ops / troubleshooting).

Example:

```json
{
  "dfx_config_path": "/data/dfx/config/dfx_config.json",
  "dfx_config_reload_interval": 5
}
```

## How to use

With either online mode or offline mode, users can use additional configuration. Take Qwen3 as an example:

**Online mode**:

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"config_key":"config_value"}'
```

**Offline mode**:

```python
from vllm import LLM

LLM(model="Qwen/Qwen3-8B", additional_config={"config_key":"config_value"})
```

### Configuration options

The following table lists additional configuration options available in vLLM Ascend:

| Name                                | Type | Default | Description                                                                                               |
|-------------------------------------|------|---------|-----------------------------------------------------------------------------------------------------------|
| `xlite_graph_config`                | dict | `{}`    | Configuration options for Xlite graph mode                                                                |
| `weight_prefetch_config`            | dict | `{}`    | Configuration options for weight prefetch                                                                 |
| `finegrained_tp_config`             | dict | `{}`    | Configuration options for module tensor parallelism                                                       |
| `ascend_compilation_config`         | dict | `{}`    | Configuration options for ascend compilation                                                              |
| `eplb_config`                       | dict | `{}`    | Configuration options for eplb |
| `refresh`                           | bool | `false` | Whether to refresh global Ascend configuration content. This is usually used by rlhf or ut/e2e test case. |
| `dump_config`                       | dict | `None`  | Inline msprobe dump configuration. vLLM-Ascend will materialize it to a temporary JSON file and pass that file to the debugger. |
| `dump_config_path`                  | str  | `None`  | Configuration file path for msprobe dump (compatible legacy option). At DFX bootstrap, the path is written to `dump.msprobe_config_path`. If DFX JSON does not set `dump.auto_max_times` / `dump.manual_dump` and msprobe `dump_enable` is true or omitted (msprobe default on), DFX also seeds `dump.manual_dump=true` (with `auto_max_times=0`). Explicit DFX dump keys are not overwritten. `dump_enable=false` does not seed. |
| `dump_config_isolate_by_dp`         | bool | `False` | Whether to materialize a per-DP msprobe config when `VLLM_DP_RANK` exists. Default `False`: all DPs share the same `dump_config_path` / inline dump config. When `True`, each DP uses its own copy under `<source_dir>/dp<rank>/...` and `dump_path` is auto-suffixed with `dp<rank>` to avoid cross-DP dump_enable interference and mixed dump outputs. Operate on the `dp<rank>` copy for hot updates. If the source `dump_config_path` is missing, startup fails (no silent fallback to shared path). Enable for multi-DP dump unless you intentionally share one msprobe config/path. |
| `dfx_config_path` / `dfx-config`    | str  | `None`  | Path to DFX runtime JSON (`dump` / `ascend_log` / `log` / `report` / `detector` / `input_filter`). Default: `<cwd>/dfx/config/dfx_config.json`. Hot reload: per-DP leader read + in-DP broadcast, or local file poll. `report.save_sensitive_info` defaults to `false` (lengths only in anomaly / dump_finish reports). `log.print_sampling_meta` / `log.print_output_on_finish` default `false` (ops logs only). `print_output_on_finish` accumulates only while enabled (no backfill); mid-request enable may yield a partial or empty finish log. |
| `dfx_config`                        | dict | `None`  | Inline DFX config overlay (**same schema** as the DFX JSON file). Deep-merged at startup after defaults / `dfx_config_path`. Typical use: enable detectors without editing a file. |
| `dfx_config_isolate_by_dp`          | bool | `False` | Whether to materialize a per-DP DFX config when `VLLM_DP_RANK` exists. Default `False`: all DPs share one DFX JSON path. When `True`, with explicit `dfx_config_path`/`dfx-config` each DP uses `<source_dir>/dp<rank>/<dfx_config_name>`; with default path, each DP uses `<cwd>/dfx/config/dp<rank>/dfx_config.json`. Operate on the `dp<rank>` copy for hot updates. If an explicit source config path is missing, startup fails (no silent fallback to shared path). |
| `dfx_config_reload_interval`        | float| `0`     | DFX JSON hot-reload period in seconds. Default `0` (disabled). Set `> 0` to enable periodic refresh. Also written into JSON as `reload_interval_seconds` for visibility; the startup value remains authoritative. **Required `> 0` for `dump.manual_dump`.** |
| `dfx_report_dir`                    | str  | `None`  | Directory for short anomaly reports. Default: sibling `dfx/report` next to the config dir. |
| `enable_async_exponential`          | bool | `False` | Whether to enable asynchronous exponential overlap. To enable asynchronous exponential, set this config to True.        |
| `enable_shared_expert_dp`           | bool | `False` | When the expert is shared in DP, it delivers better performance but consumes more memory. Currently only DeepSeek series models are supported. |
| `multistream_overlap_shared_expert` | bool | `False` | Whether to enable multi-stream shared expert. This option only takes effect on MoE models with shared experts. |
| `multistream_overlap_gate`          | bool | `False` | Whether to enable multi-stream overlap gate. This option only takes effect on MoE models with shared experts.  |
| `recompute_scheduler_enable`        | bool | `False` | Whether to enable the recompute scheduler. **Only valid in PD-disaggregated mode** (`kv_role` is `kv_producer` or `kv_consumer`). **Do not enable in PD-mixed mode** (no `kv_transfer_config`, or `kv_role` is `kv_both`); startup will fail with a clear error. |
| `enable_cpu_binding`                | bool | `True`  | Enables Ascend-native CPU binding on ARM servers. Set to `False` to disable. See [CPU Binding](../feature_guide/cpu_binding.md). |
| `SLO_limits_for_dynamic_batch`      | int  | `-1`    | SLO limits for dynamic batch. This is new scheduler to support dynamic batch feature                            |
| `enable_npugraph_ex`                | bool | `False` | Whether to enable npugraph_ex graph mode.                                                                 |
| `pa_shape_list`                     | list | `[]`    | The custom shape list of page attention ops.                                                              |
| `enable_kv_nz`                      | bool | `False` | Whether to enable KV cache NZ layout. This option only takes effects on models using MLA (e.g., DeepSeek).                                      |
| `layer_sharding`                    | dict | `{}`    | Configuration options for Layer Sharding Linear. Layer Sharding can only be enabled in PD-disaggregated's P node. |
| `enable_sparse_c8`                  | bool | `False` | Whether to enable KV cache C8 in DSA models (e.g., DeepSeek V3.2 and GLM5). Not supported on Ascend 950 devices now |
| `c8_enable_reshape_optim`           | bool | `False` | Whether to enable StoreKVBlock operator achieves acceleration under the C8 feature (this means that enable_sparse_c8 needs to be enabled). In the PD separation scenario, only the P node is enabled. |
| `enable_mc2_hierarchy_comm`         | bool | `False` | Enable dispatch/combine op inter-node communication by ROCE. |
| `profiling_chunk_config`            | dict | `{}`    | Configuration options for dynamic chunked pipeline parallel. See [Dynamic Chunked Pipeline Parallel](../feature_guide/dynamic_chunk_pipeline_parallel.md) for details. |
| `enable_balance_scheduling`         | bool | `False` | Whether to enable balance scheduling. Can also be configured via `VLLM_ASCEND_BALANCE_SCHEDULING` environment variable (deprecated). |
| `enable_flashcomm1`                 | bool | `False` | Whether to enable FlashComm1 optimization. Can also be configured via `VLLM_ASCEND_ENABLE_FLASHCOMM1` environment variable (deprecated). |
| `enable_matmul_allreduce`           | bool | `False` | Whether to enable matmul allreduce optimization. Can also be configured via `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` environment variable (deprecated). |
| `flashcomm2_parallel_size`          | int  | `0`     | FlashComm2 parallel size. Can also be configured via `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` environment variable (deprecated). |
| `msmonitor_use_daemon`              | bool | `False` | Whether to use daemon mode for msmonitor. Can also be configured via `MSMONITOR_USE_DAEMON` environment variable (deprecated). |
| `enable_mlapo`                      | bool | `True`  | Whether to enable MLAPO (Model Layer-wise Adaptive Parallel Optimization). Can also be configured via `VLLM_ASCEND_ENABLE_MLAPO` environment variable (deprecated). |
| `weight_nz_mode`                    | int  | `1`     | Weight NZ mode. Can also be configured via `VLLM_ASCEND_ENABLE_NZ` environment variable (deprecated). |
| `enable_context_parallel`           | bool | `False` | Whether to enable context parallelism. Can also be configured via `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` environment variable (deprecated). |
| `enable_fused_mc2`                  | int  | `0`     | Fused MC2 configuration. Can also be configured via `VLLM_ASCEND_ENABLE_FUSED_MC2` environment variable (deprecated). |
| `enable_transpose_kv_cache_by_block`| bool | `True`  | Whether to enable transpose KV cache by block. Can also be configured via `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` environment variable (deprecated). |
| `enable_dsa_cp`                     | bool | `False` | Whether to enable dsa_cp for DeepSeek V3.2, DeepSeek V4, and other models with the same architecture. This feature depends on FLASHCOMM1. Please ensure that FLASHCOMM1 is enabled before enabling this feature.|
| `rejection_sampler_config`          | dict | `{}`    | Configuration options for rejection sampler (block verify and entropy verify). |
| `multistream_dsv4_dsa_overlap`      | bool | `True`  | Whether to enable dsa multi-stream overlap for DeepSeek V4.  |

The details of each configuration option are as follows:

**xlite_graph_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable Xlite graph mode. Currently only Llama, Qwen dense series models, and Qwen3-VL are supported. |
| `full_mode` | bool | `False` | Whether to enable Xlite for both the prefill and decode stages. By default, Xlite is only enabled for the decode stage. |

**weight_prefetch_config**

| Name             | Type | Default                                                     | Description                        |
|------------------|------|-------------------------------------------------------------|------------------------------------|
| `enabled`        | bool | `False`                                                     | Whether to enable weight prefetch. |
| `prefetch_ratio` | dict | `{"attn": {"qkv": 1.0, "o": 1.0}, "moe": {"gate_up": 0.8}, "mlp": { "gate_up": 1.0,  "down": 1.0}}` | Prefetch ratio of each weight.     |

**finegrained_tp_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `lmhead_tensor_parallel_size`    | int  | `0` | The custom tensor parallel size of lm_head.    |
| `oproj_tensor_parallel_size`     | int  | `0` | The custom tensor parallel size of o_proj.     |
| `embedding_tensor_parallel_size` | int  | `0` | The custom tensor parallel size of embedding. |
| `mlp_tensor_parallel_size`       | int  | `0` | The custom tensor parallel size of mlp.       |

**ascend_compilation_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enable_npugraph_ex`               | bool | `True` | Whether to enable npugraph_ex backend.                                                 |
| `enable_static_kernel` | bool | `False` | Whether to enable static kernel. Suitable for scenarios where shape changes are minimal and some time is available for static kernel compilation. |
| `fuse_norm_quant`  | bool | `True` | Whether to enable fuse_norm_quant pass. |
| `fuse_qknorm_rope` | bool | `True` | Whether to enable fuse_qknorm_rope pass. If Triton is not in the environment, set it to False. |
| `fuse_allreduce_rms` | bool | `False` | Whether to enable fuse_allreduce_rms pass. It's set to False because of conflict with SP. |
| `fuse_muls_add` | bool | `True` | Whether to enable fuse_muls_add pass.|

**eplb_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dynamic_eplb`                   | bool| `False`| Whether to enable dynamic EPLB. |
| `expert_map_path`                | str | `None` | When using expert load balancing for an MoE model, an expert map path needs to be passed in.|
| `expert_heat_collection_interval`| int | `400`  | Forward iterations when EPLB begins. |
| `algorithm_execution_interval`   | int | `30`   | The forward iterations when the EPLB worker will finish CPU tasks. |
| `expert_map_record_path`         | str | `None` | Save the expert load calculation results to a new expert table in the specified directory.|
| `num_redundant_experts`          | int | `0`    | Specify redundant experts during initialization. |
| `eplb_policy_type`               | int | `1`    | EPLB balancing policy: `0`=Random, `1`=DefaultEplb (open-source algorithm), `2`=SwiftBalanceEplb (optimized for low-bandwidth), `3`=FlashLB (statistical method with sliding windows). |

**profiling_chunk_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled`       | bool  | `False` | Whether to enable dynamic chunked pipeline parallel. Requires `pipeline-parallel-size > 1`. |
| `smooth_factor` | float | `1.0`   | Smoothing factor (0 < x ≤ 1.0). Higher values trust the dynamic prediction more; `0.0` disables dynamic adjustment. |
| `min_chunk`     | int   | `4096`  | Minimum chunk size for dynamic calculation. Should be smaller than `max-num-batched-tokens`. |
| `need_timing` | bool | True | Enable/disable Online Calibration |

**rejection_sampler_config**

> **Note**: Both block verify and entropy verify improve speculative decoding performance (higher acceptance rate, lower latency) at the cost of reduced sampling precision. A larger `posterior_alpha` makes the adjustment more aggressive — it further lowers the acceptance threshold for high-entropy tokens, improving throughput but degrading output quality. Users should tune these parameters based on their specific model weights and application scenario to find the right trade-off between performance and precision.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enable_block_verify`   | bool  | `False` | Whether to enable block verify mode. Block verify evaluates all draft tokens as a block using cumulative probability products, which can improve acceptance rate. |
| `enable_entropy_verify` | bool  | `False` | Whether to enable entropy verify mode. Entropy verify adjusts the acceptance threshold based on the entropy of the target distribution — higher entropy (uncertain) tokens get a lower threshold (easier to accept), while lower entropy (confident) tokens get a stricter threshold. |
| `posterior_threshold`   | float | `0.95`  | Upper bound for the entropy-adjusted acceptance threshold. Must be in (0, 1]. The effective threshold is `min(exp(-entropy * posterior_alpha), posterior_threshold)`. |
| `posterior_alpha`       | float | `0.4`   | Scaling factor for entropy in the threshold computation. Must be >= 0. Higher values make the threshold more sensitive to entropy — high-entropy tokens become much easier to accept, improving performance but reducing precision. |

### Example

An example of additional configuration is as follows:

```python
{
    "weight_prefetch_config": {
        "enabled": True,
        "prefetch_ratio": {
            "attn": {
                "qkv": 1.0,
                "o": 1.0,
            },
            "moe": {
                "gate_up": 0.8
            },
            "mlp": {
                "gate_up": 1.0,
                "down": 1.0
            }
        },
    },
    "finegrained_tp_config": {
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
    },
    "enable_kv_nz": False,
    "multistream_overlap_shared_expert": True,
    "rejection_sampler_config": {
        "enable_block_verify": True,
        "enable_entropy_verify": True,
        "posterior_threshold": 0.95,
        "posterior_alpha": 0.4,
    },
    "refresh": False
}
```
