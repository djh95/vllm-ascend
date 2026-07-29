# Dumper 代码审查：待完善清单（历史快照）

> **已过时**：本文为 2026-07 审查快照，多项问题（如硬编码 `setLevel(DEBUG)`、旧 API 名）已在
> `vllm_ascend/dfx/` + `ascend_log` / `apply_ascend_log_level` 落地后失效。  
> 请以 [dfx_design.md](./dfx_design.md) / [dumper_design.md](./dumper_design.md) 与当前代码为准；
> 下文仅作历史参考。
>
> 审查时间：2026-07-25；路径更新：2026-07-28  
> 审查范围（历史）：原 `vllm_ascend/dumper.py`；现实现已迁至 `vllm_ascend/dfx/`。  
> 调用点：`vllm_ascend/worker/v2/model_runner.py`、`vllm_ascend/worker/model_runner_v1.py`

---

## 严重程度说明

| 标记 | 含义 |
|------|------|
| 🔴 | Bug / 功能异常 |
| 🟡 | 逻辑瑕疵 / 不够健壮 |
| 🟢 | 代码风格 / 可维护性 |

---

## 一、dumper.py 自身问题

### 🔴 1. `logger.setLevel(logging.DEBUG)` 硬编码——绕过环境变量

**位置**: `dumper.py:36`

```python
logger = init_logger_ascend(__name__)
logger.setLevel(logging.DEBUG)  # ← 硬编码!
```

**问题**：直接 `setLevel(logging.DEBUG)` 绕过了 `VLLM_LOGGING_LEVEL` 环境变量配置。正确的做法是依赖 vLLM logging 体系——父 logger `"vllm"` 的 level 由 `VLLM_LOGGING_LEVEL` 控制，子 logger 继承即可。当前这行代码你加上后 debug 日志确实出来了，但它是用 "硬改" 方式实现的，不优雅。

**建议**：删除这一行，改为启动时设置 `VLLM_LOGGING_LEVEL=DEBUG`。

---

### ✅ 2. `load_model` 中 `start_dump_data()` 是图模式预启动

**位置**: `model_runner_v2.py:197-198`

```python
def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
    super().load_model(load_dummy_weights, *args, **kwargs)
    if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        self.dumper.start_dump_data()  # ← 只 start，没有 finalize!
```

**结论**：这不是状态泄漏。v1/v2 都在图模式下于 `load_model` 末尾提前启动
`AclGraphDumper`，用于覆盖后续构图/capture；第一次 dummy/capture 经
`execute_model(..., dummy_run=True)` 后调用 `finalize_dump_data(dump=False)` 闭合。
不能在 `load_model` 末尾立即 finalize，否则会改变图捕获边界。

---

### 🔴 3. `check_spec_acceptance_anomaly` 无条件弹出 `debug_log_full` 标记

**位置**: `dumper.py:231`

```python
def check_spec_acceptance_anomaly(self, req_idx, req_id, req_state, ...):
    if not req_id:
        return
    ...
    draft_len = getattr(req_state, "prev_num_draft_len", 0) or 0
    if draft_len <= 0:
        return
    self._debug_log_full_by_req_id.pop(req_id, None)  # ← 无条件 pop!
    ...
```

**问题**：`_debug_log_full_by_req_id.pop(req_id, None)` 在 draft_len > 0 时**每次都会执行**，即使后续 spec 异常检查没有触发。如果 token_logprob 检测已经在该步写入了 `debug_log_full["req1"] = True`，但 spec 检测未触发（接受率正常），这个 pop 会把 token_logprob 写入的标记误删。

**建议**：把 pop 移到 `should_log_full` 判断通过之后：

```python
if not should_log_full:
    return
# 移到这里：
self._debug_log_full_by_req_id.pop(req_id, None)
```

---

### ✅ 4. `_msprobe_dump_total_count` 只统计成功激活

**位置**: `dumper.py:610-628`

```python
def enable_msprobe_dump_if_needed(self, ...):
    ...
    if self._use_pending_dump_sync():
        self._pending_dump = True
        self._pending_dump_req_id = req_id
        self._msprobe_dumped_req_ids.add(req_id)     # ← 记录了
        self._msprobe_last_dump_ts = now_ts           # ← 记录了
        # ❌ _msprobe_dump_total_count 没有递增!
        ...
        return True

    return self._activate_msprobe_dump(req_id)  # ← 这里面才递增
```

**结论**：pending 只是预备状态，OR 同步或配置加载仍可能失败，不应提前消耗配额。
计数继续在 `_activate_msprobe_dump` 成功后递增；pending 日志改为
`next_activation_count`，避免把当前成功次数误读为本次已经激活。

---

### 🟡 5. `check_spec_acceptance_anomaly` 退化读取 `req_state.output_token_ids`

**位置**: `dumper.py:247-251`

```python
req_output_token_ids = getattr(self.runner.input_batch, "req_output_token_ids", None)
if req_output_token_ids is not None and 0 <= req_idx < len(req_output_token_ids):
    output_token_ids_raw = req_output_token_ids[req_idx]
else:
    output_token_ids_raw = getattr(req_state, "output_token_ids", None)  # ← 全是 -1!
```

**问题**：如果 `input_batch.req_output_token_ids` 不可用，退化到 `req_state.output_token_ids`。但在异步场景下，后者全是 `-1` 占位符。日志里会打印 `output_token_ids=[-1, -1, -1, ...]`，日志本身没意义且误导。

**建议**：退化时加上 warning 日志，或者直接从 `input_batch` 强制读取，不退化到 `req_state`。

---

### 🟡 6. `sync_dump_pending_or` 返回值被丢弃

**位置**: `model_runner_v2.py:229`

```python
self.dumper.sync_dump_pending_or(
    async_mode=self.use_async_scheduling,
    allow_arm=not dummy_run,
)
# ← 返回值 bool 被丢弃!
```

**问题**：`sync_dump_pending_or` 返回 `bool`（表示是否处于 dump 状态），但 v2 的 `execute_model` 丢弃了返回值。v1 中也一样丢弃了。返回值的目标用途不明确——可能最初设计是让调用方根据返回决定是否执行某些逻辑。

**建议**：如果返回值不需要，去掉返回类型或改为 `-> None`。如果需要，在调用方使用它。

---

### 🟡 7. `enable_msprobe_dump_if_needed` 的 `skip_related_check` 语义不完整

**位置**: `dumper.py:590`

```python
def enable_msprobe_dump_if_needed(self, ..., *, skip_related_check: bool = False):
    ...
    if not get_pp_group().is_last_rank:   # ← skip_related_check 不跳过这个
        return False
    if not skip_related_check and not self.is_related_local_request(...):
        return False
```

**问题**：参数名叫 `skip_related_check`，但实际上只跳过了 `is_related_local_request`，仍然检查了 `is_last_rank`。调用方（`check_token_logprob_anomaly`）使用 `skip_related_check=True` 时注释写的是：

```python
# Token/logprob check uses output snapshots (esp. async get_output);
# live input_batch may already be empty, so skip related-local gate.
```

这个注释说的是对的——`is_related_local_request` 依赖 live `input_batch`，但 `is_last_rank` 检查是合理的，不需要跳过。不过参数名 `skip_related_check` 容易让人以为会跳过所有前置检查。

**建议**：重命名为 `skip_liveness_check` 或在文档中更清晰地说明。

---

### 🟢 8. `_debug_log_full_by_req_id` 冗余别名

**位置**: `dumper.py:97`

```python
self.full_log_requests_this_step: dict[str, bool] = {}
...
self._debug_log_full_by_req_id: dict[str, bool] = self.full_log_requests_this_step
```

**问题**：`_debug_log_full_by_req_id` 就是 `full_log_requests_this_step` 的别名，两者指向同一个 dict。代码中有的地方用 `_debug_log_full_by_req_id`，有的用 `full_log_requests_this_step`，造成混淆。

外部调用方直接用 `self.dumper.full_log_requests_this_step`。

**建议**：统一使用一个名字，删除别名。

---

### 🟢 9. `finalize_dump_data(**kwargs)` 用 kwargs 传参

**位置**: `dumper.py:194`

```python
def finalize_dump_data(self, **kwargs) -> None:
    ...
    dump_kw = kwargs.get("dump", True)
```

**问题**：核心参数 `dump` 隐藏在 `**kwargs` 中，IDE 无法提示，调用方容易写错。当前调用方全都正确使用了 `dump=False` / `dump=not dummy_run`，但缺乏类型安全保障。

**建议**：改为显式参数：

```python
def finalize_dump_data(self, *, dump: bool = True) -> None:
```

---

### 🟢 10. `start_dump_data` 方法层级混乱

**位置**: `dumper.py:172`

```python
def start_dump_data(self) -> None:
    self.full_log_requests_this_step.clear()  # ← 这是 per-step 逻辑，跟 debugger 无关
    if self._debugger is None:
        return
    if self._debugger_started:
        return
    ...
```

**问题**：`start_dump_data` 做了两件事：
1. 清理 per-step 标记（`full_log_requests_this_step.clear()`）——每次都需要的
2. 启动 debugger——只在有 debugger 且未启动时需要

每步第一条 `full_log_requests_this_step.clear()` 跟 debugger 启动逻辑混在一起，职责不单一。

**建议**：拆分为 `reset_per_step_state()` 和 `ensure_debugger_started()` 两个方法，或在方法内加注释分块。

---

## 二、v2 model_runner.py 调用方问题

### 🔴 11. `_attach_observability_fields` 同步路径的 debug_log_full 时机

**位置**: `model_runner_v2.py:283-284`

```python
# 同步路径
if isinstance(output, ModelRunnerOutput):
    model_runner_output_fields = getattr(ModelRunnerOutput, "__dataclass_fields__", {})
    if "debug_log_full" in model_runner_output_fields:
        model_runner_output.debug_log_full = dict(
            self.dumper.full_log_requests_this_step
        )
```

**对比 sample_tokens 中的 check 时机** (`model_runner_v2.py:253`)：

```python
if isinstance(output, ModelRunnerOutput):
    # Sync: super() already called get_output(); check immediately.
    self.dumper.check_all_token_logprobs(...)
    # ← 这里调用 check，check 中可能写 full_log_requests_this_step
    self._attach_observability_fields(output)  # ← 然后在这里快照
```

同步路径的顺序是 `check_all_token_logprobs` → `_attach_observability_fields`，时机正确。但 `_attach_observability_fields` 内部还调用了 `finalize_dump_data` 之后的逻辑吗？不对，让我看一下...

实际上 v2 的 `execute_model` 中：
```python
self.dumper.start_dump_data()
try:
    return super().execute_model(...)  # 这里面会调用 self.sample_tokens
finally:
    self.dumper.finalize_dump_data(dump=not dummy_run)
```

`finalize_dump_data` 在 finally 块，先于 `sample_tokens` 吗？不对，`super().execute_model()` 内部包含了 sample_tokens，所以顺序是：

1. `start_dump_data()`
2. forward
3. `sample_tokens()` → `check_all_token_logprobs()` → `_attach_observability_fields()` (快照)
4. `finalize_dump_data()`

所以快照在 finalize 之前，不会被 `disable_msprobe_dump_if_needed` 等影响。时机是正确的，但调用链较长——`_attach_observability_fields` 里面直接读 `dumper.full_log_requests_this_step`，依赖对调用顺序的隐式理解。

**建议**：在 `_attach_observability_fields` 添加注释说明调用时序依赖。

---

### 🟢 12. v2 和 v1 的 `AscendAsyncOutput` 实现不一致

**v1** (`model_runner_v1.py:266-296`)：
```python
class AscendAsyncGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    def __init__(self, *args, dumper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dumper = dumper

    def get_output(self):
        output = super().get_output()
        if self._dumper is None:
            return output
        self._dumper.check_all_token_logprobs(...)
        output.debug_log_full = dict(self._dumper.full_log_requests_this_step)
        return output
```

**v2** (`model_runner_v2.py:64-87`)：
```python
class AscendAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, inner: AsyncOutput, dumper: Dumper):
        self._inner = inner
        self._dumper = dumper

    def get_output(self) -> ModelRunnerOutput:
        output = self._inner.get_output()
        self._dumper.check_all_token_logprobs(...)
        output.debug_log_full = dict(self._dumper.full_log_requests_this_step)
        return output
```

**差异**：
- v1 继承 `AsyncGPUModelRunnerOutput`，v2 继承 `AsyncModelRunnerOutput`
- v1 的 `_dumper` 可以是 None，v2 不能
- v1 用 `*args/**kwargs`，v2 用显式参数
- v2 有更完善的 docstring

**建议**：两个类的行为一致即可，不需要强行统一。但 v2 的 `_dumper` 应该也允许 None（防御性编程）。

---

## 三、v1 model_runner.py 调用方关注点

### 🟡 13. v1 的 start/finalize 配对点多且分散

**位置**: `model_runner_v1.py`

| start 位置 | finalize 位置 | 场景 |
|---|---|---|
| L2114 | L2120 | EC producer 编码 |
| L2154 | L2408/L2418/L2640/L3757 | 正常推理 / 提前返回 / dummy |

**问题**：`start_dump_data` 被调用了，但 `finalize_dump_data` 有 4 个可能的配对点（L2408、L2418、L2640、L3757），取决于执行路径。如果未来有人新增了一条提前返回路径但忘了加 finalize，debugger 状态会悬挂。

**建议**：用 try/finally 包裹需要配对的逻辑，或者用 context manager 自动配对。

---

## 四、汇总优先级

| 优先级 | 编号 | 问题 | 影响 |
|--------|------|------|------|
| P0 | 3 | debug_log_full 被无条件 pop | token_logprob 标记可能丢失 |
| P1 | 1 | logger.setLevel 硬编码 | 日志级别不可控 |
| P2 | 5 | 退化到 req_state.output_token_ids | 异步下日志全是 -1 |
| P2 | 9 | finalize_dump_data(**kwargs) | 接口不够清晰 |
| P3 | 6-8,10-13 | 其余 | 代码可维护性 |
