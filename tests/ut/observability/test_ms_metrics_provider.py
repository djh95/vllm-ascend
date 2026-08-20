# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import sys
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import yaml

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "vllm_ascend" / "observability"


def _load_all_provider_configs(config_paths):
    configs = []
    for config_path in config_paths:
        configs.extend(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    return configs


def _load_source(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _PACKAGE_ROOT / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses / typing that look up sys.modules[cls.__module__].
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_provider_api(monkeypatch, **attributes):
    package = ModuleType("ms_service_metric")
    provider_api = ModuleType("ms_service_metric.provider_api")
    for name, value in attributes.items():
        setattr(provider_api, name, value)
    monkeypatch.setitem(sys.modules, "ms_service_metric", package)
    monkeypatch.setitem(sys.modules, "ms_service_metric.provider_api", provider_api)


def _resolve_module_path(module_name: str) -> Path:
    if module_name.startswith("vllm."):
        return _REPOSITORY_ROOT.parent / "vllm" / Path(*module_name.split(".")).with_suffix(".py")
    return _REPOSITORY_ROOT / Path(*module_name.split(".")).with_suffix(".py")


def test_get_metric_provider_returns_packaged_yaml(monkeypatch):
    @dataclass
    class FakeMetricProvider:
        name: str
        config_paths: tuple[str, ...]
        priority: int
        owned_symbol_prefixes: tuple[str, ...]
        framework_package: str | None = None
        handler_module_prefixes: tuple[str, ...] = ()
        ownership_mode: str = "overlay"

    _install_provider_api(monkeypatch, MetricProvider=FakeMetricProvider)
    provider_module = _load_source("test_vllm_ascend_metric_provider", "provider.py")

    provider = provider_module.get_metric_provider()

    assert provider.name == "vllm-ascend"
    assert provider.framework_package == "vllm_ascend"
    assert provider.ownership_mode == "overlay"
    assert provider.owned_symbol_prefixes == (
        "vllm_ascend.",
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.async_utils",
        "vllm.v1.executor.multiproc_executor",
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer",
    )
    assert provider.handler_module_prefixes == ("vllm_ascend.observability.",)
    assert provider.config_paths == tuple(sorted(provider.config_paths))
    assert sorted(Path(path).name for path in provider.config_paths) == sorted(
        [
            "executor_metrics.yaml",
            "eplb_metrics.yaml",
            "flashcomm_metrics.yaml",
            "graph_metrics.yaml",
            "kv_metrics.yaml",
            "lifecycle_metrics.yaml",
            "parallel_metrics.yaml",
            "scheduler_metrics.yaml",
            "spec_decode_metrics.yaml",
            "async_scheduling_metrics.yaml",
        ]
    )
    assert all(Path(path).is_file() for path in provider.config_paths)
    config = _load_all_provider_configs(provider.config_paths)
    assert len(config) >= 40
    assert all(item["symbol"].startswith(("vllm_ascend.", "vllm.")) for item in config)
    assert all("id" not in item for item in config)
    assert all(
        item.get("handler", "").startswith(
            (
                "ms_service_metric.provider_handlers:",
                "vllm_ascend.observability.handlers:",
            )
        )
        for item in config
    )


def test_setup_registers_provider_entry_point_and_yaml_package_data():
    setup_path = _REPOSITORY_ROOT / "setup.py"
    syntax_tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
    )
    keywords = {keyword.arg: keyword.value for keyword in setup_call.keywords}
    entry_points = ast.literal_eval(keywords["entry_points"])
    package_data = ast.literal_eval(keywords["package_data"])

    assert entry_points["ms_service_metric.providers"] == [
        "vllm-ascend = vllm_ascend.observability:get_metric_provider"
    ]
    assert package_data["vllm_ascend.observability"] == ["config/*.yaml"]


def test_provider_entry_point_loads_without_metric_core(monkeypatch):
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.delitem(sys.modules, "ms_service_metric", raising=False)
    monkeypatch.delitem(sys.modules, "ms_service_metric.provider_api", raising=False)

    entry_point = EntryPoint(
        name="vllm-ascend",
        value="vllm_ascend.observability:get_metric_provider",
        group="ms_service_metric.providers",
    )

    assert callable(entry_point.load())


def test_ascend_handler_module_loads_from_yaml_path(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=Mock(),
    )
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    for module_name in (
        "vllm_ascend.observability",
        "vllm_ascend.observability.provider",
        "vllm_ascend.observability.handlers",
        "vllm_ascend.observability.handlers.eplb",
        "vllm_ascend.observability.handlers.flashcomm",
        "vllm_ascend.observability.flashcomm_stats",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    handler_module = __import__(
        "vllm_ascend.observability.handlers",
        fromlist=["eplb_do_update_hotness_handler"],
    )

    assert callable(handler_module.eplb_do_update_hotness_handler)
    assert callable(handler_module.eplb_transfer_stats_handler)
    assert callable(handler_module.eplb_async_worker_status_handler)
    assert callable(handler_module.flashcomm_gate_handler)
    assert callable(handler_module.flashcomm_path_handler)
    assert callable(handler_module.flashcomm_forward_flush_handler)


def test_provider_yaml_symbols_exist_in_current_vllm_ascend_source():
    config_paths = sorted((_PACKAGE_ROOT / "config").glob("*.yaml"))
    assert len(config_paths) == 10
    config = _load_all_provider_configs(config_paths)

    for item in config:
        module_name, attribute_path = item["symbol"].split(":", 1)
        source_path = _resolve_module_path(module_name)
        assert source_path.is_file(), f"Missing symbol module: {item['symbol']}"

        syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        if "." not in attribute_path:
            assert any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == attribute_path
                for node in syntax_tree.body
            ), f"Missing symbol function: {item['symbol']}"
            continue

        owner_name, method_name = attribute_path.split(".", 1)
        owner = next(
            (node for node in syntax_tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name),
            None,
        )
        assert owner is not None, f"Missing symbol owner: {item['symbol']}"
        method_names = {
            node.name for node in ast.walk(owner) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert method_name in method_names, f"Missing symbol method: {item['symbol']}"


def test_eplb_handler_records_rank_zero_hotness(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers", "handlers/eplb.py")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {
                "current_mean": 2.0,
                "current_max": 4.0,
                "update_mean": 3.0,
                "update_max": 6.0,
                "current_imbalance_list": [1.1, 1.2],
                "update_imbalance_list": [1.3, 1.4],
            },
        },
    )()

    result = handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert result == "updated"
    assert metrics.get_or_create_metric.call_count == 5
    metrics.get_or_create_metric.assert_any_call(
        "eplb:expert_hotness:current_mean",
        metric_type="gauge",
        label_names=["rank", "phase"],
    )
    metrics.get_or_create_metric.assert_any_call(
        "eplb:expert_hotness:imbalance",
        metric_type="gauge",
        label_names=["rank", "phase", "layer"],
    )
    assert metrics.record_metric.call_count == 8


def test_eplb_handler_records_rebalance_and_load_balance(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_rebalance", "handlers/eplb.py")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_rebalance_result": {
                "result": "success",
                "policy_type": 2,
                "fallback_layers": 0,
            },
            "latest_map_consistency": {
                "fallback_layers": 0,
                "duplicate_count": 0,
                "missing_count": 0,
                "num_valid_experts": 8,
            },
            "latest_load_balance": {
                "avg_tokens": 10.0,
                "max_tokens": 12.0,
                "balancedness": 10.0 / 12.0,
            },
        },
    )()

    assert handlers.eplb_do_update_hotness_handler(lambda _: "ok", worker) == "ok"
    metrics.record_metric.assert_any_call(
        "eplb:rebalance:result",
        value=1.0,
        labels={"rank": "0", "result": "success", "policy_type": "2"},
    )
    metrics.record_metric.assert_any_call(
        "eplb:load_balance:balancedness",
        value=10.0 / 12.0,
        labels={"rank": "0"},
    )
    metrics.record_metric.assert_any_call(
        "eplb:map_consistency:num_valid_experts",
        value=8.0,
        labels={"rank": "0"},
    )


def test_eplb_handler_registers_metrics_once_per_recorder(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    first_metrics = Mock()
    current_metrics = [first_metrics]
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: current_metrics[0],
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_cache", "handlers/eplb.py")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {"current_mean": 1.0},
        },
    )()

    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)
    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert first_metrics.get_or_create_metric.call_count == 5
    assert first_metrics.record_metric.call_count == 2

    second_metrics = Mock()
    current_metrics[0] = second_metrics
    handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker)

    assert second_metrics.get_or_create_metric.call_count == 5
    second_metrics.record_metric.assert_called_once()


def test_eplb_handler_skips_nonzero_rank_hotness(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_nonzero", "handlers/eplb.py")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 1,
            "latest_expert_hotness": {"current_mean": 1.0},
            "latest_rebalance_result": {"result": "skipped", "policy_type": 2},
        },
    )()

    assert handlers.eplb_do_update_hotness_handler(lambda _: "updated", worker) == "updated"
    metrics.record_metric.assert_called_once_with(
        "eplb:rebalance:result",
        value=1.0,
        labels={"rank": "1", "result": "skipped", "policy_type": "2"},
    )


def test_eplb_handler_given_metric_failure_then_preserves_inference_result(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    metrics = Mock()
    metrics.get_or_create_metric.side_effect = RuntimeError("registry unavailable")
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_failure", "handlers/eplb.py")
    worker = type(
        "Worker",
        (),
        {
            "rank_id": 0,
            "latest_expert_hotness": {"current_mean": 1.0},
        },
    )()

    assert (
        handlers.eplb_do_update_hotness_handler(
            lambda _: "updated",
            worker,
        )
        == "updated"
    )


def test_eplb_transfer_stats_handler_records_volume(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_transfer", "handlers/eplb.py")
    loader = type(
        "Loader",
        (),
        {
            "comm_group": type("G", (), {"rank_in_group": 3})(),
            "latest_transfer_stats": {
                "layer_id": 2,
                "send_experts": 4,
                "recv_experts": 5,
                "comm_ops": 9,
                "est_bytes": 1024,
            },
        },
    )()

    assert handlers.eplb_transfer_stats_handler(lambda _: None, loader) is None
    metrics.record_metric.assert_any_call(
        "eplb:transfer:est_bytes",
        value=1024.0,
        labels={"rank": "3", "layer": "2"},
    )


def test_eplb_async_worker_status_handler_records_gauges(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    handlers = _load_source("test_vllm_ascend_metric_handlers_async", "handlers/eplb.py")
    updator = type(
        "Updator",
        (),
        {
            "rank_id": 0,
            "latest_async_worker_status": {
                "worker_alive": 1.0,
                "pending_layers": 2.0,
                "seconds_since_progress": 3.5,
                "cur_iterations": 7.0,
            },
        },
    )()

    assert handlers.eplb_async_worker_status_handler(lambda _: "done", updator) == "done"
    metrics.record_metric.assert_any_call(
        "eplb:async_worker:alive",
        value=1.0,
        labels={"rank": "0"},
    )
    metrics.record_metric.assert_any_call(
        "eplb:async_worker:seconds_since_progress",
        value=3.5,
        labels={"rank": "0"},
    )


def test_flashcomm_classify_decision_and_gate_handler(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter", "HISTOGRAM": "histogram"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    stats_mod = _load_source("test_flashcomm_stats", "flashcomm_stats.py")
    handlers = _load_source("test_flashcomm_handlers_gate", "handlers/flashcomm.py")

    assert (
        stats_mod.classify_decision(
            enable_sp=False,
            is_moe_model=False,
            is_draft_model=False,
            num_tokens=2000,
            flash_comm_v1_enabled=False,
        )
        == "config_off"
    )
    assert (
        stats_mod.classify_decision(
            enable_sp=True,
            is_moe_model=False,
            is_draft_model=True,
            num_tokens=2000,
            flash_comm_v1_enabled=False,
        )
        == "dense_draft"
    )
    assert (
        stats_mod.classify_decision(
            enable_sp=True,
            is_moe_model=False,
            is_draft_model=False,
            num_tokens=500,
            flash_comm_v1_enabled=False,
        )
        == "dense_below_threshold"
    )

    out = handlers.flashcomm_gate_handler(
        lambda **kwargs: stats_mod.FlashCommMetricHooks.publish_forward_gate(**kwargs),
        decision="enabled",
        flash_comm_v1_enabled=True,
        num_tokens=2048,
        pad_size=2,
    )
    assert out.decision == "enabled"
    metrics.record_metric.assert_any_call(
        "flashcomm:decision_total",
        value=1.0,
        labels={"decision": "enabled"},
    )


def test_flashcomm_flush_handler_records_histograms(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter", "HISTOGRAM": "histogram"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    stats_mod = _load_source("test_flashcomm_stats_flush", "flashcomm_stats.py")
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    obs = ModuleType("vllm_ascend.observability")
    obs.__path__ = [str(_PACKAGE_ROOT)]
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability", obs)
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability.flashcomm_stats", stats_mod)
    handlers = _load_source("test_flashcomm_handlers_flush", "handlers/flashcomm.py")

    stats_mod.FlashCommMetricHooks.publish_forward_gate(
        decision="enabled",
        flash_comm_v1_enabled=True,
        num_tokens=100,
        pad_size=4,
    )
    stats_mod.note_collective("all_gather", elapsed_ms=1.5, nbytes=10)
    stats_mod.note_collective("reduce_scatter", elapsed_ms=2.5, nbytes=20)

    assert handlers.flashcomm_forward_flush_handler(lambda *_a, **_k: "ok", object()) == "ok"
    metrics.record_metric.assert_any_call("flashcomm:active_tokens", value=100.0, labels={})
    metrics.record_metric.assert_any_call("flashcomm:padding_ratio", value=0.04, labels={})
    metrics.record_metric.assert_any_call("flashcomm:all_gather_ms", value=1.5, labels={})
    metrics.record_metric.assert_any_call("flashcomm:reduce_scatter_ms", value=2.5, labels={})
    metrics.record_metric.assert_any_call(
        "flashcomm:communication_bytes_total",
        value=10.0,
        labels={"op": "all_gather"},
    )


def test_flashcomm_collective_timer_notes_failure(monkeypatch):
    stats_mod = _load_source("test_flashcomm_stats_timer", "flashcomm_stats.py")
    with stats_mod.CollectiveTimer("all_gather"):
        pass
    snap = stats_mod.snapshot_and_reset()
    assert snap is not None
    assert snap.all_gather_ms >= 0.0

    try:
        with stats_mod.CollectiveTimer("reduce_scatter"):
            raise RuntimeError("HCCL connection failed")
    except RuntimeError:
        pass
    snap = stats_mod.snapshot_and_reset()
    assert snap is not None
    assert snap.failures[("reduce_scatter", "connection")] == 1


def test_async_scheduling_note_handlers_record_counters(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter", "HISTOGRAM": "histogram"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    obs = ModuleType("vllm_ascend.observability")
    obs.__path__ = [str(_PACKAGE_ROOT)]
    handlers_pkg = ModuleType("vllm_ascend.observability.handlers")
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability", obs)
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability.handlers", handlers_pkg)

    stats_mod = _load_source("test_async_scheduling_stats", "async_scheduling_stats.py")
    common_mod = _load_source("test_async_common", "handlers/common.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability.handlers.common", common_mod)
    handlers = _load_source("test_async_scheduling_handlers", "handlers/async_scheduling.py")

    handlers.async_note_stale_discard_handler(stats_mod.AsyncSchedulingMetricHooks.note_stale_discard)
    handlers.async_note_underflow_handler(stats_mod.AsyncSchedulingMetricHooks.note_placeholder_underflow)

    metrics.record_metric.assert_any_call(
        "async_scheduling:stale_output_discard_total",
        value=1.0,
        labels={},
    )
    metrics.record_metric.assert_any_call(
        "async_scheduling:placeholder_underflow_total",
        value=1.0,
        labels={},
    )


def test_async_scheduler_patch_soft_fails_placeholder_underflow(monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("vllm")
    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.request import RequestStatus
    except ModuleNotFoundError as exc:
        pytest.skip(f"vllm runtime dependencies unavailable: {exc}")

    import vllm_ascend.patch.platform.patch_async_scheduler as patch_mod

    patch_mod.apply_async_scheduler_observability_patch()

    class _FakeRequest:
        request_id = "req-1"
        status = None
        num_output_placeholders = 0
        num_computed_tokens = 10
        async_tokens_to_discard = 0

    request = _FakeRequest()
    request.status = RequestStatus.RUNNING

    scheduler = object.__new__(AsyncScheduler)
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.cache_blocks = Mock()

    monkeypatch.setattr(
        "vllm.v1.core.sched.scheduler.Scheduler._update_request_with_output",
        lambda _self, _req, token_ids: (token_ids, False),
    )

    new_token_ids, stopped = AsyncScheduler._update_request_with_output(scheduler, request, [7, 8])
    assert new_token_ids == [7, 8]
    assert stopped is False
    assert request.num_output_placeholders == 0
    scheduler.kv_cache_manager.cache_blocks.assert_called_once()


def test_lifecycle_update_weights_and_routed_experts_handlers(monkeypatch):
    metric_type = type("MetricType", (), {"GAUGE": "gauge", "COUNTER": "counter", "HISTOGRAM": "histogram"})
    metrics = Mock()
    _install_provider_api(
        monkeypatch,
        MetricType=metric_type,
        get_metric_recorder=lambda: metrics,
    )
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    obs = ModuleType("vllm_ascend.observability")
    obs.__path__ = [str(_PACKAGE_ROOT)]
    handlers_pkg = ModuleType("vllm_ascend.observability.handlers")
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability", obs)
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability.handlers", handlers_pkg)

    common_mod = _load_source("test_lifecycle_common", "handlers/common.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.observability.handlers.common", common_mod)
    handlers = _load_source("test_lifecycle_handlers", "handlers/lifecycle.py")

    worker = Mock()
    worker._observability_last_wake_sleep_opt = True
    assert handlers.worker_update_weights_handler(lambda *_a, **_k: "ok", worker, {"chunk": 1}) == "ok"
    update_calls = [
        call
        for call in metrics.record_metric.call_args_list
        if call.args and call.args[0] == "lifecycle:update_weights:duration_ms"
    ]
    assert len(update_calls) == 1
    assert update_calls[0].kwargs["labels"] == {}
    assert isinstance(update_calls[0].kwargs["value"], float)

    runner = Mock()
    runner.model_config = Mock(enable_return_routed_experts=True)
    runner.routed_experts_initialized = True
    handlers.routed_experts_init_handler(lambda *_a, **_k: None, runner)
    metrics.record_metric.assert_any_call(
        "rl:routed_experts_state",
        value=1.0,
        labels={"state": "enabled"},
    )
    metrics.record_metric.assert_any_call(
        "rl:routed_experts_state",
        value=1.0,
        labels={"state": "ready"},
    )

    handlers.routed_experts_capture_handler(lambda *_a, **_k: None, Mock(), 0, Mock())
    metrics.record_metric.assert_any_call(
        "rl:routed_experts_state",
        value=1.0,
        labels={"state": "capturing"},
    )
