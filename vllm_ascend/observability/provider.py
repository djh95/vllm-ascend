# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metric configuration contributed to MS Service Metric."""

from pathlib import Path


def get_metric_provider():
    """Return the provider lazily so vLLM Ascend has no hard Core dependency."""
    from ms_service_metric.provider_api import MetricProvider  # type: ignore[import-not-found]

    config_dir = Path(__file__).with_name("config")
    config_paths = tuple(str(config_path) for config_path in sorted(config_dir.glob("*.yaml")))
    return MetricProvider(
        name="vllm-ascend",
        config_paths=config_paths,
        priority=200,
        framework_package="vllm_ascend",
        owned_symbol_prefixes=(
            "vllm_ascend.",
            "vllm.v1.core.sched.scheduler",
            "vllm.v1.core.sched.async_scheduler",
            "vllm.v1.worker.gpu.model_runner",
            "vllm.v1.worker.gpu_model_runner",
            "vllm.v1.worker.gpu.async_utils",
            "vllm.v1.executor.multiproc_executor",
            "vllm.model_executor.layers.fused_moe.routed_experts_capturer",
        ),
        handler_module_prefixes=("vllm_ascend.observability.",),
    )
