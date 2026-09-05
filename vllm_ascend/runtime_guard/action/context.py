#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action execution context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vllm_ascend.runtime_guard.incident import Incident
from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader
from vllm_ascend.runtime_guard.quota import DumpQuota
from vllm_ascend.runtime_guard.report import ReportWriter
from vllm_ascend.runtime_config.config import RuntimeConfig


@dataclass
class ActionContext:
    incident: Incident
    runner: Any
    runtime_config: RuntimeConfig
    report_writer: ReportWriter
    kv_reader: KvCacheReader
    quota: DumpQuota
    rank_tag: str
    tokenizer: Any | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    action_overrides: dict[str, Any] = field(default_factory=dict)
    batch_rows: list[tuple[str, int | None]] | None = None
    # Enqueue a heavy async job (B'3): actions with many per-piece payloads
    # submit incrementally instead of accumulating them in host RAM.
    submit_async: Callable[..., None] | None = None
