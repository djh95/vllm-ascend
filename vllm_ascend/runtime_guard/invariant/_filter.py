#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Shared input-filter gate for invariant checks."""

from __future__ import annotations

from typing import Any


def allow_req(
    req_id: str,
    *,
    runner: Any,
    req_idx: int | None = None,
) -> bool:
    from vllm_ascend.runtime_guard.input_filters import InputFilterManager

    return InputFilterManager.get().allow(
        req_id,
        runner=runner,
        req_idx=req_idx,
        log=False,
    )
