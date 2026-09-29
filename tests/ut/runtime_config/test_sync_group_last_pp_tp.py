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

"""Sync group is last-PP TP only; others file-poll."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm_ascend.observability.runtime_config._dist as dist


def test_sync_group_none_when_not_last_pp(monkeypatch):
    pp = SimpleNamespace(is_last_rank=False)
    tp = SimpleNamespace(world_size=2)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: pp,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: tp,
    )
    assert dist._runtime_config_sync_group_or_none() is None


def test_sync_group_tp_when_last_pp_tp_gt1(monkeypatch):
    pp = SimpleNamespace(is_last_rank=True)
    tp = SimpleNamespace(world_size=2, is_first_rank=True)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: pp,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: tp,
    )
    assert dist._runtime_config_sync_group_or_none() is tp


def test_sync_group_none_when_last_pp_tp1(monkeypatch):
    pp = SimpleNamespace(is_last_rank=True)
    tp = SimpleNamespace(world_size=1)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: pp,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: tp,
    )
    assert dist._runtime_config_sync_group_or_none() is None


def test_sync_group_none_when_pp_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("no pp")

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        _boom,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        MagicMock(),
    )
    assert dist._runtime_config_sync_group_or_none() is None
