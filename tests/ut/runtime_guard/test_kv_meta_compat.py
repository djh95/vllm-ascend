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

"""UT for KV meta compatibility gate (refuse unsupported layouts)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_ascend.runtime_guard import kv_meta_compat


@pytest.fixture(autouse=True)
def _reset_kv_meta_compat():
    kv_meta_compat.reset_for_tests()
    yield
    kv_meta_compat.reset_for_tests()


def _cfg(*, slot: bool = True, kv_audit: bool = True) -> SimpleNamespace:
    data = {
        "report": {"kv_audit": kv_audit, "block_state": False},
        "invariant": {
            "slot_consistency": {"enabled": slot},
            "kv_slot_order": {"enabled": False},
            "kv_state": {"enabled": False},
            "logits_finite": {"enabled": True},
        },
    }

    def invariant_get(section: str, key: str, default=None):
        sec = data["invariant"].get(section) or {}
        return sec.get(key, default)

    return SimpleNamespace(_data=data, invariant_get=invariant_get)


def test_sparse_blocks_and_force_disables(monkeypatch):
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(
            sparse_kv_offload_config=SimpleNamespace(enabled=True),
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=False, sliding_window=None),
            cache_config=SimpleNamespace(mamba_cache_mode="none", enable_prefix_caching=False),
            kv_transfer_config=None,
        ),
    )
    cfg = _cfg()
    assert kv_meta_compat.apply_kv_meta_compat(runner, cfg) is True
    assert kv_meta_compat.is_kv_meta_blocked()
    assert "sparse_kv_offload" in kv_meta_compat.blocked_reasons()
    assert cfg._data["invariant"]["slot_consistency"]["enabled"] is False
    assert cfg._data["report"]["kv_audit"] is False
    assert cfg._data["invariant"]["logits_finite"]["enabled"] is True


def test_sliding_window_blocks():
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                is_hybrid=False,
                sliding_window=4096,
                get_sliding_window=lambda: 4096,
            ),
            cache_config=SimpleNamespace(mamba_cache_mode="none", enable_prefix_caching=False),
            kv_transfer_config=None,
        ),
    )
    reasons = kv_meta_compat.probe_incompatible_features(runner)
    assert "sliding_window" in reasons


def test_prefix_caching_not_a_blocker_but_load_tag():
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=False, sliding_window=None, get_sliding_window=lambda: None),
            cache_config=SimpleNamespace(mamba_cache_mode="none", enable_prefix_caching=True),
            kv_transfer_config=None,
        ),
    )
    assert kv_meta_compat.probe_incompatible_features(runner) == []
    assert "prefix_caching" in kv_meta_compat.probe_external_load_features(runner)
    cfg = _cfg(slot=True)
    assert kv_meta_compat.apply_kv_meta_compat(runner, cfg) is False
    assert not kv_meta_compat.is_kv_meta_blocked()
    assert cfg._data["invariant"]["slot_consistency"]["enabled"] is True
