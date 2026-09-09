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

"""UT for KV meta compatibility gate (refuse unadapted layouts)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_ascend.runtime_guard import kv_meta_compat


@pytest.fixture(autouse=True)
def _reset_kv_meta_compat():
    kv_meta_compat.reset_for_tests()
    yield
    kv_meta_compat.reset_for_tests()


def _ascend(**kwargs):
    base = dict(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        enable_sparse_sfa_c8=False,
        enable_sparse_li_c8=False,
        enable_dsa_cp=False,
        kv_offload_config=None,
        recompute_cpu_offload_config=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _runner(*, ascend=None, **vllm_kw):
    model = SimpleNamespace(
        is_hybrid=False,
        sliding_window=None,
        get_sliding_window=lambda: None,
    )
    cache = SimpleNamespace(mamba_cache_mode="none", enable_prefix_caching=False)
    vllm = SimpleNamespace(model_config=model, cache_config=cache, kv_transfer_config=None)
    for k, v in vllm_kw.items():
        if k == "model_config":
            vllm.model_config = v
        elif k == "cache_config":
            vllm.cache_config = v
        elif k == "kv_transfer_config":
            vllm.kv_transfer_config = v
        else:
            setattr(vllm, k, v)
    return SimpleNamespace(ascend_config=ascend or _ascend(), vllm_config=vllm)


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


def test_sparse_blocks_and_force_disables():
    runner = _runner(ascend=_ascend(sparse_kv_offload_config=SimpleNamespace(enabled=True)))
    cfg = _cfg()
    assert kv_meta_compat.apply_kv_meta_compat(runner, cfg) is True
    assert kv_meta_compat.is_kv_meta_blocked()
    assert "sparse_kv_offload" in kv_meta_compat.blocked_reasons()
    assert cfg._data["invariant"]["slot_consistency"]["enabled"] is False
    assert cfg._data["report"]["kv_audit"] is False
    assert cfg._data["invariant"]["logits_finite"]["enabled"] is True


def test_sliding_window_blocks():
    runner = _runner(
        model_config=SimpleNamespace(
            is_hybrid=False,
            sliding_window=4096,
            get_sliding_window=lambda: 4096,
        ),
    )
    reasons = kv_meta_compat.probe_incompatible_features(runner)
    assert "sliding_window" in reasons


def test_prefix_caching_now_blocks():
    runner = _runner(cache_config=SimpleNamespace(mamba_cache_mode="none", enable_prefix_caching=True))
    cfg = _cfg(slot=True)
    assert kv_meta_compat.apply_kv_meta_compat(runner, cfg) is True
    assert "prefix_caching" in kv_meta_compat.blocked_reasons()
    assert cfg._data["invariant"]["slot_consistency"]["enabled"] is False


def test_kv_transfer_and_sfa_block():
    runner = _runner(
        ascend=_ascend(enable_sparse_sfa_c8=True),
        kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector"),
    )
    reasons = kv_meta_compat.probe_incompatible_features(runner)
    assert "enable_sparse_sfa_c8" in reasons
    assert "kv_transfer" in reasons
    assert "kv_connector=MooncakeConnector" in reasons


def test_dense_pa_only_allows_meta():
    runner = _runner()
    cfg = _cfg(slot=True)
    assert kv_meta_compat.probe_incompatible_features(runner) == []
    assert kv_meta_compat.apply_kv_meta_compat(runner, cfg) is False
    assert not kv_meta_compat.is_kv_meta_blocked()
    assert cfg._data["invariant"]["slot_consistency"]["enabled"] is True
