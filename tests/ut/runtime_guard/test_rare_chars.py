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

# mypy: ignore-errors
"""UT: rare_chars rc4 Guard wiring + (top1, chosen) collect without API logprobs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.observability.runtime_config.config import RuntimeConfig
from vllm_ascend.observability.runtime_guard.detector.rare import CONFIG_RC4, RareDetector
from vllm_ascend.observability.runtime_guard.detector.rare_chars import (
    RareCharsDetector,
    top1_and_chosen_logprobs,
)
from vllm_ascend.observability.runtime_guard.hooks import need_pre_sample_hook
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore


def _cfg(tmp_path: Path, **rare_kw) -> RuntimeConfig:
    RequestGuardStore.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        hot_reload=False,
    )
    sec = cfg._data["detector"]["rare_chars"]
    sec["enabled"] = True
    sec.update(rare_kw)
    cfg._invalidate_hot_path_gates()
    return cfg


def test_catalog_includes_rare_chars_defaults():
    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.detector_catalog import DETECTOR_SECTIONS

    assert "rare_chars" in DETECTOR_SECTIONS
    sec = _DEFAULTS["detector"]["rare_chars"]
    assert sec["enabled"] is False
    assert sec["table"] == "qwen3"
    assert sec["think_end"] == "auto"
    assert sec["alert_monitor"] is False


def test_top1_and_chosen_match_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    tids = torch.randint(0, 32, (4,))
    top1, chosen = top1_and_chosen_logprobs(logits, tids)
    lp = torch.log_softmax(logits.float(), dim=-1)
    assert torch.allclose(top1.cpu(), lp.max(dim=-1).values.cpu(), rtol=1e-4, atol=1e-4)
    ref_c = lp.gather(1, tids.unsqueeze(-1)).squeeze(1)
    assert torch.allclose(chosen.cpu(), ref_c.cpu(), rtol=1e-4, atol=1e-4)


def test_rare_detector_bad_utf8_rule_fires():
    """rc4 strict ``bad_utf8`` fires on a lone continuation byte (from upstream synth)."""
    from vllm_ascend.observability.runtime_guard.detector.rare.rare_vocab import load as load_rare

    T = load_rare("qwen3")
    # Find a token marked incomplete UTF-8 in the rare table.
    bad_ids = np.flatnonzero(T["u8"])
    assert len(bad_ids) > 0
    tid = int(bad_ids[0])
    # Natural-log logprobs (arbitrary; bad_utf8 rule ignores them).
    det = RareDetector("qwen3", CONFIG_RC4, think_end=None)
    st = det.start()
    # Feed enough tokens so look-ahead configs can decide (rc4 looks one ahead).
    ids = np.array([tid, tid], np.int64)
    lp = np.zeros(2, np.float64)
    res = det.feed(st, ids, lp, lp)
    res = res or det.finish(st)
    assert res is not None and res.is_ill
    assert "bad_utf8" in res.reason


def test_rare_chars_detector_from_logits_bad_utf8(tmp_path: Path):
    from vllm_ascend.observability.runtime_guard.detector.rare.rare_vocab import load as load_rare

    T = load_rare("qwen3")
    bad_ids = np.flatnonzero(T["u8"])
    tid = int(bad_ids[0])
    V = len(T["rcls"])
    # One-hot-ish logits so chosen is well-defined; rule is bad_utf8.
    logits = torch.full((1, V), -20.0)
    logits[0, tid] = 5.0

    cfg = _cfg(tmp_path, table="qwen3", think_end="none")
    det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    # First step holds (look-ahead); second step / finish decides.
    alerts = det.check_from_logits(
        logits=logits,
        sampled_token_ids=[[tid]],
        req_ids=["r1"],
    )
    alerts2 = det.check_from_logits(
        logits=logits,
        sampled_token_ids=[[tid]],
        req_ids=["r1"],
    )
    finish = det.finish_request("r1")
    fired = alerts + alerts2 + ([finish] if finish is not None else [])
    assert any(a.incident_type == "rare_chars" for a in fired)


def test_need_pre_sample_hook_for_rare_chars(tmp_path: Path):
    cfg = _cfg(tmp_path)
    guard = SimpleNamespace(
        runtime_config=cfg,
        action_executor=SimpleNamespace(can_run_detection=lambda: True),
        runner=SimpleNamespace(),
    )
    assert need_pre_sample_hook(guard) is True
    cfg._data["detector"]["rare_chars"]["enabled"] = False
    cfg._invalidate_hot_path_gates()
    assert need_pre_sample_hook(guard) is False


def test_rare_chars_disabled_is_quiet(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg._data["detector"]["rare_chars"]["enabled"] = False
    cfg._invalidate_hot_path_gates()
    det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    logits = torch.randn(1, 64)
    assert det.check_from_logits(logits=logits, sampled_token_ids=[[1]], req_ids=["r"]) == []
