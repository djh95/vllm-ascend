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

"""UTs for kv_audit bus + block/slot meta state machine."""

from __future__ import annotations

import logging

import torch

from vllm_ascend.runtime_guard import kv_audit
from vllm_ascend.runtime_guard.kv_block_meta import BlockState, KvBlockMetaTracker


def setup_function() -> None:
    kv_audit.reset_for_tests()
    KvBlockMetaTracker.reset_for_tests()


def teardown_function() -> None:
    kv_audit.reset_for_tests()
    KvBlockMetaTracker.reset_for_tests()


def test_audit_disabled_is_noop():
    kv_audit.configure(enabled=False)
    kv_audit.on_block_load([1, 2], tag="reshape_and_cache")
    assert KvBlockMetaTracker.get().block_state(1) == BlockState.UNKNOWN


def test_on_block_load_marks_sealed():
    kv_audit.configure(enabled=True, block_size=16)
    kv_audit.on_block_load([3, 5], tag="kv_load")
    t = KvBlockMetaTracker.get()
    assert t.block_state(3) == BlockState.BLOCK_SEALED_LOAD
    assert t.block_state(5) == BlockState.BLOCK_SEALED_LOAD
    detail = {e["block_id"]: e for e in t.blocks_detail([3, 5])}
    assert detail[3]["state_name"] == "BLOCK_SEALED_LOAD"
    assert detail[3]["next_offset"] == 16
    assert detail[3]["source"] == "kv_load"


def test_on_block_load_num_tokens_partial_last():
    """PD/prefix load: last block SLOT_PARTIAL when num_tokens % bs != 0."""
    kv_audit.configure(enabled=True, block_size=8)
    kv_audit.on_block_load([1, 2], tag="pd_recv", num_tokens=10)
    t = KvBlockMetaTracker.get()
    assert t.block_state(1) == BlockState.BLOCK_SEALED_LOAD
    assert t.block_state(2) == BlockState.SLOT_PARTIAL
    d = {e["block_id"]: e for e in t.blocks_detail([1, 2])}
    assert d[1]["next_offset"] == 8
    assert d[2]["next_offset"] == 2
    # Continuation at offset 2 is normal sequential write — no kv_state.
    v = t.apply_slot_writes([(2 * 8 + 2, 99)], block_size=8)
    assert not v.state
    assert not v.order
    assert t.slot_token(2 * 8 + 2) == 99


def test_on_pd_recv_load_uses_first_group():
    kv_audit.configure(enabled=True, block_size=4)
    kv_audit.on_pd_recv_load([[7, 8], [70, 80]], num_tokens=5, tag="pd_recv")
    t = KvBlockMetaTracker.get()
    assert t.block_state(7) == BlockState.BLOCK_SEALED_LOAD
    assert t.block_state(8) == BlockState.SLOT_PARTIAL
    assert t.blocks_detail([8])[0]["next_offset"] == 1
    assert t.block_state(70) == BlockState.UNKNOWN


def test_on_write_from_slot_mapping_partial():
    kv_audit.configure(enabled=True, block_size=16)
    slots = torch.tensor([16, 17, -1, 32], dtype=torch.int64)
    kv_audit.on_write_from_slot_mapping(slots, tag="reshape_and_cache")
    t = KvBlockMetaTracker.get()
    # Partial scatter → SLOT_PARTIAL, not whole-block seal.
    assert t.block_state(1) == BlockState.SLOT_PARTIAL
    assert t.block_state(2) == BlockState.SLOT_PARTIAL
    assert t.block_state(0) == BlockState.UNKNOWN
    assert t.slot_token(16) is None  # token stamped later at note_kv
    assert t.blocks_detail([1])[0]["next_offset"] == 2


def test_merge_slot_writes_stamps_after_reshape():
    t = KvBlockMetaTracker.get()
    assert not t.apply_slot_writes([(16, None), (17, None)], block_size=16)
    assert not t.merge_slot_writes([(16, 100), (17, 101)], block_size=16)
    assert t.slot_token(16) == 100
    assert t.slot_token(17) == 101
    assert t.block_state(1) == BlockState.SLOT_PARTIAL


def test_check_and_fill_stamps_tokenless_advanced_slots():
    t = KvBlockMetaTracker.get()
    t.apply_slot_writes([(36, None), (37, None)], block_size=4)
    mismatches, unverified = t.check_and_fill(
        [10, 11, 12, 13],
        block_ids=[9],
        block_size=4,
        end_pos=4,
    )
    assert mismatches == []
    assert unverified == 4
    assert t.slot_token(36) == 10
    assert t.slot_token(37) == 11
    assert t.slot_token(38) == 12
    assert t.slot_token(39) == 13
    assert t.block_state(9) == BlockState.BLOCK_SEALED


def test_invalidate_clears_meta_and_slots():
    kv_audit.configure(enabled=True, block_size=4)
    t = KvBlockMetaTracker.get()
    t.apply_slot_writes([(8, 10), (9, 11)], block_size=4)
    kv_audit.on_invalidate([2], reason="zero")
    assert t.block_state(2) == BlockState.UNKNOWN
    assert t.slots_detail([8, 9]) == []


def test_apply_slot_writes_sequential_and_seal():
    t = KvBlockMetaTracker.get()
    assert not t.apply_slot_writes([(36, 100), (37, 101), (38, 102), (39, 103)], block_size=4)
    assert t.block_state(9) == BlockState.BLOCK_SEALED_FILL
    assert t.slot_token(37) == 101


def test_apply_slot_writes_order_violation_skipped():
    t = KvBlockMetaTracker.get()
    v = t.apply_slot_writes([(36, 100), (38, 102)], block_size=4)
    assert len(v.order) == 1
    assert v.order[0].violation == "gap"
    assert t.slot_token(36) is None


def test_sealed_degrade_on_rewrite_offset_zero():
    t = KvBlockMetaTracker.get()
    t.on_block_load([9], block_size=4, source="kv_load")
    v = t.apply_slot_writes([(36, 200)], block_size=4)
    assert not v.state
    assert not v.order
    assert t.block_state(9) == BlockState.SLOT_PARTIAL
    assert t.slot_token(36) == 200
    assert t.slot_token(37) is None


def test_sealed_nonzero_rewrite_alerts_and_accounts_offsets():
    t = KvBlockMetaTracker.get()
    t.on_block_load([9], block_size=4, source="kv_load")
    assert t.block_state(9) == BlockState.BLOCK_SEALED_LOAD
    v = t.apply_slot_writes([(38, 201), (39, 202)], block_size=4)
    assert len(v.state) == 1
    s = v.state[0]
    assert s.violation == "sealed_nonzero_rewrite"
    assert s.block_id == 9
    assert s.offsets == (2, 3)
    assert s.prev_source == "kv_load"
    assert s.prev_state_name == "BLOCK_SEALED_LOAD"
    assert s.token_ids == (201, 202)
    assert not v.order
    assert t.slot_token(38) == 201
    assert t.slot_token(39) == 202
    assert t.blocks_detail([9])[0]["next_offset"] == 4
    # Mid-continue after load (larger block): alert by default, then account.
    KvBlockMetaTracker.reset_for_tests()
    t = KvBlockMetaTracker.get()
    t.on_block_load([9], block_size=8, source="kv_load")
    v = t.apply_slot_writes([(9 * 8 + 2, 10)], block_size=8)
    assert v.state and not v.order
    assert t.block_state(9) == BlockState.SLOT_PARTIAL
    assert t.slot_token(9 * 8 + 2) == 10
    assert t.blocks_detail([9])[0]["next_offset"] == 3
    assert not t.apply_slot_writes([(9 * 8 + 3, 11)], block_size=8)
    assert t.slot_token(9 * 8 + 3) == 11
    assert t.blocks_detail([9])[0]["next_offset"] == 4


def test_sealed_fill_nonzero_alerts_and_skips():
    t = KvBlockMetaTracker.get()
    assert not t.apply_slot_writes(
        [(36, 100), (37, 101), (38, 102), (39, 103)],
        block_size=4,
    )
    assert t.block_state(9) == BlockState.BLOCK_SEALED_FILL
    v = t.apply_slot_writes([(38, 201)], block_size=4)
    assert len(v.state) == 1
    assert v.state[0].violation == "sealed_fill_nonzero_rewrite"
    assert v.state[0].prev_state_name == "BLOCK_SEALED_FILL"
    # Skip apply: tokens / state unchanged.
    assert t.slot_token(38) == 102
    assert t.block_state(9) == BlockState.BLOCK_SEALED_FILL


def test_on_block_copy_clones_state_and_tokens():
    kv_audit.configure(enabled=True, block_size=4)
    t = KvBlockMetaTracker.get()
    assert not t.apply_slot_writes([(8, 10), (9, 11)], block_size=4)
    assert t.block_state(2) == BlockState.SLOT_PARTIAL
    kv_audit.on_block_copies([(2, 5)], block_size=4)
    assert t.block_state(5) == BlockState.SLOT_PARTIAL
    assert t.blocks_detail([5])[0]["next_offset"] == 2
    assert t.slot_token(20) == 10
    assert t.slot_token(21) == 11
    assert t.slot_token(8) == 10  # src untouched


def test_sealed_nonzero_silent_when_alert_disabled():
    """output_len==0 path: degrade+account without sealed_nonzero finding."""
    t = KvBlockMetaTracker.get()
    t.on_block_load([9], block_size=8, source="kv_load")
    v = t.apply_slot_writes(
        [(9 * 8 + 2, 10)],
        block_size=8,
        alert_sealed_nonzero=False,
    )
    assert not v.state
    assert not v.order
    assert t.block_state(9) == BlockState.SLOT_PARTIAL
    assert t.slot_token(9 * 8 + 2) == 10
    assert t.blocks_detail([9])[0]["next_offset"] == 3


def test_check_and_fill_mismatch_and_unverified():
    t = KvBlockMetaTracker.get()
    t.apply_slot_writes([(36, 100), (37, 101)], block_size=4)
    mismatches, unverified = t.check_and_fill(
        [100, 999, 102, 103],
        block_ids=[9],
        block_size=4,
        end_pos=4,
    )
    assert len(mismatches) == 1
    assert mismatches[0]["pos"] == 1
    assert unverified >= 2


def test_check_pad_slots_logs_on_violation(caplog):
    kv_audit.configure(enabled=True, pad_check=True)
    slots = torch.tensor([-1, -1, 5], dtype=torch.int64)
    with caplog.at_level(logging.ERROR):
        kv_audit.check_pad_slots(slots, where="dummy_run")
    assert any("pad violation" in r.message for r in caplog.records)


def test_check_pad_slots_ok_when_all_negative(caplog):
    kv_audit.configure(enabled=True, pad_check=True)
    slots = torch.tensor([-1, -1], dtype=torch.int64)
    with caplog.at_level(logging.ERROR):
        kv_audit.check_pad_slots(slots, where="dummy_run")
    assert not any("pad violation" in r.message for r in caplog.records)


def test_report_kv_audit_auto_on_with_slot_invariants(tmp_path, monkeypatch):
    from vllm_ascend.runtime_config.config import RuntimeConfig

    monkeypatch.chdir(tmp_path)
    cfg = RuntimeConfig(config_path=tmp_path / "runtime" / "config" / "runtime_config.json")
    assert cfg.report_kv_audit() is False
    cfg._data["invariant"]["slot_consistency"]["enabled"] = True
    cfg._invalidate_hot_path_gates()
    assert cfg.report_kv_audit() is True
    cfg._data["invariant"]["slot_consistency"]["enabled"] = False
    cfg._data["invariant"]["kv_slot_order"]["enabled"] = True
    cfg._invalidate_hot_path_gates()
    assert cfg.report_kv_audit() is True
    cfg._data["invariant"]["kv_slot_order"]["enabled"] = False
    cfg._data["report"]["block_state"] = True
    cfg._invalidate_hot_path_gates()
    assert cfg.report_kv_audit() is True


def test_sync_clears_tracker_when_kv_meta_disabled(tmp_path, monkeypatch):
    from vllm_ascend.runtime_config.config import RuntimeConfig
    from vllm_ascend.runtime_guard.invariant.slot_consistency import SlotConsistencyState
    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor

    monkeypatch.chdir(tmp_path)
    t = KvBlockMetaTracker.get()
    t.apply_slot_writes([(0, 11), (1, 12)], block_size=4)
    assert t.slot_token(0) == 11

    p = object.__new__(RuntimeGuardProcessor)
    p.runner = type("R", (), {"block_size": 4, "vllm_config": None})()
    p.runtime_config = RuntimeConfig(
        config_path=tmp_path / "runtime" / "config" / "runtime_config.json"
    )
    p._slot_consistency = SlotConsistencyState()
    p._slot_consistency._checked_first.add("req-a")

    # All consumers off → clear ledger + first-check set.
    p._sync_kv_audit()
    assert t.slot_token(0) is None
    assert t.slot_token(1) is None
    assert t.block_state(0) == BlockState.UNKNOWN
    assert p._slot_consistency._checked_first == set()

    # Keep block_state on → do not clear.
    p.runtime_config._data["report"]["block_state"] = True
    p.runtime_config._invalidate_hot_path_gates()
    t.apply_slot_writes([(4, None)], block_size=4)
    assert t.block_state(1) == BlockState.SLOT_PARTIAL
    p._sync_kv_audit()
    assert t.block_state(1) == BlockState.SLOT_PARTIAL

    # Turn block_state off → clear again.
    p.runtime_config._data["report"]["block_state"] = False
    p.runtime_config._invalidate_hot_path_gates()
    p._sync_kv_audit()
    assert t.block_state(1) == BlockState.UNKNOWN
