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

"""P0 UT: token_repeat pure logic + kv dump_kv empty-block safety."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend.runtime_guard.detector.token_repeat import TokenRepeatState, push_token_repeat
from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader, _slice_blocks


def test_push_token_repeat_scores_repeats():
    st = TokenRepeatState()
    ignore: set[int] = set()
    # Unique tokens → score 0
    for tid in (1, 2, 3, 4):
        assert push_token_repeat(st, tid, window=8, ignore=ignore) == 0
    # Repeat 1 → score 1
    assert push_token_repeat(st, 1, window=8, ignore=ignore) == 1
    assert st.repeat_sum >= 1


def test_push_token_repeat_ignore_ids():
    st = TokenRepeatState()
    ignore = {0}
    assert push_token_repeat(st, 0, window=8, ignore=ignore) == 0
    assert len(st.content) == 0


def test_slice_blocks_empty_returns_none():
    t = torch.randn(4, 8, 2)
    assert _slice_blocks(t, []) is None


def test_slice_blocks_selects_requested():
    t = torch.arange(0, 4 * 8).reshape(4, 8).float()
    sliced = _slice_blocks(t, [1, 3])
    assert sliced is not None
    out, used = sliced
    assert used == [1, 3]
    assert out.shape[0] == 16  # 2 blocks * 8
    assert torch.equal(out[:8], t[1])
    assert torch.equal(out[8:], t[3])


def test_slice_blocks_partial_out_of_range_reports_used_ids():
    # B'5c: payload metadata must reflect the blocks actually dumped.
    t = torch.arange(0, 4 * 8).reshape(4, 8).float()
    sliced = _slice_blocks(t, [1, 9])  # 9 out of range
    assert sliced is not None
    out, used = sliced
    assert used == [1]
    assert out.shape[0] == 8
    assert torch.equal(out, t[1])


def test_snapshot_skips_empty_block_ids(tmp_path: Path):
    runner = SimpleNamespace(
        kv_caches={"layers.0": torch.randn(8, 16, 4)},
    )
    reader = KvCacheReader(runner)
    snaps = reader.snapshot_request_blocks(
        req_id="r1",
        block_ids=[],
        out_dir=tmp_path / "kv",
        dump_all_blocks=False,
    )
    assert snaps == []


def test_snapshot_request_blocks_writes_req_slice(tmp_path: Path):
    cache = torch.randn(4, 8, 2)
    runner = SimpleNamespace(kv_caches={"L0": cache})
    reader = KvCacheReader(runner)
    snaps = reader.snapshot_request_blocks(
        req_id="r1",
        block_ids=[0, 2],
        out_dir=tmp_path / "kv",
        dump_all_blocks=False,
    )
    assert len(snaps) == 1
    assert snaps[0].payload["dump_all_blocks"] is False
    assert snaps[0].payload["tensor"].shape[0] == 16
    written = KvCacheReader.write_snapshots(snaps)
    assert len(written) == 1
    assert Path(written[0]).is_file()
