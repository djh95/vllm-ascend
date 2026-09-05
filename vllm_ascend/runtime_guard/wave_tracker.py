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

"""Per-step wave index for incident correlation."""

from __future__ import annotations


class WaveTracker:
    def __init__(self) -> None:
        self._wave = 0
        self._sample_waves: dict[str, int] = {}

    def advance(self, *, allow_arm: bool = True) -> None:
        if allow_arm:
            self._wave += 1

    def current_wave(self) -> int:
        return self._wave

    def record_sample_waves(self, req_ids: list[str] | None) -> None:
        if not req_ids:
            return
        wave = self._wave
        for req_id in req_ids:
            if req_id:
                self._sample_waves[str(req_id)] = wave

    def take_sample_wave(self, req_id: str) -> int | None:
        return self._sample_waves.pop(str(req_id), None)

    def pending(self, req_id: str) -> bool:
        """True when a stamp is still unconsumed (async output not yet materialized)."""
        return str(req_id) in self._sample_waves

    def discard(self, req_id: str) -> None:
        self._sample_waves.pop(str(req_id), None)

    def discard_many(self, req_ids: list[str] | None) -> None:
        # B2: reap-time cleanup — finished requests must not accumulate stamps.
        if not req_ids:
            return
        for req_id in req_ids:
            self.discard(req_id)
