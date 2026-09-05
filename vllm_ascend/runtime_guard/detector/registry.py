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

"""Ordered anomaly detector registry.

Kept deliberately minimal: ``DetectorManager`` (see ``manager.py``) uses it
internally for iteration / clear-finished and holds concrete detectors as
private references.
"""

from __future__ import annotations

from collections.abc import Iterator

from vllm_ascend.runtime_guard.detector.base import AnomalyDetector


class DetectorRegistry:
    """Ordered registry keyed by ``AnomalyDetector.incident_type``."""

    def __init__(self) -> None:
        self._items: list[AnomalyDetector] = []
        self._by_type: dict[str, AnomalyDetector] = {}

    def register(self, detector: AnomalyDetector) -> AnomalyDetector:
        key = str(getattr(detector, "incident_type", "") or type(detector).__name__)
        if key in self._by_type:
            raise ValueError(f"detector incident_type already registered: {key}")
        self._items.append(detector)
        self._by_type[key] = detector
        return detector

    def get(self, incident_type: str) -> AnomalyDetector | None:
        return self._by_type.get(incident_type)

    def clear_finished(self, req_id: str) -> None:
        for det in self._items:
            det.clear_finished(req_id)

    def __iter__(self) -> Iterator[AnomalyDetector]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[AnomalyDetector]:
        return list(self._items)
