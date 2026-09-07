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

"""Runtime guard shared types and constants.

- ``ILL_TYPE_*``: anomaly category codes (legacy ILLDetector alignment).
"""

from __future__ import annotations

# Align with msprobe response_anomaly ILLDetector ill_type codes.
ILL_TYPE_NONE = 0
ILL_TYPE_RARE = 1
ILL_TYPE_GARBLED = 2
ILL_TYPE_REPEAT = 3
ILL_TYPE_NAN = 4

ILL_TYPE_NAME: dict[int, str] = {
    ILL_TYPE_NONE: "none",
    ILL_TYPE_RARE: "rare",
    ILL_TYPE_GARBLED: "garbled",
    ILL_TYPE_REPEAT: "repetition",
    ILL_TYPE_NAN: "nan",
}
