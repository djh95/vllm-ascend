#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

"""JSONC loading (``//`` / ``/* */`` comments + trailing commas).

Extracted from config.py so both the hot-reload reader and external tooling
can parse the shipped ``.jsonc`` template the same way.
"""

from __future__ import annotations

import json
from typing import Any


def _strip_jsonc(text: str) -> str:
    """String-aware removal of comments and commas directly preceding ``}`` / ``]``.

    Trailing commas may sit before a comment that itself precedes ``}`` / ``]``
    (e.g. ``{ "a": 1, /* note */ }``). Lookahead therefore skips both whitespace
    and comments before deciding whether to drop the comma.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False

    def _skip_ws_and_comments(start: int) -> int:
        j = start
        while j < n:
            if text[j] in " \t\r\n":
                j += 1
                continue
            if text[j] == "/" and j + 1 < n and text[j + 1] == "/":
                j += 2
                while j < n and text[j] != "\n":
                    j += 1
                continue
            if text[j] == "/" and j + 1 < n and text[j + 1] == "*":
                j += 2
                while j < n and not (text[j] == "*" and j + 1 < n and text[j + 1] == "/"):
                    j += 1
                j += 2
                continue
            break
        return j

    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == ",":
            j = _skip_ws_and_comments(i + 1)
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def loads_jsonc(text: str) -> Any:
    """``json.loads`` that also accepts ``//`` / ``/* */`` comments and trailing commas.

    Keeps the shipped ``.jsonc`` example template loadable as-is.
    """
    return json.loads(_strip_jsonc(text))
