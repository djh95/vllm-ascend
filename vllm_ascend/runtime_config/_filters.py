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
"""Schema validation/normalization for ``input_filter.filters`` configs.

Owned by runtime_config so the config subsystem never imports from
runtime_guard; the guard side consumes normalized filter configs only.
"""

from __future__ import annotations

from typing import Any

MODE_INCLUDE = "include"
MODE_EXCLUDE = "exclude"
_VALID_MODES = frozenset({MODE_INCLUDE, MODE_EXCLUDE})

LENGTH_OPS = frozenset({"gt", "gte", "lt", "lte", "eq", "between"})
CONTAINS_MATCH = frozenset({"any", "subsequence"})


def _parse_mode(raw: Any, *, index: int) -> str:
    mode = str(raw if raw is not None else MODE_INCLUDE).lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"input_filter.filters[{index}].mode must be '{MODE_INCLUDE}' or '{MODE_EXCLUDE}', got {raw!r}"
        )
    return mode


def _parse_int_list(raw: Any, *, label: str) -> list[int]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list of ints")
    try:
        return [int(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} entries must be ints") from exc


def _parse_prefix_lists(raw: Any, *, label: str) -> list[list[int]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list of int lists")
    out: list[list[int]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"{label}[{i}] must be a list of ints, got {type(item).__name__}")
        try:
            out.append([int(x) for x in item])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{i}] entries must be ints") from exc
    return out


def normalize_input_filter_configs(configs: Any) -> list[dict[str, Any]]:
    """Validate / normalize ``input_filter.filters`` JSON list into filter configs."""
    if configs is None:
        return []
    if not isinstance(configs, list):
        raise ValueError("input_filter.filters must be a list of filter objects")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(configs):
        if not isinstance(item, dict):
            raise ValueError(f"input_filter.filters[{i}] must be an object")
        ftype = str(item.get("type", "")).strip()
        if not ftype:
            raise ValueError(f"input_filter.filters[{i}].type is required")
        mode = _parse_mode(item.get("mode"), index=i)
        normalized: dict[str, Any] = {"type": ftype, "mode": mode}
        if ftype in ("input_token_id_prefix", "prefix"):
            prefixes = _parse_prefix_lists(
                item.get("prefixes", []),
                label=f"input_filter.filters[{i}].prefixes",
            )
            normalized["type"] = "input_token_id_prefix"
            normalized["prefixes"] = prefixes
        elif ftype in ("prompt_length", "length"):
            op = str(item.get("op", "eq")).lower()
            if op not in LENGTH_OPS:
                raise ValueError(f"input_filter.filters[{i}].op must be one of {sorted(LENGTH_OPS)}, got {op!r}")
            normalized["type"] = "prompt_length"
            normalized["op"] = op
            if op == "between":
                if "min" not in item and "max" not in item:
                    raise ValueError(f"input_filter.filters[{i}] between requires min and/or max")
                if "min" in item and item["min"] is not None:
                    normalized["min"] = int(item["min"])
                if "max" in item and item["max"] is not None:
                    normalized["max"] = int(item["max"])
                lo = normalized.get("min", 0)
                hi = normalized.get("max", lo)
                if hi < lo:
                    raise ValueError(f"input_filter.filters[{i}] between max < min")
            else:
                if "value" not in item:
                    raise ValueError(f"input_filter.filters[{i}] op={op} requires value")
                normalized["value"] = int(item["value"])
        elif ftype in ("prompt_contains_token_ids", "contains_token_ids", "contains"):
            token_ids = _parse_int_list(
                item.get("token_ids", []),
                label=f"input_filter.filters[{i}].token_ids",
            )
            match = str(item.get("match", "any")).lower()
            if match not in CONTAINS_MATCH:
                raise ValueError(f"input_filter.filters[{i}].match must be 'any' or 'subsequence', got {match!r}")
            normalized["type"] = "prompt_contains_token_ids"
            normalized["token_ids"] = token_ids
            normalized["match"] = match
        else:
            raise ValueError(
                f"input_filter.filters[{i}].type unsupported: {ftype!r} "
                "(supported: input_token_id_prefix, prompt_length, prompt_contains_token_ids)"
            )
        out.append(normalized)
    return out
