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

"""Small shared helpers used across runtime_guard modules."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def kv_dump_wave_dirname(wave: int | None) -> str:
    """Subdir under ``{type}/{req_id}/`` separating dumps across steps."""
    if wave is None:
        return "wave_unknown"
    return f"wave_{int(wave)}"


def is_int_list(value: Any) -> bool:
    """True when ``value`` is a non-empty ``list[int]`` (bool excluded)."""
    return (
        isinstance(value, list) and bool(value) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    )


def is_list_of_int_lists(value: Any) -> bool:
    """True when ``value`` is a non-empty list of int lists."""
    return isinstance(value, list) and bool(value) and all(is_int_list(x) for x in value)


def normalize_token_ids(token_ids: Any) -> list[int]:
    """Normalize tensor / nested tensors / sequences to ``list[int]``."""
    if token_ids is None:
        return []
    if torch.is_tensor(token_ids):
        return [int(x) for x in token_ids.tolist()]
    out: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, torch.Tensor):
            out.append(int(token_id.item()))
        else:
            out.append(int(token_id))
    return out


def filter_valid_token_ids(token_ids: Any) -> list[int]:
    """Normalize and drop async / pad placeholders (``-1``)."""
    return [tid for tid in normalize_token_ids(token_ids) if tid != -1]


def freeze_sampled_rows(req_ids: list[str] | None, sampled_rows: Any) -> list[list[int]]:
    """Host copy of this step's per-req sampled ids (no shared mutation)."""
    ids = list(req_ids or [])
    out: list[list[int]] = []
    for i, _rid in enumerate(ids):
        try:
            row = sampled_rows[i] if sampled_rows is not None else None
        except (IndexError, TypeError, KeyError):
            out.append([])
            continue
        out.append(filter_valid_token_ids(row))
    return out


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode a token-id list to text (``skip_special_tokens=False``)."""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def write_kv_dump_skipped_finished(
    dump_root: str | Path,
    *,
    req_id: str,
    incident_type: str,
    stage: str,
    rank_tag: str = "",
) -> Path | None:
    """Last-PP TP0: leave a marker when dump is skipped because req finished/reaped.

    Path: ``{dump_root}/{incident_type}/{req_id}/dump_skipped_finished.json``
    """
    if not req_id:
        return None
    out_dir = Path(dump_root) / str(incident_type or "unknown") / str(req_id)
    path = out_dir / "dump_skipped_finished.json"
    payload = {
        "reason": "finished_or_reaped",
        "req_id": str(req_id),
        "incident_type": str(incident_type or "unknown"),
        "stage": str(stage),
        "rank_tag": str(rank_tag or ""),
        "ts": time.time(),
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.warning(
            "[runtime_guard dump_kv] skipped finished/reaped req_id=%s type=%s stage=%s marker=%s",
            req_id,
            incident_type,
            stage,
            path,
        )
        return path
    except OSError as exc:
        logger.warning(
            "[runtime_guard dump_kv] failed to write skip marker req_id=%s path=%s: %s",
            req_id,
            path,
            exc,
        )
        return None


def write_kv_dump_request_info(
    dump_root: str | Path,
    *,
    req_id: str,
    incident_type: str,
    detail: dict[str, Any] | None,
    rank_tag: str = "",
    wave: int | None = None,
    block_ids: list[int] | None = None,
    tokenizer: Any | None = None,
    save_sensitive_info: bool = False,
    decode_token_ids: bool = True,
    max_prompt_token_ids: int = 1000,
    max_output_token_ids: int = 1000,
) -> Path | None:
    """Last-PP TP0: write report-like request metadata next to KV ``.pt`` shards.

    Path: ``{dump_root}/{incident_type}/{req_id}/{wave_N}/request_info.json``
    """
    if not req_id:
        return None
    from vllm_ascend.runtime_guard.report import dumps_report_json, sanitize_report_detail

    out_dir = (
        Path(dump_root)
        / str(incident_type or "unknown")
        / str(req_id)
        / kv_dump_wave_dirname(wave)
    )
    path = out_dir / "request_info.json"
    safe_detail = sanitize_report_detail(
        detail,
        save_sensitive_info=save_sensitive_info,
        max_prompt_token_ids=max_prompt_token_ids,
        max_output_token_ids=max_output_token_ids,
        decode_token_ids=decode_token_ids and save_sensitive_info,
        tokenizer=tokenizer if (decode_token_ids and save_sensitive_info) else None,
    )
    payload = {
        "ts": time.time(),
        "incident_type": str(incident_type or "unknown"),
        "req_id": str(req_id),
        "rank": str(rank_tag or ""),
        "dump_arm_wave": int(wave) if wave is not None else None,
        "block_ids": list(block_ids) if block_ids is not None else safe_detail.get("block_ids"),
        "decode_token_ids": bool(decode_token_ids and save_sensitive_info),
        "max_prompt_token_ids": int(max_prompt_token_ids),
        "max_output_token_ids": int(max_output_token_ids),
        "detail": safe_detail,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_report_json(payload, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "[runtime_guard dump_kv] request_info req_id=%s type=%s path=%s",
            req_id,
            incident_type,
            path,
        )
        return path
    except OSError as exc:
        logger.warning(
            "[runtime_guard dump_kv] failed to write request_info req_id=%s path=%s: %s",
            req_id,
            path,
            exc,
        )
        return None


def accepted_token_counts(
    sampled_token_ids: Any,
    *,
    placeholder_token_id: int = -1,
) -> Any:
    """Count accepted tokens per request from rejection-sampler output.

    Used for non-hybrid MTP / speculative paths where accepted counts are
    derived from ``PLACEHOLDER_TOKEN_ID`` padding rather than a dedicated
    ``num_accepted_tokens`` buffer.
    """
    if sampled_token_ids is None:
        return []
    if torch.is_tensor(sampled_token_ids):
        if sampled_token_ids.numel() == 0:
            return torch.zeros(sampled_token_ids.size(0), dtype=torch.int32)
        return (sampled_token_ids != placeholder_token_id).sum(dim=-1).to(dtype=torch.int32).cpu()
    counts: list[int] = []
    for row in sampled_token_ids:
        if row is None:
            counts.append(0)
            continue
        if torch.is_tensor(row):
            counts.append(int((row != placeholder_token_id).sum().item()))
        else:
            counts.append(sum(1 for t in row if t != placeholder_token_id))
    return counts


def load_model_tokenizer(runner: Any) -> Any | None:
    """Load model tokenizer via ``cached_tokenizer_from_config``.

    Returns ``None`` if runner/config missing; raises if the load itself fails.
    """
    if runner is None:
        return None
    from vllm.tokenizers import cached_tokenizer_from_config

    vllm_config = getattr(runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
    if model_config is None:
        return None
    return cached_tokenizer_from_config(model_config)

