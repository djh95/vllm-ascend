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

"""Read paged KV cache blocks for a request (direct D2H, no msprobe)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def _iter_kv_tensors(kv_caches: Any) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    if kv_caches is None:
        return out
    if isinstance(kv_caches, dict):
        for name, val in kv_caches.items():
            if isinstance(val, torch.Tensor):
                out.append((str(name), val))
            elif isinstance(val, (list, tuple)):
                for i, t in enumerate(val):
                    if isinstance(t, torch.Tensor):
                        out.append((f"{name}[{i}]", t))
    elif isinstance(kv_caches, list):
        for i, val in enumerate(kv_caches):
            if isinstance(val, torch.Tensor):
                out.append((f"layer_{i}", val))
            elif isinstance(val, (list, tuple)):
                for j, t in enumerate(val):
                    if isinstance(t, torch.Tensor):
                        out.append((f"layer_{i}[{j}]", t))
    return out


def _slice_blocks(tensor: torch.Tensor, block_ids: list[int]) -> tuple[torch.Tensor, list[int]] | None:
    """Concatenate selected blocks along the block dimension when possible.

    Returns ``None`` when ``block_ids`` is empty so callers refuse a full-cache
    copy (empty ids must not mean "dump everything"). Returns the payload
    together with the block ids actually used (B'5c: partial out-of-range ids
    must not be recorded as if fully dumped).
    """
    if not block_ids:
        return None
    if tensor.ndim < 2:
        return tensor.detach().cpu(), list(block_ids)
    # Common paged layout: [num_blocks, block_size, ...]
    max_block = int(tensor.shape[0]) - 1
    valid = [b for b in block_ids if 0 <= int(b) <= max_block]
    if not valid:
        return None
    try:
        parts = [tensor[int(b)] for b in valid]
        if len(parts) == 1:
            sliced = parts[0]
        else:
            sliced = torch.cat(parts, dim=0)
    except Exception:
        sliced = tensor[valid]
    return sliced.detach().cpu(), valid


@dataclass
class KvDumpSnapshot:
    """CPU-side KV payload ready for async ``torch.save``."""

    path: Path
    payload: dict[str, Any]


class KvCacheReader:
    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def _kv_sources(self) -> list[tuple[str, Any]]:
        runner = self._runner
        sources: list[tuple[str, Any]] = []
        kv_dict = getattr(runner, "kv_caches", None)
        if isinstance(kv_dict, dict) and kv_dict:
            sources.append(("kv_caches", kv_dict))
        kv_list = getattr(runner, "kv_caches", None)
        if isinstance(kv_list, list) and kv_list and not sources:
            sources.append(("kv_caches", kv_list))
        return sources

    def iter_request_snapshots(
        self,
        *,
        req_id: str,
        block_ids: list[int],
        out_dir: Path,
        dump_all_blocks: bool = False,
    ) -> "Iterator[KvDumpSnapshot]":
        """Yield per-layer CPU snapshots for one request (D2H per layer).

        B'3: a generator (not a list) lets callers enqueue each layer for
        async save as soon as it lands on host — full-cache dumps no longer
        accumulate every layer in RAM before the first write.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = list(block_ids)
        produced = False
        # B'6: record which TP shard produced the dump so cross-TP comparisons
        # can validate head slicing instead of guessing.
        try:
            tp_rank = int(getattr(self._runner, "tp_rank", 0) or 0)
        except Exception:
            tp_rank = 0
        for src_name, kv_caches in self._kv_sources():
            for layer_name, tensor in _iter_kv_tensors(kv_caches):
                num_kv_heads = int(tensor.shape[-2]) if tensor.dim() >= 3 else None
                if dump_all_blocks:
                    payload_tensor = tensor.detach().cpu()
                    used_ids = ids
                    suffix = "all"
                else:
                    if not ids:
                        logger.warning(
                            "[runtime_guard dump_kv] skip layer=%s req_id=%s: empty block_ids "
                            "(refusing full-cache D2H)",
                            layer_name,
                            req_id,
                        )
                        continue
                    sliced = _slice_blocks(tensor, ids)
                    if sliced is None:
                        logger.warning(
                            "[runtime_guard dump_kv] skip layer=%s req_id=%s: no valid blocks in %s",
                            layer_name,
                            req_id,
                            ids,
                        )
                        continue
                    payload_tensor, used_ids = sliced
                    suffix = "req"
                safe_layer = layer_name.replace("/", "_")
                path = out_dir / f"{req_id}_{safe_layer}_{suffix}.pt"
                produced = True
                yield KvDumpSnapshot(
                    path=path,
                    payload={
                        "req_id": req_id,
                        "block_ids": used_ids,
                        "dump_all_blocks": dump_all_blocks,
                        "layer": layer_name,
                        "source": src_name,
                        "tp_rank": tp_rank,
                        "num_kv_heads": num_kv_heads,
                        "tensor": payload_tensor,
                    },
                )
        if not produced:
            logger.warning("[runtime_guard dump_kv] no kv tensors found req_id=%s", req_id)

    def snapshot_request_blocks(
        self,
        *,
        req_id: str,
        block_ids: list[int],
        out_dir: Path,
        dump_all_blocks: bool = False,
    ) -> list[KvDumpSnapshot]:
        """Sync D2H of KV for one request. Does not write files."""
        return list(
            self.iter_request_snapshots(
                req_id=req_id,
                block_ids=block_ids,
                out_dir=out_dir,
                dump_all_blocks=dump_all_blocks,
            )
        )

    @staticmethod
    def write_snapshots(snapshots: list[KvDumpSnapshot]) -> list[str]:
        """Async-safe: write previously snapshotted CPU tensors to disk."""
        written: list[str] = []
        for snap in snapshots:
            snap.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(snap.payload, snap.path)
            written.append(str(snap.path))
        if written:
            req_id = snapshots[0].payload.get("req_id")
            logger.info(
                "[runtime_guard dump_kv] wrote req_id=%s files=%d dir=%s",
                req_id,
                len(written),
                snapshots[0].path.parent,
            )
        return written

    def dump_request_blocks(
        self,
        *,
        req_id: str,
        block_ids: list[int],
        out_dir: Path,
        dump_all_blocks: bool = False,
    ) -> list[str]:
        """Sync snapshot + write (compat path). Prefer snapshot then async write."""
        snaps = self.snapshot_request_blocks(
            req_id=req_id,
            block_ids=block_ids,
            out_dir=out_dir,
            dump_all_blocks=dump_all_blocks,
        )
        return self.write_snapshots(snaps)
