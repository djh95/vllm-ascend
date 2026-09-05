# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for native ``dump_kv`` analysis scripts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return data


def report_detail(report: dict[str, Any]) -> dict[str, Any]:
    detail = report.get("detail")
    return detail if isinstance(detail, dict) else {}


def token_n(detail: dict[str, Any]) -> tuple[int, int, int]:
    """Prefer id lists; fall back to *_token_count when lists absent/empty."""
    prompt_ids = detail.get("prompt_token_ids")
    output_ids = detail.get("output_token_ids")
    if isinstance(prompt_ids, list) and prompt_ids:
        n_prompt = len(prompt_ids)
    else:
        n_prompt = int(detail.get("prompt_token_count") or 0)
    if isinstance(output_ids, list) and output_ids:
        n_output = len(output_ids)
    else:
        n_output = int(detail.get("output_token_count") or 0)
    return n_prompt, n_output, n_prompt + n_output


def token_id_labels(detail: dict[str, Any], n_total: int) -> tuple[list[Any], int | None]:
    """Return (labels_per_position, n_prompt_or_None)."""
    prompt_ids = detail.get("prompt_token_ids")
    output_ids = detail.get("output_token_ids")
    if isinstance(prompt_ids, list) and isinstance(output_ids, list) and (prompt_ids or output_ids):
        labels = list(prompt_ids) + list(output_ids)
        return labels[:n_total] if n_total else labels, len(prompt_ids)
    return list(range(n_total)), None


def block_ids_from_detail(detail: dict[str, Any], report: dict[str, Any] | None = None) -> list[int]:
    raw = detail.get("block_ids")
    if raw is None and report is not None:
        raw = report.get("block_ids")
    if not isinstance(raw, list):
        return []
    return [int(x) for x in raw]


def assert_block_capacity(block_ids: list[int], n_tokens: int, block_size: int) -> None:
    if n_tokens <= 0:
        return
    if not block_ids:
        raise ValueError("block_ids empty; cannot map tokens to KV slots")
    cap = len(block_ids) * block_size
    if cap < n_tokens:
        raise ValueError(f"block capacity {cap} < num_tokens {n_tokens} (blocks={len(block_ids)} size={block_size})")


@dataclass
class NativeLayerDump:
    path: Path
    layer: str
    req_id: str
    block_ids: list[int]
    dump_all_blocks: bool
    source: str
    tensor: Any  # torch.Tensor


def load_native_pt(path: Path) -> NativeLayerDump:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "tensor" not in obj:
        raise ValueError(f"{path}: expected dump_kv dict with 'tensor'")
    return NativeLayerDump(
        path=path,
        layer=str(obj.get("layer") or path.stem),
        req_id=str(obj.get("req_id") or ""),
        block_ids=[int(x) for x in (obj.get("block_ids") or [])],
        dump_all_blocks=bool(obj.get("dump_all_blocks")),
        source=str(obj.get("source") or ""),
        tensor=obj["tensor"],
    )


def load_kv_dir(kv_dir: Path) -> dict[str, NativeLayerDump]:
    """Load all ``*.pt`` under a request dump dir, keyed by ``layer`` name."""
    if not kv_dir.is_dir():
        raise FileNotFoundError(f"kv dir missing: {kv_dir}")
    out: dict[str, NativeLayerDump] = {}
    for path in sorted(kv_dir.glob("*.pt")):
        dump = load_native_pt(path)
        # Last write wins if duplicate layer names.
        out[dump.layer] = dump
    if not out:
        raise FileNotFoundError(f"no .pt files under {kv_dir}")
    return out


def cosine(a: Any, b: Any) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    na, nb = a.norm(), b.norm()
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def cos_is_clean(c: float, thresh: float) -> bool:
    """True when similarity is at/above threshold (NaN counts as divergent)."""
    return not math.isnan(c) and c >= thresh


def cos_is_bad(c: float, thresh: float) -> bool:
    return not cos_is_clean(c, thresh)


def _cos_rank_key(c: float) -> float:
    """Sort key so NaN ranks worse than any finite cosine (Python ``min`` skips NaN otherwise)."""
    return float("-inf") if math.isnan(c) else c


def argmin_layer_cos(layer_cos: dict[str, float]) -> tuple[str, float]:
    """Return ``(layer, cos)`` with the worst cosine; NaN always wins as worst."""
    if not layer_cos:
        raise ValueError("layer_cos is empty")
    name = min(layer_cos, key=lambda k: _cos_rank_key(layer_cos[k]))
    return name, layer_cos[name]


def maxdiff(a: Any, b: Any) -> float:
    return float((a.double() - b.double()).abs().max())


def gather_token_rows(
    dump: NativeLayerDump,
    *,
    n_tokens: int,
    block_size: int,
    head: int | None = 0,
) -> Any:
    """Return tensor shaped ``(N, D)`` for the first ``n_tokens`` sequence positions.

    Native ``dump_kv`` with ``dump_all_blocks=false`` concatenates selected blocks
    along dim0 → layout ``[len(block_ids)*block_size, ...]`` in block_ids order,
    assuming the sequence starts at offset 0 of ``block_ids[0]``.

    With ``dump_all_blocks=true``, gathers via ``block_ids`` from the full cache.
    """
    import torch

    t = dump.tensor
    if not hasattr(t, "dim"):
        raise TypeError(f"layer={dump.layer}: not a tensor")

    if dump.dump_all_blocks:
        block_ids = dump.block_ids
        assert_block_capacity(block_ids, n_tokens, block_size)
        rows = []
        for i in range(n_tokens):
            blk = block_ids[i // block_size]
            off = i % block_size
            row = t[int(blk), off]
            rows.append(row)
        stacked = torch.stack(rows, dim=0)
    else:
        # Already concatenated: [n_blocks * block_size, ...]
        if t.shape[0] < n_tokens:
            raise ValueError(
                f"layer={dump.layer}: tensor length {t.shape[0]} < num_tokens {n_tokens}"
            )
        stacked = t[:n_tokens]

    if stacked.dim() == 3:
        # [N, num_heads, head_dim]
        h = 0 if head is None else int(head)
        stacked = stacked[:, h, :]
    elif stacked.dim() > 3:
        stacked = stacked.reshape(stacked.shape[0], -1)
    return stacked.double()


def _natural_key(name: str) -> tuple:
    parts = re.split(r"(\d+)", name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def matched_layers(buggy: dict[str, NativeLayerDump], ref: dict[str, NativeLayerDump]) -> list[str]:
    names = sorted(set(buggy) & set(ref), key=_natural_key)
    if not names:
        raise ValueError(
            f"no common layer names (buggy={len(buggy)} ref={len(ref)}); "
            f"buggy sample={list(buggy)[:5]} ref sample={list(ref)[:5]}"
        )
    return names


def token_id_sequence(detail: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Return (prompt_token_ids, output_token_ids) from report detail."""
    prompt = detail.get("prompt_token_ids")
    output = detail.get("output_token_ids")
    if not isinstance(prompt, list) or not isinstance(output, list):
        raise ValueError("report detail missing prompt_token_ids/output_token_ids lists")
    return [int(x) for x in prompt], [int(x) for x in output]


def validate_matching_token_ids(
    buggy_detail: dict[str, Any],
    ref_detail: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Ensure buggy/ref reports describe the same force-fed token sequence."""
    b_prompt, b_output = token_id_sequence(buggy_detail)
    r_prompt, r_output = token_id_sequence(ref_detail)
    if b_prompt != r_prompt or b_output != r_output:
        msg = (
            "buggy/ref report token ids differ "
            f"(prompt {len(b_prompt)} vs {len(r_prompt)}, "
            f"output {len(b_output)} vs {len(r_output)})"
        )
        if strict:
            raise ValueError(msg)
        print(f"[warn] {msg}; using buggy report for token labels")


@dataclass
class TokenLayerCos:
    token_idx: int
    token_id: Any
    role: str
    min_cos: float
    min_layer: str
    layer_cos: dict[str, float]
    layer_maxdiff: dict[str, float]


@dataclass
class KvSimilarityAnalysis:
    n_tokens: int
    n_prompt: int | None
    layers: list[str]
    cos_thresh: float
    per_token: list[TokenLayerCos]
    first_bad_idx: int | None

    @property
    def first_bad(self) -> TokenLayerCos | None:
        if self.first_bad_idx is None:
            return None
        return self.per_token[self.first_bad_idx]


def token_role(i: int, n_prompt: int | None) -> str:
    if n_prompt is None:
        return "pos"
    return "prompt" if i < n_prompt else f"out[{i - n_prompt}]"


def compare_kv_dumps(
    *,
    buggy_dir: Path,
    ref_dir: Path,
    detail: dict[str, Any],
    report: dict[str, Any] | None = None,
    num_tokens: int | None = None,
    block_size: int = 128,
    head: int = 0,
    cos_thresh: float = 0.99,
) -> KvSimilarityAnalysis:
    """Per-token min-over-layers cosine; locate first token below ``cos_thresh``."""
    n_p, n_o, n_from_report = token_n(detail)
    n_tokens = num_tokens if num_tokens is not None else n_from_report
    if n_tokens <= 0:
        raise ValueError("no token count/ids in report; pass num_tokens")
    labels, n_prompt = token_id_labels(detail, n_tokens)
    block_ids = block_ids_from_detail(detail, report)

    buggy = load_kv_dir(buggy_dir)
    ref = load_kv_dir(ref_dir)
    layers = matched_layers(buggy, ref)
    sample = buggy[layers[0]]
    ids = block_ids or sample.block_ids
    if sample.dump_all_blocks:
        assert_block_capacity(ids, n_tokens, block_size)
    elif ids:
        assert_block_capacity(ids, n_tokens, block_size)

    cb = {
        name: gather_token_rows(buggy[name], n_tokens=n_tokens, block_size=block_size, head=head)
        for name in layers
    }
    cr = {
        name: gather_token_rows(ref[name], n_tokens=n_tokens, block_size=block_size, head=head)
        for name in layers
    }

    per_token: list[TokenLayerCos] = []
    first_bad_idx: int | None = None
    for i in range(n_tokens):
        layer_cos: dict[str, float] = {}
        layer_maxdiff: dict[str, float] = {}
        for name in layers:
            layer_cos[name] = cosine(cb[name][i], cr[name][i])
            layer_maxdiff[name] = maxdiff(cb[name][i], cr[name][i])
        min_layer, min_cos = argmin_layer_cos(layer_cos)
        tok = labels[i] if i < len(labels) else i
        per_token.append(
            TokenLayerCos(
                token_idx=i,
                token_id=tok,
                role=token_role(i, n_prompt),
                min_cos=min_cos,
                min_layer=min_layer,
                layer_cos=layer_cos,
                layer_maxdiff=layer_maxdiff,
            )
        )
        if first_bad_idx is None and cos_is_bad(min_cos, cos_thresh):
            first_bad_idx = i

    return KvSimilarityAnalysis(
        n_tokens=n_tokens,
        n_prompt=n_prompt,
        layers=layers,
        cos_thresh=cos_thresh,
        per_token=per_token,
        first_bad_idx=first_bad_idx,
    )


def format_kv_similarity_tables(
    analysis: KvSimilarityAnalysis,
    *,
    buggy_dir: Path,
    ref_dir: Path,
) -> str:
    """Human-readable two-table report (Table1 per-token min cos; Table2 first bad)."""
    lines: list[str] = []
    n_layers = len(analysis.layers)
    lines.append("=" * 78)
    lines.append(
        f"TABLE 1  逐 token: min-over-{n_layers}-layers cos (阈值 {analysis.cos_thresh:.2f})"
    )
    lines.append(f"         buggy={buggy_dir} ref={ref_dir}")
    lines.append("=" * 78)

    seg_start: int | None = None
    for row in analysis.per_token:
        if cos_is_clean(row.min_cos, analysis.cos_thresh):
            if seg_start is None:
                seg_start = row.token_idx
            continue
        if seg_start is not None:
            lines.append(f"token {seg_start}-{row.token_idx - 1}  clean")
            seg_start = None
        lines.append(
            f"token {row.token_idx:<4d} tok={row.token_id!s:<6} {row.role:<7} "
            f"min_cos={row.min_cos:.4f}  @{row.min_layer}"
        )
    if seg_start is not None:
        lines.append(f"token {seg_start}-{analysis.n_tokens - 1}  clean")

    if analysis.first_bad_idx is None:
        lines.append("")
        lines.append("无坏点：KV 在匹配层上全部一致 → 问题可能在采样/后处理")
        return "\n".join(lines)

    bad = analysis.first_bad
    assert bad is not None
    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"TABLE 2  第一坏点 token {bad.token_idx} "
        f"(tok={bad.token_id}, {bad.role}) 逐层 cos / maxdiff"
    )
    lines.append("=" * 78)
    lines.append(f"{'layer':<40} {'cos':<12} {'maxdiff':<12}")
    prev_ok = True
    for name in analysis.layers:
        c = bad.layer_cos[name]
        md = bad.layer_maxdiff[name]
        mark = ""
        is_bad = cos_is_bad(c, analysis.cos_thresh)
        if prev_ok and is_bad:
            mark = "  <-- 首发散层"
        if is_bad:
            prev_ok = False
        lines.append(f"{name:<40} {c:<12.4f} {md:<12.4e}{mark}")
    return "\n".join(lines)


def analysis_to_json(analysis: KvSimilarityAnalysis) -> dict[str, Any]:
    """Serialize analysis for ``--json-out``."""
    first_bad = None
    if analysis.first_bad is not None:
        fb = analysis.first_bad
        first_bad = {
            "token_idx": fb.token_idx,
            "token_id": fb.token_id,
            "role": fb.role,
            "min_cos": fb.min_cos,
            "min_layer": fb.min_layer,
            "layers": [
                {"layer": name, "cos": fb.layer_cos[name], "maxdiff": fb.layer_maxdiff[name]}
                for name in analysis.layers
            ],
        }
    return {
        "n_tokens": analysis.n_tokens,
        "n_prompt": analysis.n_prompt,
        "cos_thresh": analysis.cos_thresh,
        "layers": analysis.layers,
        "first_bad": first_bad,
        "per_token": [
            {
                "token_idx": row.token_idx,
                "token_id": row.token_id,
                "role": row.role,
                "min_cos": row.min_cos,
                "min_layer": row.min_layer,
            }
            for row in analysis.per_token
        ],
    }
