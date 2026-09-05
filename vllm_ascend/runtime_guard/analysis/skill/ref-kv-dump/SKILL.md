---
name: runtime-guard-ref-kv-dump
description: >-
  Capture a reference KV dump via runtime_guard dump_kv (same token IDs as the
  buggy report) and compare against the buggy kv_cache dir: first-divergence
  two tables and per-layer cosine. Use for token_repeat / KV corruption after
  verify_request_kv PASS. No msprobe.
---

# Reference KV via native dump_kv

## Goal

Compare **buggy** `kv_cache/<type>/<req_id>/*.pt` vs **ref** dump from a clean
run that force-feeds the same `prompt_token_ids + output_token_ids`.

## Why force-feed token IDs

1. Prefill vs decode paths differ — ref must exercise the same path class when possible.
2. Re-tokenizing plaintext ≠ report token ids.
3. Ref must not greedy-gen its own tokens — feed buggy output ids exactly.

## Produce ref dump

1. From the buggy report:

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.prepare_ref_inputs \
  --report <report_*.json> --out ref_inputs.json
```

2. Start a **clean** engine (same model / TP / mode) with `dump_kv` in
   `on_trigger` for `manual_trigger` (or arm `dump.manual_dump`).
   Requires `runtime_config_reload_interval > 0` so workers consume the arm.
3. Force-feed token ids (prefer HTTP completions with int `prompt`):

```bash
# default --feed=history → prompt + output[:-1] (leave last token for decode)
python -m vllm_ascend.runtime_guard.analysis.scripts.request_from_report \
  --report <report_*.json> \
  --url http://127.0.0.1:8000/v1/completions \
  --model <served-model-name>
```

Or use `force_feed_token_ids` from `ref_inputs.json` via Python `LLM` /
token-id API — not OpenAI string/`chat` prompts.
4. Optional 3-pass (same methodology as before, dumps via `dump_kv` each time):
   - Pass1 prefill `prompt + output[:-1]` (`--feed history`)
   - Pass2 decode `output[-1]`
   - Pass3 optional full prefill `prompt + output` (`--feed full`)
5. Collect ref dir: `runtime/report/kv_cache/<type>/<ref_req_id>/`.

## Compare (two tables)

Prefer **two reports** (buggy + ref) so token-id sequences are checked:

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.compare_kv_similarity \
  --buggy-dir <bad_kv_dir> \
  --ref-dir   <ref_kv_dir> \
  --buggy-report <buggy_report_*.json> \
  --ref-report   <ref_report_*.json> \
  --block-size 128 --cos-thresh 0.99
```

Single-report shortcut (labels/N from buggy report only):

```bash
python -m vllm_ascend.runtime_guard.analysis.scripts.locate_first_divergence \
  --buggy-dir <bad_kv_dir> \
  --ref-dir   <ref_kv_dir> \
  --report    <report_*.json> \
  --block-size 128 --cos-thresh 0.99

python -m vllm_ascend.runtime_guard.analysis.scripts.compare_per_layer \
  --buggy-dir <bad_kv_dir> \
  --ref-dir   <ref_kv_dir> \
  --report    <report_*.json>
```

**Table 1**: per-token min-cos over matched layers → first bad token.  
**Table 2**: that token’s per-layer cos / maxdiff → first divergent layer.

Assumes sequence starts at offset 0 of `block_ids[0]`. Layers matched by
payload `layer` name (intersection of both dirs), ordered by natural sort
(`layer_2` < `layer_10`) — the first-divergence layer index is trustworthy.
`block_ids` in each payload is the ids actually dumped (out-of-range ids are
dropped, not silently recorded).

Default `--head 0` compares only KV head 0 when tensors are `[N,H,D]`; a clean
head-0 cos does not prove other heads match.

**TP>1**: each rank's dump carries `tp_rank` / `num_kv_heads` — before comparing
across ranks, confirm the head slicing matches (KV heads are TP-sliced); a cos
gap may be layout, not numerics.

## Before compare

Run `runtime-guard-analysis` → `verify_request_kv` PASS on the buggy side.

## Related

- `runtime-guard-investigation` — full case orchestration
- `runtime-guard-analysis` — summarize / correlate / verify / inspect
