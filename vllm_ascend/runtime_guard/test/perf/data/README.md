# Long prompt corpus for C5/C6 (1k-128k token tiers)

Source: LongBench (THUDM/LongBench, mirrored at hf-mirror.com as zai-org/LongBench),
data.zip downloaded 2026-09-20. LongBench assembles public documents for long-context
evaluation; data is for research/test use per its upstream license.

Selection: top-N longest `context` fields per source file (min 2k chars), fields
kept: source, doc_id, chars, question (truncated 200 chars), text.

- narrativeqa.jsonl  x6  (English books, ~210k chars max)
- gov_report.jsonl   x4  (English reports, ~257k chars max)
- passage_count.jsonl x3 (English essays, ~126k chars max)
- dureader.jsonl     x4  (Chinese QA, ~27k chars max)
- multifieldqa_zh.jsonl x4 (Chinese docs)

Total: 21 docs, ~2.59M chars (~600k+ tokens) -- enough to build prompts up to
128k tokens by concatenating distinct docs. Prompts are cut to EXACT token counts
with the served model tokenizer by c56_driver.py, and every request carries a
unique zero-padded sequence prefix ([c56-%06d]) so prefix caching cannot reuse
KV across requests.

Build script (reproducible): see git history of this file; selection is purely
"longest contexts per file", no content curation.
