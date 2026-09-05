# runtime_guard live perf isolation (NPU)

UT microbenches in `tests/ut/runtime_guard/test_hot_path_overhead.py` are **not**
a substitute for these experiments.

## Experiments (must run on Ascend)

| ID | Compare | Claim | How |
|----|---------|-------|-----|
| E1 | **Without patch** (checkout parent / main without `RuntimeGuardProcessor.bind`) vs **with patch, no `--additional-config`** | TPS / temp=0 tokens equal within noise | Two builds; same `bench.sh` label `Aprime` vs `A` |
| E2 | Patch + `runtime_config_reload_interval=0`, detectors off, dump off vs E1-A | ≈ equal | Config B' |
| E3 | Patch + reload&gt;0, detectors/dump off vs E2 | ≤~1–2% TPS drop typical | Config B; hot-reload only |
| E4 | temp=0 same prompt: HTTP `content` / token ids across E1–E3 | Bit/string equal (trim) | Functional isolation |

Use rotation and stats from `/Users/djh/code/skills/test/dfx-perf-bench/SKILL.md`
and `dfx-e2e-test/scripts/bench.sh` (point `BASE_URL` / model at your case).

## Pass bar (suggested)

- `mean_tps(E2) / mean_tps(E1) ≥ 0.98` (or within that SKU’s measured noise)
- `mean_tps(E3) / mean_tps(E2) ≥ 0.98`
- E4: outputs equal

## Cannot automate here

No Ascend NPU in the agent environment → E1–E4 are **manual / CI-NPU jobs**,
listed so they are not mistaken for covered by UT.
