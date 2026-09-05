"""Unified v1/v2 pre-sample wrap + v2 output proxy + sample-phase wiring tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.runtime_guard.runner_hooks import (
    check_before_sample_from_batch,
    wrap_compute_logits_for_pre_sample,
)
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor, SamplePhaseResult


class _FakeDfx:
    def __init__(self, runner=None):
        self.runner = runner
        self.calls = []

    def check_before_sample(self, **kwargs):
        self.calls.append(kwargs)


class _FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def compute_logits(self, hidden_states):
        self.calls += 1
        return self.logits


def _make_runner(logits, dfx, scheduler_output=None):
    model = _FakeModel(logits)
    return SimpleNamespace(model=model, runtime_guard=dfx, _rg_scheduler_output=scheduler_output), model


def test_wrap_fires_once_for_two_compute_logits_calls():
    dfx = _FakeDfx()
    runner, model = _make_runner("logits-tensor", dfx, scheduler_output="so")
    with wrap_compute_logits_for_pre_sample(runner, "batch"):
        assert model.compute_logits("h") == "logits-tensor"
        assert model.compute_logits("h") == "logits-tensor"
    assert model.calls == 2
    assert len(dfx.calls) == 1  # once-fire: v2 samples + prompt logprobs both call it
    assert dfx.calls[0]["logits"] == "logits-tensor"
    assert dfx.calls[0]["scheduler_output"] == "so"


def test_wrap_restores_compute_logits():
    dfx = _FakeDfx()
    runner, model = _make_runner("L", dfx)
    original = model.compute_logits
    with wrap_compute_logits_for_pre_sample(runner, "batch"):
        pass
    assert model.compute_logits == original
    assert not hasattr(model, "__dict__") or "compute_logits" not in model.__dict__ or model.__dict__["compute_logits"] is original


def test_from_batch_falls_back_to_runner_positions_and_logits_indices():
    runner = SimpleNamespace(_rg_positions="pos-t", logits_indices=[1, 2])
    dfx = _FakeDfx(runner=runner)
    check_before_sample_from_batch(dfx, "L", SimpleNamespace(), scheduler_output=None)
    call = dfx.calls[0]
    assert call["positions"] == "pos-t"
    assert call["logits_indices"] == [1, 2]


def test_from_batch_prefers_input_batch_fields():
    runner = SimpleNamespace(_rg_positions="pos-runner", logits_indices=[9])
    dfx = _FakeDfx(runner=runner)
    batch = SimpleNamespace(positions="pos-batch", logits_indices=[7], num_tokens=5)
    check_before_sample_from_batch(dfx, "L", batch, scheduler_output=None)
    call = dfx.calls[0]
    assert call["positions"] == "pos-batch"
    assert call["logits_indices"] == [7]
    assert call["total_scheduled_tokens"] == 5  # input_batch.num_tokens fallback


def test_from_batch_total_from_scheduler_output():
    dfx = _FakeDfx()
    so = SimpleNamespace(total_num_scheduled_tokens=17)
    check_before_sample_from_batch(dfx, "L", SimpleNamespace(), scheduler_output=so)
    assert dfx.calls[0]["total_scheduled_tokens"] == 17


def test_run_sample_phase_invokes_ensure_logprobs_and_spec_hooks():
    """Production golden path: ensure_logprobs + check_after_spec run via orchestrator."""
    RuntimeGuardProcessor.reset_for_tests()
    runner = MagicMock()
    runner.ascend_config = SimpleNamespace(
        runtime_config=None,
        runtime_config_reload_interval=0,
    )
    # Minimal bind may need more; use bare processor pattern from review tests.
    from tests.ut.runtime_guard.test_review_regressions import _bare_processor

    p = _bare_processor()
    calls: list[str] = []

    def ensure():
        calls.append("ensure_logprobs")

    def after_spec(*_a, **_k):
        calls.append("check_after_spec")

    p.ensure_logprobs_for_detection = ensure  # type: ignore[method-assign]
    p.check_after_spec = after_spec  # type: ignore[method-assign]
    p.should_check_after_spec = lambda: True  # type: ignore[method-assign]
    p.note_kv_block_writes = lambda *a, **k: calls.append("note_kv")  # type: ignore[method-assign]
    p.mark_finished = lambda *a, **k: calls.append("mark_finished")  # type: ignore[method-assign]
    p.record_sample_waves = lambda *a, **k: calls.append("waves")  # type: ignore[method-assign]
    p.check_after_sample = lambda *a, **k: calls.append("after_sample")  # type: ignore[method-assign]

    def sample_fn():
        calls.append("sample")
        return SamplePhaseResult(
            scheduler_output=None,
            input_batch=None,
            model_runner_output=None,
            sampler_output=SimpleNamespace(sampled_token_ids=[1]),
            valid_sampled_token_ids=[1],
            logprobs_lists=None,
            req_ids_output_copy=["r1"],
            invalid_req_indices=None,
            finished_req_ids=None,
        )

    p.run_sample_phase(
        sample_fn=sample_fn,
        speculative_config=object(),
        need_accepted_tokens=False,
        use_async=False,
        accepted_token_nums_fn=lambda _r: [1],
    )
    assert calls[0] == "ensure_logprobs"
    assert "sample" in calls
    assert "check_after_spec" in calls
    assert "after_sample" in calls
    # Spec check runs before wave stamp / after_sample in the orchestrator.
    assert calls.index("check_after_spec") < calls.index("after_sample")


def test_runners_call_run_sample_phase():
    """Source contract: v1/v2 sample_tokens must invoke the orchestrator."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "vllm_ascend" / "worker"
    v1 = (root / "model_runner_v1.py").read_text(encoding="utf-8")
    v2 = (root / "v2" / "model_runner.py").read_text(encoding="utf-8")
    assert "run_sample_phase(" in v1
    assert "run_sample_phase(" in v2
    assert "ensure_logprobs_for_detection" in v1 or "run_sample_phase" in v1
    assert "_rg_spec_num_sampled" in v2
