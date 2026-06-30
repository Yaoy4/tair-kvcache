"""Phase 2b.4a — tests for the prefill-batch hook glue.

`admit_prefill_batch_latency(backend, states, now)` is the small helper the
SGLang hook calls when a PD backend is active and an extend batch arrives.
It now calls ``backend.try_admit_prefill_batch`` as a single unit so the
predictor receives the sum of all request input lengths — mirroring how a
real prefill node processes a batch as one forward pass.
"""
import pytest

from hisim.simulation.pd_runtime import admit_prefill_batch_latency
from hisim.simulation.pd_types import PDRequestState, RequestPhase


class _StubBackend:
    """Records the try_admit_prefill_batch call and returns a programmable end time."""

    def __init__(self, end_time):
        self._end_time = end_time
        self.calls = []  # list of (rids, total_tokens, now) per call

    def try_admit_prefill_batch(self, reqs, now):
        total_tokens = sum(r.input_length for r in reqs)
        self.calls.append(([r.rid for r in reqs], total_tokens, now))
        end_t = self._end_time
        # mimic real BackendA: set phase + timestamps on all requests
        for req in reqs:
            req.phase = RequestPhase.RUNNING_PREFILL
            req.prefill_start_time = now
            req.prefill_end_time = end_t
        return 0, end_t


def _state(rid, input_len=128):
    return PDRequestState(
        rid=rid,
        arrival_time=0.0,
        phase=RequestPhase.WAITING_PREFILL,
        input_length=input_len,
        output_length=4,
    )


def test_admit_prefill_batch_latency_returns_end_minus_now():
    be = _StubBackend(end_time=0.0025)
    states = [_state(f"r{i}") for i in range(3)]
    lat = admit_prefill_batch_latency(be, states, now=0.0)
    assert lat == pytest.approx(0.0025)


def test_admit_prefill_batch_latency_uses_relative_clock():
    be = _StubBackend(end_time=1.010)
    states = [_state("a"), _state("b")]
    lat = admit_prefill_batch_latency(be, states, now=1.0)
    assert lat == pytest.approx(0.010)


def test_admit_prefill_batch_latency_empty_returns_zero():
    be = _StubBackend(end_time=0.0)
    assert admit_prefill_batch_latency(be, [], now=42.0) == 0.0
    assert be.calls == []


def test_admit_prefill_batch_latency_marks_all_requests_running():
    be = _StubBackend(end_time=0.002)
    states = [_state("a"), _state("b")]
    admit_prefill_batch_latency(be, states, now=0.5)
    assert all(s.phase == RequestPhase.RUNNING_PREFILL for s in states)
    assert all(s.prefill_start_time == 0.5 for s in states)


def test_admit_prefill_batch_latency_all_requests_share_end_time():
    """All requests in a batch must receive the same prefill_end_time."""
    be = _StubBackend(end_time=1.007)
    states = [_state("a", input_len=100), _state("b", input_len=200), _state("c", input_len=300)]
    admit_prefill_batch_latency(be, states, now=1.0)
    # All three share the same end_time (batch-level, not per-request max)
    assert all(s.prefill_end_time == pytest.approx(1.007) for s in states)


def test_admit_prefill_batch_latency_issues_single_batch_call():
    """The backend must be called exactly once regardless of batch size."""
    be = _StubBackend(end_time=0.005)
    states = [_state(f"r{i}", input_len=50 * (i + 1)) for i in range(5)]
    admit_prefill_batch_latency(be, states, now=0.0)
    assert len(be.calls) == 1
    rids, total_tokens, now_arg = be.calls[0]
    assert rids == [s.rid for s in states]
    assert total_tokens == sum(s.input_length for s in states)
    assert now_arg == 0.0
