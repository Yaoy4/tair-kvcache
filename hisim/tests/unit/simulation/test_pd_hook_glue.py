"""Phase 2b.4a — tests for the prefill-batch hook glue.

`admit_prefill_batch_latency(backend, states, now)` is the small helper the
SGLang hook calls when a PD backend is active and an extend batch arrives.
It admits every request onto the backend's prefill pool and returns the
batch's predicted latency (= max prefill end time minus `now`).
"""
import pytest

from hisim.simulation.pd_runtime import admit_prefill_batch_latency
from hisim.simulation.pd_types import PDRequestState, RequestPhase


class _StubBackend:
    """Records each try_admit_prefill call and returns a programmable end time."""

    def __init__(self, end_times):
        self._end_times = list(end_times)
        self.calls = []

    def try_admit_prefill(self, req, now):
        self.calls.append((req.rid, now))
        end_t = self._end_times.pop(0)
        # mimic real BackendA side effects on the request state
        req.phase = RequestPhase.RUNNING_PREFILL
        req.prefill_start_time = now
        return 0, end_t


def _state(rid, input_len=128):
    return PDRequestState(
        rid=rid,
        arrival_time=0.0,
        phase=RequestPhase.WAITING_PREFILL,
        input_length=input_len,
        output_length=4,
    )


def test_admit_prefill_batch_latency_returns_max_minus_now():
    be = _StubBackend(end_times=[0.001, 0.0025, 0.002])
    states = [_state(f"r{i}") for i in range(3)]
    lat = admit_prefill_batch_latency(be, states, now=0.0)
    assert lat == pytest.approx(0.0025)
    assert [c[0] for c in be.calls] == ["r0", "r1", "r2"]


def test_admit_prefill_batch_latency_uses_relative_clock():
    be = _StubBackend(end_times=[1.005, 1.010])
    states = [_state("a"), _state("b")]
    lat = admit_prefill_batch_latency(be, states, now=1.0)
    assert lat == pytest.approx(0.010)


def test_admit_prefill_batch_latency_empty_returns_zero():
    be = _StubBackend(end_times=[])
    assert admit_prefill_batch_latency(be, [], now=42.0) == 0.0


def test_admit_prefill_batch_latency_marks_each_request_running():
    be = _StubBackend(end_times=[0.001, 0.002])
    states = [_state("a"), _state("b")]
    admit_prefill_batch_latency(be, states, now=0.5)
    assert all(s.phase == RequestPhase.RUNNING_PREFILL for s in states)
    assert all(s.prefill_start_time == 0.5 for s in states)
