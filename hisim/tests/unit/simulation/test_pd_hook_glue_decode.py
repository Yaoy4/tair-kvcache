"""Phase 2b.4b — tests for decode-routing + KV transition hook glue."""
import pytest

from hisim.simulation.pd_runtime import (
    decode_batch_latency,
    finalize_prefill_batch,
    drain_kv_ready_and_admit_decode,
)
from hisim.simulation.pd_types import PDRequestState, RequestPhase


# ---------------------------------------------------------------------------
# decode_batch_latency
# ---------------------------------------------------------------------------


class _DecodeStubBackend:
    def __init__(self, end_time):
        self._end_time = end_time
        self.calls = []

    def try_admit_decode_batch(self, reqs, now):
        self.calls.append((len(reqs), now))
        return 0, self._end_time


def _decode_state(rid, past_kv=10):
    s = PDRequestState(
        rid=rid,
        arrival_time=0.0,
        phase=RequestPhase.RUNNING_DECODE,
        input_length=past_kv,
        output_length=4,
        current_past_kv_length=past_kv,
    )
    return s


def test_decode_batch_latency_returns_end_minus_now():
    be = _DecodeStubBackend(end_time=1.005)
    states = [_decode_state(f"r{i}") for i in range(3)]
    lat = decode_batch_latency(be, states, now=1.0)
    assert lat == pytest.approx(0.005)
    assert be.calls == [(3, 1.0)]


def test_decode_batch_latency_empty_returns_zero():
    be = _DecodeStubBackend(end_time=0.0)
    assert decode_batch_latency(be, [], now=5.0) == 0.0
    assert be.calls == []


# ---------------------------------------------------------------------------
# finalize_prefill_batch + drain_kv_ready_and_admit_decode
# ---------------------------------------------------------------------------


class _StubController:
    def __init__(self, kv_dur):
        self._kv_dur = kv_dur
        self.prefill_done_calls = []
        self.poll_calls = []
        self.admit_calls = []
        self._waiting = []  # WAITING_DECODE queue

    def compute_kv_ready_time(self, req, now):
        return now + self._kv_dur

    def compute_batch_kv_ready_time(self, total_tokens, now):
        # Stub: ignores total_tokens, returns now + kv_dur for predictable assertions
        return now + self._kv_dur

    def on_prefill_done(self, req, now, kv_ready_time):
        self.prefill_done_calls.append((req.rid, now, kv_ready_time))
        req.phase = RequestPhase.KV_TRANSIT
        req.prefill_end_time = now
        req.kv_ready_time = kv_ready_time

    def poll_kv_ready(self, now):
        self.poll_calls.append(now)
        ready = []
        # nothing to model here in stub — caller drives by hand
        return ready

    def admit_decode(self, capacity, now):
        self.admit_calls.append((capacity, now))
        out = self._waiting[:capacity]
        self._waiting = self._waiting[capacity:]
        for r in out:
            r.phase = RequestPhase.RUNNING_DECODE
            r.decode_start_time = now
            r.current_past_kv_length = r.input_length
        return out


class _StubBackendForTransition:
    def __init__(self, controller):
        self._ctrl = controller

    def controller(self):
        return self._ctrl

    # finalize_prefill_batch uses these
    def compute_kv_ready_time(self, req, now):
        return self._ctrl.compute_kv_ready_time(req, now)

    def compute_batch_kv_ready_time(self, total_tokens, now):
        return self._ctrl.compute_batch_kv_ready_time(total_tokens, now)

    def on_prefill_done(self, req, now, kv_ready_time):
        self._ctrl.on_prefill_done(req, now, kv_ready_time)


def _running_prefill_state(rid, input_len=100):
    return PDRequestState(
        rid=rid,
        arrival_time=0.0,
        phase=RequestPhase.RUNNING_PREFILL,
        input_length=input_len,
        output_length=4,
        prefill_start_time=0.0,
    )


def test_finalize_prefill_batch_calls_on_prefill_done_with_kv_ready_time():
    ctrl = _StubController(kv_dur=0.002)
    be = _StubBackendForTransition(ctrl)
    states = [_running_prefill_state("a"), _running_prefill_state("b")]
    finalize_prefill_batch(be, states, now=0.005)
    # Both requests share the same kv_ready_time computed from now=0.005
    assert ctrl.prefill_done_calls == [
        ("a", 0.005, pytest.approx(0.007)),
        ("b", 0.005, pytest.approx(0.007)),
    ]
    assert all(s.phase == RequestPhase.KV_TRANSIT for s in states)


def test_finalize_prefill_batch_all_requests_share_kv_ready_time():
    """All requests in a batch share a single kv_ready_time (sum KV model)."""
    ctrl = _StubController(kv_dur=0.002)
    be = _StubBackendForTransition(ctrl)
    a = _running_prefill_state("a")
    b = _running_prefill_state("b")
    # Even with different prefill_end_times, kv_ready_time is computed
    # from the `now` argument (batch end time), not per-request prefill_end_time.
    a.prefill_end_time = 0.005
    b.prefill_end_time = 0.008

    finalize_prefill_batch(be, [a, b], now=0.010)

    # kv_ready_time is now + kv_dur = 0.010 + 0.002 = 0.012 for both
    assert ctrl.prefill_done_calls == [
        ("a", 0.005, pytest.approx(0.012)),
        ("b", 0.008, pytest.approx(0.012)),
    ]
    assert a.kv_ready_time == pytest.approx(0.012)
    assert b.kv_ready_time == pytest.approx(0.012)


def test_drain_kv_ready_and_admit_decode_polls_and_admits():
    ctrl = _StubController(kv_dur=0)

    class _PollingCtrl(_StubController):
        def __init__(self):
            super().__init__(kv_dur=0)
            self._ready_on_next_poll = []

        def stash_ready(self, reqs):
            for r in reqs:
                r.phase = RequestPhase.WAITING_DECODE
                self._waiting.append(r)

        def poll_kv_ready(self, now):
            self.poll_calls.append(now)
            return []

    ctrl = _PollingCtrl()
    be = _StubBackendForTransition(ctrl)
    r1 = _decode_state("r1")
    r1.phase = RequestPhase.WAITING_DECODE
    ctrl.stash_ready([r1])
    admitted = drain_kv_ready_and_admit_decode(be, now=1.0, capacity=4)
    assert ctrl.poll_calls == [1.0]
    assert ctrl.admit_calls == [(4, 1.0)]
    assert [a.rid for a in admitted] == ["r1"]
    assert r1.phase == RequestPhase.RUNNING_DECODE
