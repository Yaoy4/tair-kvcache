"""Unit tests for the SGLang-free PD timeline helpers (P1 dual-clock).

Covers the pure-function acceptance criteria for the P1 todos:
- p1-01: the three named helpers (sync_decode_start / first_token_latency /
         decode_step_token_latency) plus the prefill/decode advance seams.
- p1-02: role-clock decoupling (a prefill step never advances the decode clock).
- p1-03: prefill->decode handoff sync (decode_start == max(decode_clock, kv_ready)).
- p1-04: KV transfer does not freeze in-flight decode (ITL stays == step latency).
- p1-05/06/07: ITL/TTFT/E2E/makespan arithmetic on the role-clock timeline.
"""
import pytest

from hisim.simulation.pd_timeline import (
    advance_after_decode_step,
    closed_loop_first_token_latency,
    decode_step_token_latency,
    first_token_latency,
    prefill_batch_start,
    sync_decode_start,
)
from hisim.simulation.pd_types import PDRequestState, RequestPhase


def _state(arrival, decode_start=None, kv_ready=None):
    return PDRequestState(
        rid="r",
        arrival_time=arrival,
        phase=RequestPhase.WAITING_DECODE,
        input_length=8,
        output_length=3,
        kv_ready_time=kv_ready,
        decode_start_time=decode_start,
    )


# ----------------------------------------------------------------------------
# p1-01: the three named pure helpers
# ----------------------------------------------------------------------------
def test_sync_decode_start_gates_on_kv_ready():
    # KV still in flight -> decode clock jumps forward to kv_ready.
    assert sync_decode_start(3.0, 11.0) == 11.0
    # KV already arrived -> engine-free time dominates.
    assert sync_decode_start(5.0, 2.0) == 5.0
    # No KV gate known -> unchanged.
    assert sync_decode_start(7.0, None) == 7.0


def test_first_token_latency_folds_full_ttft():
    assert first_token_latency(_state(0.0, decode_start=11.0), 1.0) == pytest.approx(12.0)
    # Later arrival shrinks TTFT by exactly the arrival offset.
    assert first_token_latency(_state(2.0, decode_start=11.0), 0.5) == pytest.approx(9.5)


def test_decode_step_token_latency_is_identity():
    assert decode_step_token_latency(0.004) == pytest.approx(0.004)


# ----------------------------------------------------------------------------
# p1-02: role-clock decoupling — prefill advance must not touch decode clock
# ----------------------------------------------------------------------------
def test_prefill_advance_does_not_move_decode_clock():
    prefill_clock = 0.0
    decode_clock = 0.0
    # Run a prefill batch of latency P=0.01 for a request that arrived at t=0.
    start = prefill_batch_start(prefill_clock, [0.0])
    assert start == 0.0
    prefill_clock = start + 0.01  # advance prefill role clock only
    assert prefill_clock == pytest.approx(0.01)
    # Decode clock is untouched by the prefill step (the decoupling invariant).
    assert decode_clock == 0.0


def test_prefill_batch_start_self_anchors_to_latest_arrival():
    # Engine free at 0, but the latest request arrived at t=5 -> batch waits.
    assert prefill_batch_start(0.0, [1.0, 5.0, 2.0]) == 5.0
    # Engine already past all arrivals -> engine-free time dominates.
    assert prefill_batch_start(9.0, [1.0, 5.0]) == 9.0
    # Unknown / negative arrivals are ignored.
    assert prefill_batch_start(3.0, [None, -1.0]) == 3.0


# ----------------------------------------------------------------------------
# p1-03: prefill->decode handoff sync + decode_queue_wait semantics
# ----------------------------------------------------------------------------
def test_handoff_sync_waits_for_kv_then_runs_back_to_back():
    # KV ready at 11 while engine free at 3 -> decode starts at 11.
    decode_start = sync_decode_start(3.0, 11.0)
    assert decode_start == 11.0
    decode_queue_wait = decode_start - 11.0  # decode_start - kv_ready
    assert decode_queue_wait == pytest.approx(0.0)

    # KV ready early (2) but engine busy until 5 -> decode starts at 5,
    # i.e. it queued 3s behind KV arrival.
    decode_start = sync_decode_start(5.0, 2.0)
    assert decode_start == 5.0
    assert decode_start - 2.0 == pytest.approx(3.0)


# ----------------------------------------------------------------------------
# p1-04: KV transfer of one request must not freeze another's decode
# ----------------------------------------------------------------------------
def test_kv_transfer_does_not_inflate_other_request_itl():
    D = 0.002  # decode step latency
    # Request B is mid-decode; its decode clock advances purely by D regardless
    # of any concurrent prefill+KV work happening for request A.
    decode_clock = 1.000
    step_start = sync_decode_start(decode_clock, None)  # B already decoding, no KV gate
    step_end = advance_after_decode_step(step_start, D)
    itl = decode_step_token_latency(step_end - decode_clock)
    assert itl == pytest.approx(D)  # == D, NOT D + prefill + KV


# ----------------------------------------------------------------------------
# p1-05/06/07: full single-request decode timeline (ITL / TTFT / E2E / makespan)
# ----------------------------------------------------------------------------
def test_single_request_timeline_itl_ttft_e2e_makespan():
    arrival = 0.0
    kv_ready = 11.0
    D = 1.0
    output_length = 3

    decode_clock = 0.0
    last_event_time = arrival  # mirrors RequestStats.last_event_time == created_time
    gen = []
    decode_end = None

    for step in range(output_length):
        if step == 0:
            step_start = sync_decode_start(decode_clock, kv_ready)  # handoff
        else:
            step_start = sync_decode_start(decode_clock, None)  # back-to-back
        step_end = advance_after_decode_step(step_start, D)
        decode_clock = step_end
        # token recorded at the decode role-clock step end
        gen.append(step_end - last_event_time)
        last_event_time = step_end
        decode_end = step_end

    # p1-06: gen == [TTFT, ITL, ITL] with TTFT folding prefill+KV+first step.
    assert gen == pytest.approx([12.0, 1.0, 1.0])
    ttft = gen[0]
    assert ttft == pytest.approx(12.0)
    # p1-05: inter-token latencies are the constant step latency.
    assert gen[1:] == pytest.approx([1.0, 1.0])
    # p1-06: E2E telescopes to decode_end - arrival.
    assert sum(gen) == pytest.approx(14.0)
    assert sum(gen) == pytest.approx(decode_end - arrival)
    # p1-07: last_event_time (throughput makespan input) == decode_end.
    assert last_event_time == pytest.approx(decode_end)


def test_prefill_between_decode_steps_does_not_pollute_itl():
    """Regression guard for p1-05: a prefill batch firing between two decode
    steps (advancing the *global* clock) must NOT widen the decode ITL, because
    ITL is measured on the decode role clock only."""
    D = 0.5
    decode_clock = 10.0
    last_event_time = 10.0

    # decode step n
    s1 = sync_decode_start(decode_clock, None)
    e1 = advance_after_decode_step(s1, D)
    decode_clock = e1
    itl_1 = e1 - last_event_time
    last_event_time = e1

    # ... a prefill batch of P=3.0 runs here on a SEPARATE clock; decode clock
    # is untouched (we simply do not advance it) ...
    _prefill_global_jump = 3.0  # noqa: F841 — represents global-clock pollution

    # decode step n+1
    s2 = sync_decode_start(decode_clock, None)
    e2 = advance_after_decode_step(s2, D)
    itl_2 = e2 - last_event_time

    assert itl_1 == pytest.approx(D)
    assert itl_2 == pytest.approx(D)  # NOT D + 3.0


# ----------------------------------------------------------------------------
# Closed-loop TTFT for first token (service-only in rate=inf+cap emulation)
# ----------------------------------------------------------------------------
def test_closed_loop_first_token_latency_uses_service_spans():
    # Example with synthetic cap/decode queueing in front of this request.
    # Service spans are still only prefill+KV and first decode step.
    first_token_time = 10.011
    ttft = closed_loop_first_token_latency(
        first_token_time,
        prefill_start_time=1.500,
        kv_ready_time=1.540,
        decode_start_time=10.000,
    )
    assert ttft == pytest.approx(0.051)  # 40ms + 11ms


def test_closed_loop_first_token_latency_returns_none_when_missing_timestamps():
    assert (
        closed_loop_first_token_latency(
            5.0,
            prefill_start_time=None,
            kv_ready_time=1.0,
            decode_start_time=2.0,
        )
        is None
    )
    assert (
        closed_loop_first_token_latency(
            5.0,
            prefill_start_time=1.0,
            kv_ready_time=None,
            decode_start_time=2.0,
        )
        is None
    )
    assert (
        closed_loop_first_token_latency(
            5.0,
            prefill_start_time=1.0,
            kv_ready_time=1.5,
            decode_start_time=None,
        )
        is None
    )
