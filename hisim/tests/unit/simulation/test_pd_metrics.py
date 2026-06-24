"""Phase 3 tests: PD stage-aware metrics.

Scope:
- pd_metrics.compute_stage_durations: derives prefill_queue_wait /
  kv_transfer_time / decode_queue_wait from a PDRequestState.
- pd_metrics.populate_request_stats: writes stage durations onto a
  RequestStats instance.
- RequestStats has stage fields with safe defaults (zero) so existing
  non-disagg paths keep working.
- calc_metrics aggregates stage percentiles when stage fields are populated.
"""
import math

import pytest

from hisim.simulation.pd_metrics import (
    compute_stage_durations,
    flush_finished_states,
    populate_request_stats,
)
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.types import RequestStats
from hisim.simulation.utils import calc_metrics


def _make_finished_state(
    rid: str,
    arrival: float,
    prefill_start: float,
    prefill_end: float,
    kv_ready: float,
    decode_start: float,
    input_length: int = 8,
    output_length: int = 4,
) -> PDRequestState:
    return PDRequestState(
        rid=rid,
        arrival_time=arrival,
        phase=RequestPhase.FINISHED,
        input_length=input_length,
        output_length=output_length,
        prefill_start_time=prefill_start,
        prefill_end_time=prefill_end,
        kv_ready_time=kv_ready,
        decode_start_time=decode_start,
        decode_step_count=output_length,
        current_past_kv_length=input_length + output_length,
    )


def test_compute_stage_durations_basic():
    state = _make_finished_state(
        "r1",
        arrival=1.0,
        prefill_start=1.5,
        prefill_end=2.0,
        kv_ready=2.25,
        decode_start=2.4,
    )
    d = compute_stage_durations(state)
    assert d["prefill_queue_wait"] == pytest.approx(0.5)
    assert d["kv_transfer_time"] == pytest.approx(0.25)
    assert d["decode_queue_wait"] == pytest.approx(0.15)


def test_compute_stage_durations_missing_timestamps_returns_zero():
    state = PDRequestState(rid="r1", arrival_time=1.0)
    d = compute_stage_durations(state)
    assert d == {
        "prefill_queue_wait": 0.0,
        "kv_transfer_time": 0.0,
        "decode_queue_wait": 0.0,
    }


def test_compute_stage_durations_never_negative():
    # Pathological inputs: decode_start < kv_ready should clamp to 0
    state = _make_finished_state(
        "r1",
        arrival=1.0,
        prefill_start=1.0,
        prefill_end=2.0,
        kv_ready=3.0,
        decode_start=2.5,
    )
    d = compute_stage_durations(state)
    assert d["decode_queue_wait"] == 0.0


def test_request_stats_has_stage_fields_default_zero():
    s = RequestStats()
    assert s.prefill_queue_wait == 0.0
    assert s.kv_transfer_time == 0.0
    assert s.decode_queue_wait == 0.0


def test_populate_request_stats_writes_durations():
    state = _make_finished_state(
        "r1",
        arrival=1.0,
        prefill_start=1.5,
        prefill_end=2.0,
        kv_ready=2.25,
        decode_start=2.4,
    )
    stats = RequestStats(rid="r1")
    populate_request_stats(stats, state)
    assert stats.prefill_queue_wait == pytest.approx(0.5)
    assert stats.kv_transfer_time == pytest.approx(0.25)
    assert stats.decode_queue_wait == pytest.approx(0.15)


def _make_completed_request_stats(
    rid: str,
    pq: float,
    kvt: float,
    dq: float,
    ttft: float = 0.05,
    tpot: float = 0.01,
    output_length: int = 4,
) -> RequestStats:
    gen_lats = [ttft] + [tpot] * (output_length - 1)
    s = RequestStats(
        rid=rid,
        last_event_time=10.0,
        input_length=8,
        output_length=output_length,
        queue_start=0.0,
        queue_end=pq,
        created_time=0.0,
        gen_token_latencies=gen_lats,
    )
    s.prefill_queue_wait = pq
    s.kv_transfer_time = kvt
    s.decode_queue_wait = dq
    return s


def test_calc_metrics_emits_stage_percentiles_when_populated():
    reqs = [
        _make_completed_request_stats(f"r{i}", pq=0.1 * (i + 1),
                                       kvt=0.02 * (i + 1),
                                       dq=0.03 * (i + 1))
        for i in range(10)
    ]
    m = calc_metrics(reqs)
    # All stage metric keys present and non-NaN
    for key in [
        "mean_prefill_queue_ms",
        "p50_prefill_queue_ms",
        "p95_prefill_queue_ms",
        "p99_prefill_queue_ms",
        "mean_kv_transfer_ms",
        "p50_kv_transfer_ms",
        "p95_kv_transfer_ms",
        "p50_decode_queue_ms",
        "p95_decode_queue_ms",
    ]:
        assert key in m, f"missing metric {key}"
        assert not math.isnan(m[key])
    # Sanity: p95 prefill_queue should be > p50
    assert m["p95_prefill_queue_ms"] >= m["p50_prefill_queue_ms"]


def test_calc_metrics_stage_percentiles_zero_when_unpopulated():
    """Non-disagg path: stage fields default to 0; metrics should be 0 not error."""
    s = RequestStats(
        rid="r1",
        last_event_time=1.0,
        input_length=4,
        output_length=2,
        queue_start=0.0,
        queue_end=0.0,
        created_time=0.0,
        gen_token_latencies=[0.05, 0.01],
    )
    m = calc_metrics([s])
    assert m["mean_prefill_queue_ms"] == 0.0
    assert m["p95_kv_transfer_ms"] == 0.0
    assert m["p99_decode_queue_ms"] == 0.0


def test_flush_finished_states_populates_stats_and_gcs():
    finished = _make_finished_state(
        "r1", arrival=1.0, prefill_start=1.5, prefill_end=2.0,
        kv_ready=2.25, decode_start=2.4,
    )
    in_flight = PDRequestState(
        rid="r2", arrival_time=1.0, phase=RequestPhase.RUNNING_DECODE,
    )
    pd_states = {"r1": finished, "r2": in_flight}
    stats_r1 = RequestStats(rid="r1")
    stats_r2 = RequestStats(rid="r2")
    request_stats = {"r1": stats_r1, "r2": stats_r2}

    flushed = flush_finished_states(pd_states, request_stats)
    assert flushed == 1
    # r1 removed, r2 kept.
    assert "r1" not in pd_states
    assert "r2" in pd_states
    # r1 stats populated; r2 untouched.
    assert stats_r1.prefill_queue_wait == pytest.approx(0.5)
    assert stats_r1.decode_queue_wait == pytest.approx(0.15)
    assert stats_r2.prefill_queue_wait == 0.0


def test_flush_finished_states_no_op_when_stats_missing():
    """Hook may flush a state for which RequestStats was never created
    (e.g. warmup); should drop PD state without raising."""
    finished = _make_finished_state(
        "r1", arrival=1.0, prefill_start=1.0, prefill_end=2.0,
        kv_ready=2.0, decode_start=2.0,
    )
    pd_states = {"r1": finished}
    flushed = flush_finished_states(pd_states, request_stats={})
    assert flushed == 1
    assert pd_states == {}
