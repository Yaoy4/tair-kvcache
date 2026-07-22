"""Phase 2a: tests for BackendA — single-process virtual-time PD adapter.

BackendA owns per-role replica pools (busy_until vectors) and delegates
latency prediction to the per-role predictors held in DisaggPredictors.
It exposes a small API:

  * try_admit_prefill(req, now)   -> (replica_idx, prefill_end_time) | None
  * compute_kv_ready_time(req, now) -> float       (delegates to controller helper)
  * try_admit_decode_step(req, now) -> (replica_idx, end_time) | None
  * earliest_pool_time(pool)      -> float          (introspection)

These exercises must stay hook-free; tests rely only on PD core + factory +
a stub predictor (no AIC perf DB, no sglang).
"""

import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.types import SchedulerConfig
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_runtime import (
    finalize_prefill_batch,
    record_prefill_sampled_tokens,
)


def _model():
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="t",
    )


def _base():
    return SchedulerConfig(
        model=_model(),
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )


class _StubPredictor:
    """Per-device tokens-per-us. Lower coefficient → faster device."""

    _PREFILL_US_PER_TOK = {"fast": 0.1, "slow": 1.0}
    _DECODE_US_PER_STEP = {"fast": 5.0, "slow": 50.0}

    def __init__(self, model, hw, config, **kwargs):
        self.model = model
        self.hw = hw
        self.config = config
        self._prefill_us = self._PREFILL_US_PER_TOK[hw.name]
        self._decode_us = self._DECODE_US_PER_STEP[hw.name]

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self._prefill_us * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        # Cost grows with batch_size (linear) and (optionally) with past_kv.
        if isinstance(past_kv_length, int):
            pkv_total = past_kv_length * batch_size
        else:
            pkv_total = sum(int(x) for x in past_kv_length)
        # 5 us per slot in batch + 0.001 us per past kv token (small)
        return (self._decode_us * batch_size + 0.001 * pkv_total) * 1e-6


class _StubHW:
    def __init__(self, name):
        self.name = name


def _hw(name):
    return _StubHW(name)


def _bundle(prefill_device="fast", decode_device="fast",
            prefill_replicas=2, decode_replicas=2,
            bw_gbps=100.0, latency_us=10.0,
            decode_queue_mode="single_replica",
            prefill_max_running_per_replica=8,
            decode_max_running_per_replica=64):
    cfg = DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(
            device_name=prefill_device,
            tp_size=1,
            replicas=prefill_replicas,
            max_running_per_replica=prefill_max_running_per_replica,
        ),
        decode=RolePredictorConfig(
            device_name=decode_device,
            tp_size=1,
            replicas=decode_replicas,
            max_running_per_replica=decode_max_running_per_replica,
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=bw_gbps, latency_us=latency_us),
        decode_queue_mode=decode_queue_mode,
    )
    return build_disagg(
        model=_model(),
        base_sched_config=_base(),
        disagg_config=cfg,
        predictor_factory=_StubPredictor,
        hw_factory=_hw,
    )


def _req(rid, input_len=512, output_len=4, arrival=0.0):
    return PDRequestState(
        rid=rid,
        arrival_time=arrival,
        phase=RequestPhase.WAITING_PREFILL,
        input_length=input_len,
        output_length=output_len,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_backend_a_initial_clocks_are_zero():
    be = BackendA(_bundle())
    assert be.earliest_pool_time("prefill") == 0.0
    assert be.earliest_pool_time("decode") == 0.0
    assert be.prefill_pool_size() == 2
    assert be.decode_pool_size() == 2


def test_backend_a_rejects_unknown_pool():
    be = BackendA(_bundle())
    with pytest.raises(ValueError):
        be.earliest_pool_time("garbage")


# ---------------------------------------------------------------------------
# Prefill admission
# ---------------------------------------------------------------------------


def test_try_admit_prefill_uses_predicted_latency():
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=1))
    r = _req("r0", input_len=1000)
    idx, end_t = be.try_admit_prefill(r, now=0.0)
    # fast: 0.1 us/token * 1000 = 100 us = 1e-4 s
    assert idx == 0
    assert end_t == pytest.approx(1e-4)
    assert r.phase == RequestPhase.RUNNING_PREFILL
    assert r.prefill_start_time == 0.0


def test_try_admit_prefill_parallel_on_two_replicas():
    be = BackendA(_bundle(prefill_replicas=2, prefill_device="fast"))
    a = _req("a", input_len=1000)
    b = _req("b", input_len=1000)
    ia, ea = be.try_admit_prefill(a, now=0.0)
    ib, eb = be.try_admit_prefill(b, now=0.0)
    assert ia != ib              # different replicas → real parallelism
    assert ea == eb == pytest.approx(1e-4)  # both finish at the same virtual time


def test_try_admit_prefill_queues_when_all_replicas_busy():
    be = BackendA(_bundle(prefill_replicas=1, prefill_device="fast"))
    a = _req("a", input_len=1000)
    b = _req("b", input_len=1000)
    _, end_a = be.try_admit_prefill(a, now=0.0)
    _, end_b = be.try_admit_prefill(b, now=0.0)
    # second request had to wait for first to finish
    assert end_b == pytest.approx(end_a + 1e-4)


def test_heterogeneous_devices_yield_different_prefill_latencies():
    be_fast = BackendA(_bundle(prefill_device="fast"))
    be_slow = BackendA(_bundle(prefill_device="slow"))
    r_fast = _req("rf")
    r_slow = _req("rs")
    _, end_fast = be_fast.try_admit_prefill(r_fast, now=0.0)
    _, end_slow = be_slow.try_admit_prefill(r_slow, now=0.0)
    assert end_slow > end_fast * 9  # ~10x slower predictor


# ---------------------------------------------------------------------------
# KV handoff
# ---------------------------------------------------------------------------


def test_compute_kv_ready_time_uses_transfer_model():
    be = BackendA(_bundle(bw_gbps=100.0, latency_us=10.0))
    r = _req("r", input_len=1)
    # kv_bytes_per_token for FP16/TP=1/MHA = 131072 → 131072 / 100e9 + 10e-6
    expected = 10e-6 + 131072 / 100e9
    assert be.compute_kv_ready_time(r, now=0.0) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Decode step admission
# ---------------------------------------------------------------------------


def test_try_admit_decode_step_uses_predicted_step_latency():
    be = BackendA(_bundle(decode_device="fast", decode_replicas=1))
    r = _req("r")
    r.phase = RequestPhase.RUNNING_DECODE
    idx, end_t = be.try_admit_decode_step(r, now=0.0)
    assert idx == 0
    assert end_t == pytest.approx(5e-6)


def test_try_admit_decode_step_round_robins_across_replicas():
    be = BackendA(_bundle(decode_device="fast", decode_replicas=4))
    reqs = [_req(f"r{i}") for i in range(5)]
    for req in reqs:
        req.phase = RequestPhase.RUNNING_DECODE
    ends = [be.try_admit_decode_step(req, now=0.0)[1] for req in reqs[:4]]
    # all four replicas free at t=0 → all four steps end at the same virtual time
    assert ends == [pytest.approx(5e-6)] * 4
    # fifth must queue behind the earliest-free
    _, end5 = be.try_admit_decode_step(reqs[4], now=0.0)
    assert end5 == pytest.approx(2 * 5e-6)


# ---------------------------------------------------------------------------
# Round-trip: prefill -> KV -> decode lifecycle drives PDController correctly
# ---------------------------------------------------------------------------


def test_full_lifecycle_marks_request_finished():
    be = BackendA(_bundle())
    r = _req("r", input_len=100, output_len=2)
    _, prefill_end = be.try_admit_prefill(r, now=0.0)
    kv_ready = be.compute_kv_ready_time(r, now=prefill_end)
    be.on_prefill_done(r, now=prefill_end, kv_ready_time=kv_ready)
    assert r.phase == RequestPhase.KV_TRANSIT
    be.advance_to_kv_ready(r, now=kv_ready)
    assert r.phase == RequestPhase.WAITING_DECODE
    # two decode steps → finished
    for _ in range(r.output_length):
        _, end_t = be.try_admit_decode_step(r, now=be.earliest_pool_time("decode"))
        be.on_decode_step_done(r, now=end_t)
    assert r.phase == RequestPhase.FINISHED


# ---------------------------------------------------------------------------
# Phase 2b.2 — batch-aware decode (try_admit_decode_batch)
# ---------------------------------------------------------------------------


def _ready_decode_req(rid, input_len=100, output_len=4, past_kv=None):
    r = _req(rid, input_len=input_len, output_len=output_len)
    r.phase = RequestPhase.RUNNING_DECODE
    r.current_past_kv_length = past_kv if past_kv is not None else input_len
    return r


def test_try_admit_decode_batch_passes_past_kv_to_predictor():
    be = BackendA(_bundle(decode_device="fast", decode_replicas=1))
    reqs = [
        _ready_decode_req("a", past_kv=100),
        _ready_decode_req("b", past_kv=200),
        _ready_decode_req("c", past_kv=300),
    ]
    idx, end_t = be.try_admit_decode_batch(reqs, now=0.0)
    # 5 us/slot * 3 + 0.001 us * (100+200+300) = 15 + 0.6 = 15.6 us
    assert idx == 0
    assert end_t == pytest.approx(15.6e-6)


def test_try_admit_decode_batch_advances_one_replica_only():
    be = BackendA(_bundle(decode_device="fast", decode_replicas=4))
    reqs = [_ready_decode_req(f"r{i}", past_kv=0) for i in range(3)]
    idx, end_t = be.try_admit_decode_batch(reqs, now=0.0)
    # one replica advances; the other three stay at 0
    busy = sorted(be._decode_pool.busy_until)
    assert busy[:3] == [0.0, 0.0, 0.0]
    assert busy[3] == pytest.approx(end_t)


def test_try_admit_decode_batch_per_replica_queue_spreads_across_replicas():
    be = BackendA(
        _bundle(
            decode_device="fast",
            decode_replicas=2,
            decode_queue_mode="per_replica_queue",
        )
    )
    reqs = [_ready_decode_req(f"r{i}", past_kv=10 * (i + 1)) for i in range(3)]
    idx, end_t = be.try_admit_decode_batch(reqs, now=0.0)
    busy = sorted(be._decode_pool.busy_until)
    assert busy[0] > 0.0
    assert busy[1] > busy[0]
    assert idx in {0, 1}
    assert end_t == pytest.approx(busy[1])


def test_try_admit_decode_batch_uses_earliest_replica():
    be = BackendA(_bundle(decode_device="fast", decode_replicas=2))
    # warm up replica 0 with a single step
    be.try_admit_decode_step(_ready_decode_req("warm"), now=0.0)
    # next batch should land on replica 1 (still free at t=0)
    reqs = [_ready_decode_req(f"r{i}", past_kv=0) for i in range(2)]
    idx, _ = be.try_admit_decode_batch(reqs, now=0.0)
    assert idx == 1


def test_try_admit_decode_batch_rejects_empty():
    be = BackendA(_bundle())
    with pytest.raises(ValueError):
        be.try_admit_decode_batch([], now=0.0)


def test_on_decode_batch_step_done_advances_all_requests():
    be = BackendA(_bundle(decode_device="fast", decode_replicas=1))
    reqs = [_ready_decode_req(f"r{i}", input_len=10, output_len=1, past_kv=10)
            for i in range(3)]
    _, end_t = be.try_admit_decode_batch(reqs, now=0.0)
    be.on_decode_step_done_batch(reqs, now=end_t)
    # each req gets one decode step credited; output_len=1 → FINISHED
    for r in reqs:
        assert r.phase == RequestPhase.FINISHED


def test_single_replica_decode_admission_respects_capacity():
    be = BackendA(
        _bundle(
            decode_queue_mode="single_replica",
            decode_max_running_per_replica=1,
        )
    )
    ctrl = be.controller()
    reqs = [_req("a", output_len=1), _req("b", output_len=1)]
    for req in reqs:
        ctrl.on_request_arrival(req, 0.0)
        ctrl.admit_prefill(1, 0.0)
        ctrl.on_prefill_done(req, 0.0, 0.0)
        ctrl.poll_kv_ready(0.0)

    admitted = be.admit_decode_single_replica({"a", "b"}, 0.0)
    assert len(admitted) == 1
    assert ctrl.decode_waiting_count() == 1


def test_external_termination_releases_single_decode_capacity():
    be = BackendA(
        _bundle(
            decode_queue_mode="single_replica",
            decode_max_running_per_replica=1,
        )
    )
    ctrl = be.controller()
    reqs = [_req("eos", output_len=8), _req("next", output_len=8)]
    for req in reqs:
        ctrl.on_request_arrival(req, 0.0)
        ctrl.admit_prefill(1, 0.0)
        ctrl.on_prefill_done(req, 0.0, 0.0)
        ctrl.poll_kv_ready(0.0)

    assert be.admit_decode_single_replica({"eos"}, 0.0) == [reqs[0]]
    be.terminate_request(reqs[0], 0.1)
    assert be.admit_decode_single_replica({"next"}, 0.1) == [reqs[1]]


def test_external_termination_releases_chunked_prefill_capacity():
    be = BackendA(
        _bundle(
            prefill_replicas=1,
            prefill_max_running_per_replica=1,
        )
    )
    aborted = _req("aborted")
    aborted.prefill_is_final_chunk = False
    be.try_admit_prefill_batch([aborted], now=0.0)

    be.terminate_request(aborted, now=0.1)
    fresh = _req("fresh")
    be.try_admit_prefill_batch([fresh], now=0.1)

    assert fresh.phase == RequestPhase.RUNNING_PREFILL


# ---------------------------------------------------------------------------
# 0721-Retract regression: SGLang's real KV-cache-pool-full retraction evicts
# a running-decode request and re-queues the same rid for a fresh prefill.
# Before this fix, HiSim's PD_REQUEST_STATES entry stayed at RUNNING_DECODE,
# so the re-prefill crashed the whole scheduler process with:
#   ValueError: cannot schedule prefill for rid=... in phase=running_decode
# ---------------------------------------------------------------------------


def test_retract_from_running_decode_allows_reprefill_without_crash():
    """Reproduces the crash end-to-end via realistic backend calls (prefill
    -> finalize -> sample first token -> admit decode -> one decode step),
    then retracts and re-admits the SAME rid to prefill. Before the fix, the
    final try_admit_prefill_batch call below raised ValueError.
    """
    be = BackendA(_bundle(prefill_replicas=1, decode_replicas=1))
    req = _req("retract-me", input_len=128, output_len=8)
    _, prefill_end = be.try_admit_prefill_batch([req], now=0.0)
    finalize_prefill_batch(be, [req], now=prefill_end)
    record_prefill_sampled_tokens(be, [req])
    be.controller().poll_kv_ready(req.kv_ready_time)
    be.admit_decode_single_replica({req.rid}, req.kv_ready_time)
    _, decode_end = be.try_admit_decode_batch([req], req.kv_ready_time)
    be.on_decode_step_done_batch([req], decode_end)
    assert req.phase == RequestPhase.RUNNING_DECODE
    progress_before_retract = req.decode_step_count
    kv_before_retract = req.current_past_kv_length

    # SGLang retracts: evicts KV, re-queues the same rid for a fresh prefill.
    be.reset_for_retract(req, now=decode_end)
    assert req.phase == RequestPhase.WAITING_PREFILL

    # This exact call used to raise:
    # ValueError: cannot schedule prefill for rid=... in phase=running_decode
    _, reprefill_end = be.try_admit_prefill_batch([req], now=decode_end)

    assert req.phase == RequestPhase.RUNNING_PREFILL
    assert reprefill_end > decode_end
    # Already-generated output tokens must not be rewound: SGLang preserves
    # them across retraction and only rebuilds their KV cache.
    assert req.decode_step_count == progress_before_retract
    assert req.current_past_kv_length == kv_before_retract
    assert req.output_length == 8


def test_retract_releases_single_replica_decode_capacity():
    be = BackendA(
        _bundle(
            decode_queue_mode="single_replica",
            decode_max_running_per_replica=1,
        )
    )
    ctrl = be.controller()
    reqs = [_req("retracted", output_len=8), _req("next", output_len=8)]
    for req in reqs:
        ctrl.on_request_arrival(req, 0.0)
        ctrl.admit_prefill(1, 0.0)
        ctrl.on_prefill_done(req, 0.0, 0.0)
        ctrl.poll_kv_ready(0.0)

    assert be.admit_decode_single_replica({"retracted"}, 0.0) == [reqs[0]]
    # Replica is already at capacity (1/1): a second request cannot be admitted.
    assert be.admit_decode_single_replica({"next"}, 0.0) == []

    be.reset_for_retract(reqs[0], 0.1)
    assert be.admit_decode_single_replica({"next"}, 0.1) == [reqs[1]]


def test_retract_releases_per_replica_decode_capacity_and_prefill_stickiness():
    be = BackendA(
        _bundle(
            prefill_replicas=1,
            prefill_max_running_per_replica=1,
            decode_queue_mode="per_replica_queue",
            decode_replicas=1,
            decode_max_running_per_replica=1,
        )
    )
    req = _req("retract-me", input_len=64, output_len=8)
    _, prefill_end = be.try_admit_prefill_batch([req], now=0.0)
    finalize_prefill_batch(be, [req], now=prefill_end)
    record_prefill_sampled_tokens(be, [req])
    be.controller().poll_kv_ready(req.kv_ready_time)
    idx = be.bind_decode_replicas([req])[req.rid]
    be.admit_decode_for_replica(idx, {req.rid}, req.kv_ready_time)
    assert req.phase == RequestPhase.RUNNING_DECODE
    retract_time = req.kv_ready_time

    be.reset_for_retract(req, now=retract_time)

    # Decode-side reservation must be fully released, or this replica's
    # capacity would be leaked forever.
    assert req.rid not in be._decode_replica_by_rid
    assert be._decode_running_count[idx] == 0
    # Prefill-side sticky binding must also be cleared: otherwise the
    # re-admission below would silently bypass max_running_per_replica by
    # reusing the stale replica_idx without a capacity check.
    assert req.rid not in be._prefill_replica_by_rid

    # Saturate the sole prefill replica with a different, still-in-flight
    # (non-final-chunk) request so it is genuinely at capacity.
    occupier = _req("occupier", input_len=64, output_len=1)
    occupier.prefill_is_final_chunk = False
    be.try_admit_prefill_batch([occupier], now=retract_time)
    assert be._prefill_running_count[0] == 1

    # With the stale binding cleared, re-admitting the retracted request must
    # go through the real capacity check -- and since the only replica is
    # already full, it must be rejected rather than silently double-booked
    # via a leftover sticky replica_idx that skips the capacity check.
    with pytest.raises(RuntimeError, match="prefill capacity exhausted"):
        be.try_admit_prefill_batch([req], now=retract_time)


def test_per_replica_decode_binding_avoids_full_replica():
    be = BackendA(
        _bundle(
            decode_queue_mode="per_replica_queue",
            decode_replicas=2,
            decode_max_running_per_replica=1,
        )
    )
    be._decode_running_count[0] = 1
    req = _req("new")
    req.phase = RequestPhase.WAITING_DECODE

    mapping = be.bind_decode_replicas([req])

    assert mapping[req.rid] == 1


# ---------------------------------------------------------------------------
# Batch prefill admission (try_admit_prefill_batch)
# ---------------------------------------------------------------------------


def test_try_admit_prefill_batch_uses_sum_tokens():
    """Predictor must receive sum of all input_lengths, not per-request values."""
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=1))
    reqs = [_req(f"r{i}", input_len=200) for i in range(4)]  # 800 tokens total
    idx, end_t = be.try_admit_prefill_batch(reqs, now=0.0)
    # fast: 0.1 us/tok * 800 = 80 us = 8e-5 s
    assert idx == 0
    assert end_t == pytest.approx(8e-5)


def test_try_admit_prefill_batch_all_reqs_share_end_time():
    """All requests in a batch receive the same prefill_end_time."""
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=1))
    reqs = [_req("a", input_len=100), _req("b", input_len=300), _req("c", input_len=200)]
    _, end_t = be.try_admit_prefill_batch(reqs, now=0.0)
    for r in reqs:
        assert r.prefill_end_time == pytest.approx(end_t)


def test_try_admit_prefill_batch_all_reqs_share_start_time():
    """All requests in a batch receive the same prefill_start_time via controller."""
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=1))
    reqs = [_req("a"), _req("b"), _req("c")]
    be.try_admit_prefill_batch(reqs, now=0.5)
    assert all(r.prefill_start_time == pytest.approx(0.5) for r in reqs)
    assert all(r.phase == RequestPhase.RUNNING_PREFILL for r in reqs)


def test_try_admit_prefill_batch_partitions_across_replicas():
    """One central SGLang batch becomes replica-local predictor batches."""
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=4))
    reqs = [_req(f"r{i}", input_len=100) for i in range(3)]
    be.try_admit_prefill_batch(reqs, now=0.0)
    busy_count = sum(1 for t in be._prefill_pool.busy_until if t > 0.0)
    assert busy_count == 3
    assert len({be.prefill_replica_for(r.rid) for r in reqs}) == 3


def test_try_admit_prefill_batch_enforces_capacity_with_waves():
    be = BackendA(
        _bundle(
            prefill_device="fast",
            prefill_replicas=1,
            prefill_max_running_per_replica=2,
        )
    )
    reqs = [_req(f"r{i}", input_len=100) for i in range(5)]
    _, end_t = be.try_admit_prefill_batch(reqs, now=0.0)

    # Three fused waves: [2 requests], [2 requests], [1 request].
    assert end_t == pytest.approx(5e-5)
    assert [r.prefill_start_time for r in reqs] == pytest.approx(
        [0.0, 0.0, 2e-5, 2e-5, 4e-5]
    )


def test_chunked_prefill_reserves_capacity_across_scheduler_batches():
    be = BackendA(
        _bundle(
            prefill_device="fast",
            prefill_replicas=1,
            prefill_max_running_per_replica=1,
        )
    )
    chunked = _req("chunked", input_len=100)
    chunked.prefill_is_final_chunk = False
    be.try_admit_prefill_batch([chunked], now=0.0)

    fresh = _req("fresh", input_len=100)
    with pytest.raises(RuntimeError, match="unfinished chunked"):
        be.try_admit_prefill_batch([fresh], now=0.0)
    assert fresh.phase == RequestPhase.WAITING_PREFILL

    chunked.prefill_is_final_chunk = True
    _, final_end = be.try_admit_prefill_batch([chunked], now=0.0)
    finalize_prefill_batch(be, [chunked], now=final_end)
    be.try_admit_prefill_batch([fresh], now=final_end)
    assert fresh.phase == RequestPhase.RUNNING_PREFILL


def test_chunked_prefill_keeps_replica_affinity_until_handoff():
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=2))
    req = _req("chunked", input_len=100)
    req.prefill_is_final_chunk = False
    be.try_admit_prefill_batch([req], now=0.0)
    first_replica = be.prefill_replica_for(req.rid)

    req.input_length = 50
    req.prefill_is_final_chunk = True
    be.try_admit_prefill_batch([req], now=0.0)
    assert be.prefill_replica_for(req.rid) == first_replica

    prefill_end = req.prefill_end_time
    be.on_prefill_done(req, prefill_end, prefill_end)
    assert be.prefill_replica_for(req.rid) is None


def test_replica_local_prefill_batches_start_kv_handoff_independently():
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=2))
    fast = _req("fast", input_len=100)
    slow = _req("slow", input_len=300)
    _, batch_end = be.try_admit_prefill_batch([fast, slow], now=0.0)

    finalize_prefill_batch(be, [fast, slow], now=batch_end)

    assert fast.prefill_batch_id != slow.prefill_batch_id
    assert fast.prefill_end_time < slow.prefill_end_time
    assert fast.kv_ready_time < slow.kv_ready_time


def test_kv_handoff_submission_is_ordered_by_prefill_completion():
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=2))
    fast = _req("fast", input_len=100)
    slow = _req("slow", input_len=300)
    _, batch_end = be.try_admit_prefill_batch([fast, slow], now=0.0)

    # Deliberately reverse caller order. The later-finishing slow request must
    # not reserve the shared link ahead of the already-complete fast request.
    finalize_prefill_batch(be, [slow, fast], now=batch_end)

    assert fast.prefill_end_time < slow.prefill_end_time
    assert fast.kv_ready_time < slow.kv_ready_time


def test_try_admit_prefill_batch_queues_behind_busy_replica():
    """Two consecutive batches on 1 replica must serialize."""
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=1))
    batch_a = [_req("a", input_len=1000)]
    batch_b = [_req("b", input_len=1000)]
    _, end_a = be.try_admit_prefill_batch(batch_a, now=0.0)
    _, end_b = be.try_admit_prefill_batch(batch_b, now=0.0)
    # fast: 0.1 us * 1000 = 100 us each → serial → b ends at 200 us
    assert end_a == pytest.approx(1e-4)
    assert end_b == pytest.approx(2e-4)


def test_try_admit_prefill_batch_rejects_empty():
    be = BackendA(_bundle())
    with pytest.raises(ValueError):
        be.try_admit_prefill_batch([], now=0.0)


def test_prefill_batch_rejects_request_from_decode_phase():
    be = BackendA(_bundle())
    req = _req("wrong-phase", input_len=100)
    req.phase = RequestPhase.RUNNING_DECODE
    with pytest.raises(ValueError, match="cannot schedule prefill"):
        be.try_admit_prefill_batch([req], now=0.0)
    assert req.prefill_end_time is None


def test_compute_batch_kv_ready_time_uses_total_tokens():
    """KV ready time must use sum of tokens, not per-request max."""
    be = BackendA(_bundle(bw_gbps=100.0, latency_us=10.0))
    # kv_bytes_per_token for FP16/TP=1 = 131072
    total_tokens = 4
    expected = 10e-6 + 4 * 131072 / 100e9
    assert be.compute_batch_kv_ready_time(total_tokens, now=0.0) == pytest.approx(expected)


def test_final_prefill_token_is_sampled_without_decode_forward():
    be = BackendA(_bundle(prefill_replicas=1, decode_replicas=1))
    req = _req("osl-one", input_len=128, output_len=1)
    _, prefill_end = be.try_admit_prefill_batch([req], now=0.0)
    finalize_prefill_batch(be, [req], now=prefill_end)

    token_times = record_prefill_sampled_tokens(be, [req])

    assert token_times[req.rid] == req.kv_ready_time
    assert req.kv_ready_time > prefill_end
    assert be.decode_replica_time(0) == 0.0
    assert req.decode_start_time is None
    assert req.decode_step_count == 1
    assert req.current_past_kv_length == 0
    assert req.phase == RequestPhase.FINISHED
    assert req.decode_end_time == token_times[req.rid]
    assert be.controller().kv_transit_count() == 0


def test_prefill_sample_plus_explicit_decode_steps_reaches_osl():
    be = BackendA(_bundle(prefill_replicas=1, decode_replicas=1))
    req = _req("osl-three", input_len=128, output_len=3)
    _, prefill_end = be.try_admit_prefill_batch([req], now=0.0)
    finalize_prefill_batch(be, [req], now=prefill_end)

    token_times = record_prefill_sampled_tokens(be, [req])
    assert req.decode_step_count == 1
    assert req.current_past_kv_length == 0
    assert req.phase == RequestPhase.KV_TRANSIT
    assert token_times[req.rid] == req.kv_ready_time
    assert be.controller().kv_transit_count() == 1

    be.controller().poll_kv_ready(req.kv_ready_time)
    admitted = be.admit_decode_single_replica({req.rid}, req.kv_ready_time)
    assert admitted == [req]
    assert req.current_past_kv_length == 128

    _, second_end = be.try_admit_decode_batch([req], req.kv_ready_time)
    be.on_decode_step_done_batch([req], second_end)
    assert req.decode_step_count == 2
    assert req.phase == RequestPhase.RUNNING_DECODE

    _, third_end = be.try_admit_decode_batch([req], second_end)
    be.on_decode_step_done_batch([req], third_end)
    assert req.decode_step_count == 3
    assert req.phase == RequestPhase.FINISHED


def test_prefill_sampling_binds_decode_instances_without_running_forward():
    be = BackendA(
        _bundle(
            prefill_replicas=2,
            decode_replicas=2,
            decode_queue_mode="per_replica_queue",
        )
    )
    reqs = [
        _req("sample-a", input_len=128, output_len=3),
        _req("sample-b", input_len=128, output_len=3),
    ]
    _, prefill_end = be.try_admit_prefill_batch(reqs, now=0.0)
    finalize_prefill_batch(be, reqs, now=prefill_end)

    token_times = record_prefill_sampled_tokens(be, reqs)
    mapping = be.bind_decode_replicas(reqs)

    assert set(mapping.values()) == {0, 1}
    assert token_times == {req.rid: req.kv_ready_time for req in reqs}
    assert all(req.decode_step_count == 1 for req in reqs)
    assert all(req.current_past_kv_length == 0 for req in reqs)
    assert be.decode_replica_time(0) == 0.0
    assert be.decode_replica_time(1) == 0.0

