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
            decode_queue_mode="single_replica"):
    cfg = DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(
            device_name=prefill_device,
            tp_size=1,
            replicas=prefill_replicas,
            max_running_per_replica=8,
        ),
        decode=RolePredictorConfig(
            device_name=decode_device,
            tp_size=1,
            replicas=decode_replicas,
            max_running_per_replica=64,
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
    idx, end_t = be.try_admit_decode_step(r, now=0.0)
    assert idx == 0
    assert end_t == pytest.approx(5e-6)


def test_try_admit_decode_step_round_robins_across_replicas():
    be = BackendA(_bundle(decode_device="fast", decode_replicas=4))
    ends = [be.try_admit_decode_step(_req(f"r{i}"), now=0.0)[1] for i in range(4)]
    # all four replicas free at t=0 → all four steps end at the same virtual time
    assert ends == [pytest.approx(5e-6)] * 4
    # fifth must queue behind the earliest-free
    _, end5 = be.try_admit_decode_step(_req("r4"), now=0.0)
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


def test_try_admit_prefill_batch_occupies_single_replica():
    """The batch goes to ONE replica; other replicas stay free."""
    be = BackendA(_bundle(prefill_device="fast", prefill_replicas=4))
    reqs = [_req(f"r{i}", input_len=100) for i in range(3)]
    be.try_admit_prefill_batch(reqs, now=0.0)
    busy_count = sum(1 for t in be._prefill_pool.busy_until if t > 0.0)
    assert busy_count == 1


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


def test_compute_batch_kv_ready_time_uses_total_tokens():
    """KV ready time must use sum of tokens, not per-request max."""
    be = BackendA(_bundle(bw_gbps=100.0, latency_us=10.0))
    # kv_bytes_per_token for FP16/TP=1 = 131072
    total_tokens = 4
    expected = 10e-6 + 4 * 131072 / 100e9
    assert be.compute_batch_kv_ready_time(total_tokens, now=0.0) == pytest.approx(expected)

