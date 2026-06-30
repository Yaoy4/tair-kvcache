"""Phase 5b — Backend A/B parity tests.

These tests assert that BackendA (single-process virtual time) and BackendB
(two-process skeleton) return *bit-identical* scheduling decisions for the
same predictor inputs. Anything that diverges here is a bug in the Backend B
adapter layer — the PD semantic layer is shared and must not move.

The stub predictor lives at module top level so it survives multiprocessing
'spawn' pickling.
"""

from __future__ import annotations

import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_backend_b import BackendB
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.types import SchedulerConfig


# ---------------------------------------------------------------------------
# Top-level stub (picklable for 'spawn').
# ---------------------------------------------------------------------------


_PREFILL_US_PER_TOK = {"fast": 0.1, "slow": 1.0}
_DECODE_US_PER_STEP = {"fast": 5.0, "slow": 50.0}


class _StubPredictor:
    def __init__(self, model, hw, config, **kwargs):
        self._prefill_us = _PREFILL_US_PER_TOK[hw.name]
        self._decode_us = _DECODE_US_PER_STEP[hw.name]

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self._prefill_us * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        if isinstance(past_kv_length, int):
            pkv_total = past_kv_length * batch_size
        else:
            pkv_total = sum(int(x) for x in past_kv_length)
        return (self._decode_us * batch_size + 0.001 * pkv_total) * 1e-6


class _StubHW:
    def __init__(self, name):
        self.name = name


def _hw(name):
    return _StubHW(name)


def _model() -> ModelInfo:
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="parity-stub",
    )


def _base() -> SchedulerConfig:
    return SchedulerConfig(
        model=_model(),
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )


def _disagg_cfg(prefill_replicas=2, decode_replicas=2):
    return DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(
            device_name="fast",
            tp_size=1,
            replicas=prefill_replicas,
            max_running_per_replica=8,
        ),
        decode=RolePredictorConfig(
            device_name="fast",
            tp_size=1,
            replicas=decode_replicas,
            max_running_per_replica=64,
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=100.0, latency_us=10.0),
    )


def _bundle(prefill_replicas=2, decode_replicas=2):
    return build_disagg(
        model=_model(),
        base_sched_config=_base(),
        disagg_config=_disagg_cfg(prefill_replicas, decode_replicas),
        predictor_factory=_StubPredictor,
        hw_factory=_hw,
    )


# Top-level factories for BackendB workers.
def make_prefill_predictor():
    return _StubPredictor(model=None, hw=_StubHW("fast"), config=None)


def make_decode_predictor():
    return _StubPredictor(model=None, hw=_StubHW("fast"), config=None)


def _req(rid: str, input_len: int = 1024, output_len: int = 2) -> PDRequestState:
    return PDRequestState(
        rid=rid,
        arrival_time=0.0,
        input_length=input_len,
        output_length=output_len,
    )


# ---------------------------------------------------------------------------
# Backend fixture: yields a started backend, shuts it down on teardown.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["backend_a", "backend_b"])
def backend(request):
    prefill_replicas = 2
    decode_replicas = 2
    bundle = _bundle(prefill_replicas, decode_replicas)
    if request.param == "backend_a":
        yield BackendA(bundle)
        return
    be = BackendB(
        bundle=bundle,
        prefill_predictor_factory=make_prefill_predictor,
        decode_predictor_factory=make_decode_predictor,
    )
    be.start()
    try:
        yield be
    finally:
        be.shutdown()


# ---------------------------------------------------------------------------
# Parity assertions
# ---------------------------------------------------------------------------


def test_initial_pool_state(backend):
    assert backend.prefill_pool_size() == 2
    assert backend.decode_pool_size() == 2
    assert backend.earliest_pool_time("prefill") == 0.0
    assert backend.earliest_pool_time("decode") == 0.0


def test_unknown_pool_raises(backend):
    with pytest.raises(ValueError):
        backend.earliest_pool_time("garbage")


def test_prefill_admission_matches_predictor(backend):
    r = _req("r0", input_len=1000)
    idx, end_t = backend.try_admit_prefill(r, now=0.0)
    assert idx == 0
    assert end_t == pytest.approx(1e-4)  # 0.1 us/tok * 1000 = 100us
    assert r.phase == RequestPhase.RUNNING_PREFILL


def test_prefill_parallel_on_two_replicas(backend):
    a = _req("a", input_len=1000)
    b = _req("b", input_len=1000)
    ia, ea = backend.try_admit_prefill(a, now=0.0)
    ib, eb = backend.try_admit_prefill(b, now=0.0)
    assert ia != ib
    assert ea == eb == pytest.approx(1e-4)


def test_decode_batch_admission(backend):
    reqs = [
        _req("a", input_len=10),
        _req("b", input_len=10),
        _req("c", input_len=10),
    ]
    for r in reqs:
        r.current_past_kv_length = 100
    idx, end_t = backend.try_admit_decode_batch(reqs, now=0.0)
    # 5us/slot * 3 + 0.001us * 300 = 15.3us
    expected = (5 * 3 + 0.001 * 300) * 1e-6
    assert end_t == pytest.approx(expected, rel=1e-9)


def test_full_lifecycle_finishes(backend):
    r = _req("r", input_len=100, output_len=2)
    _, prefill_end = backend.try_admit_prefill(r, now=0.0)
    kv_ready = backend.compute_kv_ready_time(r, now=prefill_end)
    backend.on_prefill_done(r, now=prefill_end, kv_ready_time=kv_ready)
    assert r.phase == RequestPhase.KV_TRANSIT
    backend.advance_to_kv_ready(r, now=kv_ready)
    assert r.phase == RequestPhase.WAITING_DECODE
    for _ in range(r.output_length):
        _, end_t = backend.try_admit_decode_step(
            r, now=backend.earliest_pool_time("decode")
        )
        backend.on_decode_step_done(r, now=end_t)
    assert r.phase == RequestPhase.FINISHED


# ---------------------------------------------------------------------------
# Batch prefill parity (try_admit_prefill_batch + compute_batch_kv_ready_time)
# ---------------------------------------------------------------------------


def test_batch_prefill_uses_sum_tokens(backend):
    """try_admit_prefill_batch must call the predictor with total token count."""
    reqs = [_req(f"r{i}", input_len=200) for i in range(4)]  # 800 total
    idx, end_t = backend.try_admit_prefill_batch(reqs, now=0.0)
    # fast: 0.1 us/tok * 800 = 80 us
    assert end_t == pytest.approx(8e-5)


def test_batch_prefill_all_requests_share_timestamps(backend):
    reqs = [_req("a", input_len=100), _req("b", input_len=300)]
    _, end_t = backend.try_admit_prefill_batch(reqs, now=0.0)
    assert all(r.prefill_end_time == pytest.approx(end_t) for r in reqs)
    assert all(r.prefill_start_time == pytest.approx(0.0) for r in reqs)
    assert all(r.phase == RequestPhase.RUNNING_PREFILL for r in reqs)


def test_batch_prefill_occupies_single_replica(backend):
    reqs = [_req(f"r{i}") for i in range(3)]
    _, end_t = backend.try_admit_prefill_batch(reqs, now=0.0)
    # The batch was scheduled (took positive time)
    assert end_t > 0.0
    # With 2 prefill replicas, the other replica stays free; earliest is still 0
    assert backend.earliest_pool_time("prefill") == 0.0


def test_batch_prefill_rejects_empty(backend):
    with pytest.raises(ValueError):
        backend.try_admit_prefill_batch([], now=0.0)


def test_compute_batch_kv_ready_time_uses_total_tokens(backend):
    """Batch KV ready time must scale with total_tokens, not max per-request."""
    # kv_bytes_per_token = 131072 (FP16, TP=1), bw=100Gbps, latency=10us
    total_tokens = 8
    expected = 10e-6 + 8 * 131072 / 100e9
    assert backend.compute_batch_kv_ready_time(total_tokens, now=0.0) == pytest.approx(expected)

