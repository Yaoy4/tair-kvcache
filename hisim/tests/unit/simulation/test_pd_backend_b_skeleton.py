"""Phase 5a — Backend B skeleton tests.

Validate process lifecycle, predictor IPC, multi-replica load spread, and
drop-in API parity with Backend A. Uses a small analytic predictor inside
worker processes so the suite stays fast and DB-free.

These tests run on Linux/macOS using the 'spawn' multiprocessing start method.
Each test that needs workers uses BackendB as a context manager so shutdown
is guaranteed.
"""

from __future__ import annotations

import pytest

from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_backend_b import BackendB
from hisim.simulation.pd_factory import DisaggPredictors
from hisim.simulation.pd_transfer import BandwidthTransferModel, KVModelConfig
from hisim.simulation.pd_types import PDRequestState, RequestPhase


# ---------------------------------------------------------------------------
# Top-level analytic predictors (picklable for multiprocessing 'spawn').
# ---------------------------------------------------------------------------


_PREFILL_US_PER_TOK = 0.5
_DECODE_US_PER_STEP = 15.0


class _AnalyticPrefillPredictor:
    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * _PREFILL_US_PER_TOK * 1e-6


class _AnalyticDecodePredictor:
    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        base = _DECODE_US_PER_STEP * (1 + 0.05 * max(0, batch_size - 1))
        return base * 1e-6


def make_prefill_predictor():
    return _AnalyticPrefillPredictor()


def make_decode_predictor():
    return _AnalyticDecodePredictor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(
    prefill_replicas: int = 1, decode_replicas: int = 1
) -> DisaggPredictors:
    return DisaggPredictors(
        prefill=_AnalyticPrefillPredictor(),
        decode=_AnalyticDecodePredictor(),
        kv_bytes_per_token=1024,
        kv_model_config=KVModelConfig(kv_bytes_per_token=1024),
        transfer_model=BandwidthTransferModel(bw_gbps=200.0, latency_us=0.0),
        prefill_replicas=prefill_replicas,
        decode_replicas=decode_replicas,
    )


def _req(rid: str = "r1", input_len: int = 1024) -> PDRequestState:
    return PDRequestState(rid=rid, arrival_time=0.0, input_length=input_len)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lifecycle_start_and_shutdown():
    backend = BackendB(
        bundle=_make_bundle(prefill_replicas=2, decode_replicas=3),
        prefill_predictor_factory=make_prefill_predictor,
        decode_predictor_factory=make_decode_predictor,
    )
    backend.start()
    try:
        assert backend.prefill_pool_size() == 2
        assert backend.decode_pool_size() == 3
        assert all(w.process.is_alive() for w in backend._prefill_workers)
        assert all(w.process.is_alive() for w in backend._decode_workers)
    finally:
        backend.shutdown()
    assert backend._prefill_workers == []
    assert backend._decode_workers == []
    assert backend._started is False


def test_try_admit_prefill_returns_predictor_duration():
    with BackendB(
        bundle=_make_bundle(),
        prefill_predictor_factory=make_prefill_predictor,
        decode_predictor_factory=make_decode_predictor,
    ) as backend:
        req = _req(input_len=2000)
        idx, end = backend.try_admit_prefill(req, now=0.0)
        assert idx == 0
        expected = 2000 * _PREFILL_US_PER_TOK * 1e-6
        assert end == pytest.approx(expected, rel=1e-9)
        assert req.phase == RequestPhase.RUNNING_PREFILL
        assert backend.earliest_pool_time("prefill") == pytest.approx(end)


def test_try_admit_decode_batch_uses_predictor():
    with BackendB(
        bundle=_make_bundle(),
        prefill_predictor_factory=make_prefill_predictor,
        decode_predictor_factory=make_decode_predictor,
    ) as backend:
        r1 = _req("r1", input_len=10)
        r1.current_past_kv_length = 10
        r2 = _req("r2", input_len=20)
        r2.current_past_kv_length = 20
        idx, end = backend.try_admit_decode_batch([r1, r2], now=0.0)
        assert idx == 0
        expected = _DECODE_US_PER_STEP * (1 + 0.05 * 1) * 1e-6
        assert end == pytest.approx(expected, rel=1e-9)


def test_multi_replica_load_spread():
    """Two prefill replicas → two requests land on different workers."""
    with BackendB(
        bundle=_make_bundle(prefill_replicas=2),
        prefill_predictor_factory=make_prefill_predictor,
        decode_predictor_factory=make_decode_predictor,
    ) as backend:
        idx_a, end_a = backend.try_admit_prefill(_req("a", input_len=1000), now=0.0)
        idx_b, end_b = backend.try_admit_prefill(_req("b", input_len=1000), now=0.0)
        assert {idx_a, idx_b} == {0, 1}
        # Both end at the same time: same start (now=0), same predicted duration.
        assert end_a == pytest.approx(end_b, rel=1e-9)


def test_api_parity_with_backend_a():
    """Same predictor + same bundle → BackendA and BackendB return identical end_time."""
    bundle_a = _make_bundle()
    bundle_b = _make_bundle()
    a = BackendA(bundle_a)
    idx_a, end_a = a.try_admit_prefill(_req("x", input_len=500), now=0.0)
    with BackendB(
        bundle=bundle_b,
        prefill_predictor_factory=make_prefill_predictor,
        decode_predictor_factory=make_decode_predictor,
    ) as b:
        idx_b, end_b = b.try_admit_prefill(_req("x", input_len=500), now=0.0)
    assert idx_a == idx_b == 0
    assert end_a == pytest.approx(end_b, rel=1e-9)


def test_use_before_start_raises():
    backend = BackendB(
        bundle=_make_bundle(),
        prefill_predictor_factory=make_prefill_predictor,
        decode_predictor_factory=make_decode_predictor,
    )
    with pytest.raises(RuntimeError, match="start"):
        backend.try_admit_prefill(_req(), now=0.0)
