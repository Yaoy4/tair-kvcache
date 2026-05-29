"""Phase 5c.2 — PD backend lifecycle helpers.

``start_pd_backend`` and ``shutdown_pd_backend`` are the only points where
worker lifecycle is owned. These tests pin:

  * BackendA path: start is a no-op, no atexit handler is registered.
  * BackendB path: start spawns workers, atexit handler is registered, and
    shutdown is idempotent (multiple calls do not raise).
  * ``shutdown_pd_backend`` is None-safe and exception-safe (atexit handlers
    that raise would corrupt other handlers).
  * After ``start_pd_backend`` returns, the backend is fully usable
    (``try_admit_prefill`` works without ``RuntimeError: start required``).
"""
from __future__ import annotations

import atexit

import pytest

from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_backend_b import BackendB
from hisim.simulation.pd_factory import DisaggPredictors
from hisim.simulation.pd_runtime import shutdown_pd_backend, start_pd_backend
from hisim.simulation.pd_transfer import BandwidthTransferModel, KVModelConfig
from hisim.simulation.pd_types import PDRequestState


# ---------------------------------------------------------------------------
# Helpers (top-level → picklable for spawn).
# ---------------------------------------------------------------------------


class _AnalyticPredictor:
    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        return 1e-5 * max(1, batch_size)


class _PicklablePredictorFactory:
    def __call__(self):
        return _AnalyticPredictor()


def _empty_bundle(
    *, prefill_replicas: int = 1, decode_replicas: int = 1,
    with_predictor: bool = False,
) -> DisaggPredictors:
    pred = _AnalyticPredictor() if with_predictor else None
    return DisaggPredictors(
        prefill=pred,
        decode=pred,
        kv_bytes_per_token=1024,
        kv_model_config=KVModelConfig(kv_bytes_per_token=1024),
        transfer_model=BandwidthTransferModel(bw_gbps=100.0, latency_us=10.0),
        prefill_replicas=prefill_replicas,
        decode_replicas=decode_replicas,
    )


class _BoomBackend:
    """Backend stub whose shutdown() raises — used to prove atexit safety."""

    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1
        raise RuntimeError("shutdown blew up")


# ---------------------------------------------------------------------------
# shutdown_pd_backend: None-safe + exception-safe.
# ---------------------------------------------------------------------------


def test_shutdown_pd_backend_is_none_safe():
    # Must not raise — atexit handler must survive a None backend.
    shutdown_pd_backend(None)


def test_shutdown_pd_backend_swallows_exceptions():
    """An atexit handler that raises would crash other handlers. The
    helper must catch and swallow exceptions."""
    boom = _BoomBackend()
    shutdown_pd_backend(boom)  # must not raise
    assert boom.shutdown_calls == 1


def test_shutdown_pd_backend_no_shutdown_method_is_noop():
    """BackendA-style objects without .shutdown must be tolerated."""

    class _NoShutdown:
        pass

    shutdown_pd_backend(_NoShutdown())  # must not raise


# ---------------------------------------------------------------------------
# start_pd_backend on BackendA: no-op, no atexit handler registered.
# ---------------------------------------------------------------------------


@pytest.fixture
def capture_atexit(monkeypatch):
    """Capture every ``atexit.register`` call during the test, and ensure
    those handlers do NOT actually run at interpreter exit (would pollute
    other tests)."""
    calls: list[tuple] = []
    real_register = atexit.register

    def fake_register(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        # Intentionally do not forward to real_register: we don't want test
        # backends to be shutdown at interpreter exit (they're already
        # explicitly shutdown in the test's finally block).
        return func

    monkeypatch.setattr(atexit, "register", fake_register)
    yield calls


def test_start_pd_backend_on_backend_a_does_not_register_atexit(capture_atexit):
    backend = BackendA(_empty_bundle(with_predictor=True))
    ret = start_pd_backend(backend)
    assert ret is backend
    targeting = [c for c in capture_atexit if c[1] and c[1][0] is backend]
    assert not targeting, "BackendA should not register an atexit handler"
    # BackendA was already usable; start_pd_backend must not have broken it.
    req = PDRequestState(rid="a", arrival_time=0.0, input_length=8)
    idx, end_t = backend.try_admit_prefill(req, now=0.0)
    assert idx == 0
    assert end_t >= 0.0


# ---------------------------------------------------------------------------
# start_pd_backend on BackendB: starts, registers atexit, idempotent shutdown.
# ---------------------------------------------------------------------------


def test_start_pd_backend_on_backend_b_starts_and_registers_atexit(capture_atexit):
    factory = _PicklablePredictorFactory()
    backend = BackendB(
        bundle=_empty_bundle(),
        prefill_predictor_factory=factory,
        decode_predictor_factory=factory,
    )
    try:
        ret = start_pd_backend(backend)
        assert ret is backend
        targeting = [c for c in capture_atexit if c[1] and c[1][0] is backend]
        assert len(targeting) == 1, (
            f"BackendB should register exactly one atexit shutdown handler, "
            f"got {len(targeting)}"
        )
        # The registered handler must be shutdown_pd_backend.
        assert targeting[0][0] is shutdown_pd_backend
        # Backend is usable post-start (no RuntimeError("start required")).
        req = PDRequestState(rid="b", arrival_time=0.0, input_length=64)
        idx, end_t = backend.try_admit_prefill(req, now=0.0)
        assert idx == 0
        assert end_t > 0.0
    finally:
        shutdown_pd_backend(backend)
        # Calling shutdown twice in a row must not raise.
        shutdown_pd_backend(backend)


def test_start_pd_backend_double_call_is_safe(capture_atexit):
    """Calling start_pd_backend twice on the same BackendB instance should
    not double-start workers (start() is internally idempotent) and the
    test must not hang."""
    factory = _PicklablePredictorFactory()
    backend = BackendB(
        bundle=_empty_bundle(),
        prefill_predictor_factory=factory,
        decode_predictor_factory=factory,
    )
    try:
        start_pd_backend(backend)
        start_pd_backend(backend)
        req = PDRequestState(rid="dup", arrival_time=0.0, input_length=4)
        idx, end_t = backend.try_admit_prefill(req, now=0.0)
        assert idx == 0
        assert end_t > 0.0
    finally:
        shutdown_pd_backend(backend)
