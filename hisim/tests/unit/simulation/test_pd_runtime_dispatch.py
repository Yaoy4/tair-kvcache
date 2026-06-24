"""Phase 5c.1 — `build_pd_backend` dispatcher + BackendB build path.

These tests pin the contract that the SGLang hook gets a single, typed
entry point (``build_pd_backend``) that picks BackendA or BackendB based on
``disagg.backend``, with no SGLang-specific code in the dispatcher itself.

Coverage:

  * Disabled config → dispatcher refuses.
  * Unknown backend string → dispatcher refuses with a helpful message.
  * ``single_process`` → returns a BackendA whose pool sizes match config.
  * ``two_process`` → returns a (not-yet-started) BackendB whose pool sizes
    match config. Verified without spawning real workers.
  * End-to-end ``two_process`` with a stub role factory: build → start →
    prefill predict → shutdown completes within a tight wall-clock budget
    and produces a positive duration.
  * ``build_backend_b`` independently rejects disabled configs (defense in
    depth: the dispatcher's guard is not the only one).
"""
from __future__ import annotations

import time

import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_backend_b import BackendB
from hisim.simulation.pd_backend_protocol import PDBackendProtocol
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_runtime import (
    build_backend_a,
    build_backend_b,
    build_pd_backend,
)
from hisim.simulation.pd_types import PDRequestState
from hisim.simulation.types import SchedulerConfig


# ---------------------------------------------------------------------------
# Helpers (top-level so they survive spawn).
# ---------------------------------------------------------------------------


def _make_model() -> ModelInfo:
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="phase5c1-stub",
    )


def _make_base_sched() -> SchedulerConfig:
    return SchedulerConfig(
        model=_make_model(),
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )


def _make_disagg_cfg(
    *, backend: str = "single_process",
    prefill_replicas: int = 2,
    decode_replicas: int = 3,
) -> DisaggConfig:
    role = RolePredictorConfig(
        device_name="H20",
        tp_size=1,
        replicas=prefill_replicas,
        max_running_per_replica=8,
    )
    return DisaggConfig(
        enabled=True,
        backend=backend,
        prefill=role,
        decode=RolePredictorConfig(
            device_name="H20",
            tp_size=1,
            replicas=decode_replicas,
            max_running_per_replica=8,
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=100.0, latency_us=10.0),
    )


# ---- Stub predictor for BackendA path (no AIC DB needed). ----


class _StubHW:
    def __init__(self, name: str):
        self.name = name


def _stub_hw_factory(name: str) -> _StubHW:
    return _StubHW(name)


class _AnalyticPredictor:
    def __init__(self, model, hw, config, **kwargs):
        pass

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        return 1e-5 * max(1, batch_size)


# ---- Stub role-factory for BackendB path (no AIC DB; picklable). ----


class _AnalyticAdapterFactory:
    """Picklable factory: returns a predictor with the BackendB-required API."""

    def __call__(self):
        # BackendB workers call ``predict_prefill_seconds`` /
        # ``predict_decode_seconds`` directly — no AIC adapter wrap needed.
        return _AnalyticPredictor(model=None, hw=None, config=None)


def _stub_role_factory(model, base_sched_config, role):
    """Top-level role factory: ignores role data, returns a picklable factory."""
    return _AnalyticAdapterFactory()


# ---------------------------------------------------------------------------
# Disabled / unknown-backend guards.
# ---------------------------------------------------------------------------


def test_build_pd_backend_refuses_when_disabled():
    with pytest.raises(ValueError, match="enabled=False"):
        build_pd_backend(
            model=_make_model(),
            base_sched_config=_make_base_sched(),
            disagg_config=DisaggConfig(),
        )


def test_build_pd_backend_rejects_unknown_backend_name():
    # Bypass DisaggConfig.__post_init__ validation by mutating after construct.
    cfg = _make_disagg_cfg()
    object.__setattr__(cfg, "backend", "quantum_process")
    with pytest.raises(ValueError, match="quantum_process"):
        build_pd_backend(
            model=_make_model(),
            base_sched_config=_make_base_sched(),
            disagg_config=cfg,
        )


def test_build_backend_b_refuses_when_disabled():
    with pytest.raises(ValueError, match="enabled=False"):
        build_backend_b(
            model=_make_model(),
            base_sched_config=_make_base_sched(),
            disagg_config=DisaggConfig(),
        )


# ---------------------------------------------------------------------------
# Dispatcher returns the right concrete backend with the right pool sizes.
# ---------------------------------------------------------------------------


def test_dispatcher_single_process_returns_backend_a():
    cfg = _make_disagg_cfg(
        backend="single_process", prefill_replicas=2, decode_replicas=3
    )
    backend = build_pd_backend(
        model=_make_model(),
        base_sched_config=_make_base_sched(),
        disagg_config=cfg,
        predictor_factory=_AnalyticPredictor,
        hw_factory=_stub_hw_factory,
    )
    assert isinstance(backend, BackendA)
    assert isinstance(backend, PDBackendProtocol)
    assert backend.prefill_pool_size() == 2
    assert backend.decode_pool_size() == 3


def test_dispatcher_two_process_returns_backend_b_without_starting():
    cfg = _make_disagg_cfg(
        backend="two_process", prefill_replicas=2, decode_replicas=3
    )
    backend = build_pd_backend(
        model=_make_model(),
        base_sched_config=_make_base_sched(),
        disagg_config=cfg,
        role_factory=_stub_role_factory,
        hw_factory=_stub_hw_factory,
    )
    assert isinstance(backend, BackendB)
    assert isinstance(backend, PDBackendProtocol)
    assert backend.prefill_pool_size() == 2
    assert backend.decode_pool_size() == 3
    # MUST NOT auto-start; calling any try_admit_* before start() must fail.
    with pytest.raises(RuntimeError, match="start"):
        backend.try_admit_prefill(
            PDRequestState(rid="x", arrival_time=0.0, input_length=1), now=0.0
        )


# ---------------------------------------------------------------------------
# End-to-end two_process build → start → predict → shutdown.
# ---------------------------------------------------------------------------


def test_dispatcher_two_process_end_to_end_with_stub_role_factory():
    cfg = _make_disagg_cfg(
        backend="two_process", prefill_replicas=1, decode_replicas=1
    )
    backend = build_pd_backend(
        model=_make_model(),
        base_sched_config=_make_base_sched(),
        disagg_config=cfg,
        role_factory=_stub_role_factory,
        hw_factory=_stub_hw_factory,
    )
    t0 = time.time()
    backend.start()
    try:
        req = PDRequestState(rid="e2e", arrival_time=0.0, input_length=1024)
        idx, end_t = backend.try_admit_prefill(req, now=0.0)
        assert idx == 0
        # Analytic predictor: 1024 * 1e-6 = ~1.024e-3 s.
        assert 0.0 < end_t < 0.01
    finally:
        backend.shutdown(timeout=5.0)
    elapsed = time.time() - t0
    # Spawn + 1 call + shutdown should be well under 15s on any dev box.
    assert elapsed < 15.0, f"e2e took {elapsed:.1f}s, suspiciously slow"
