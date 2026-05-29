"""Phase 5b.1 — Real-AIC-DB spawn smoke test for BackendB + worker-error path.

Why: BackendB workers are spawned with ``multiprocessing.get_context("spawn")``,
which re-imports modules in a fresh interpreter. ``AIConfiguratorTimePredictor``
opens a real perf DB at construction time. If the worker dies during boot or
during a predict call, the parent must NOT block forever on ``out_q.get()``.
Backend B forwards worker exceptions as ``_Result(error=...)`` and re-raises
them in the parent as ``BackendBWorkerError``.

Two tests here:
    * a real-AIC-DB smoke test (skip-if-no-DB),
    * a worker-error-propagation test that uses a top-level factory which
      raises at construction time, proving the parent surfaces the error
      instead of hanging.
"""

from __future__ import annotations

import pytest

from hisim.simulation.pd_aic_factory import AICPredictorFactory
from hisim.simulation.pd_backend_b import BackendB, BackendBWorkerError
from hisim.simulation.pd_config import RolePredictorConfig
from hisim.simulation.pd_factory import DisaggPredictors
from hisim.simulation.pd_transfer import BandwidthTransferModel, KVModelConfig
from hisim.simulation.pd_types import PDRequestState


# ---------------------------------------------------------------------------
# Top-level helpers (picklable for 'spawn').
# ---------------------------------------------------------------------------


def _make_model():
    from hisim.spec import ModelInfo

    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="llama-7b-ish-smoke",
    )


def _make_base_sched(backend_version: str):
    from hisim.spec import DataType
    from hisim.simulation.types import SchedulerConfig

    return SchedulerConfig(
        model=_make_model(),
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version=backend_version,
    )


def _make_aic_factory(device_name: str, backend_version: str) -> AICPredictorFactory:
    role = RolePredictorConfig(
        device_name=device_name,
        tp_size=1,
        replicas=1,
        max_running_per_replica=8,
    )
    return AICPredictorFactory(
        model=_make_model(),
        base_sched_config=_make_base_sched(backend_version),
        role=role,
    )


def _empty_bundle() -> DisaggPredictors:
    return DisaggPredictors(
        prefill=None,
        decode=None,
        kv_bytes_per_token=1024,
        kv_model_config=KVModelConfig(kv_bytes_per_token=1024),
        transfer_model=BandwidthTransferModel(bw_gbps=200.0, latency_us=10.0),
        prefill_replicas=1,
        decode_replicas=1,
    )


# ---------------------------------------------------------------------------
# Worker-error propagation (no DB required — runs unconditionally).
# ---------------------------------------------------------------------------


class _BoomFactory:
    """Picklable factory whose __call__ always raises (worker construction fails)."""

    def __call__(self):
        raise RuntimeError("boom-during-construction")


def test_worker_construction_error_surfaces_in_parent():
    """If the predictor factory raises in the worker, the parent must
    re-raise it as BackendBWorkerError instead of blocking forever."""
    backend = BackendB(
        bundle=_empty_bundle(),
        prefill_predictor_factory=_BoomFactory(),
        decode_predictor_factory=_BoomFactory(),
    )
    backend.start()
    try:
        req = PDRequestState(rid="boom", arrival_time=0.0, input_length=1)
        with pytest.raises(BackendBWorkerError, match="boom-during-construction"):
            backend.try_admit_prefill(req, now=0.0)
    finally:
        backend.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# Real AIC DB smoke test — probe by attempting in-process construction.
# ---------------------------------------------------------------------------


def _find_usable_aic_combo():
    try:
        from hisim.spec import AcceleratorInfo
        from hisim.time_predictor.aiconfigurator import _get_default_systems_paths
    except Exception:
        return None
    if not _get_default_systems_paths():
        return None
    devices = [
        n for n in ("H20", "NVIDIA H20", "h100_sxm", "h20_sxm")
        if AcceleratorInfo.find_by_hw_name(n) is not None
    ]
    versions = ["0.4.8", "0.4.7", "0.4.6", "0.4.5", "0.4.4", "0.4.3"]
    for d in devices:
        for v in versions:
            try:
                _make_aic_factory(d, v)()
                return (d, v)
            except Exception:
                continue
    return None


_COMBO = _find_usable_aic_combo()


@pytest.mark.skipif(
    _COMBO is None,
    reason="no usable (device, backend_version) combo in local AIC DB",
)
def test_backend_b_spawns_real_aic_worker_and_predicts():
    """One real AIC predictor worker; one prefill call; expect positive duration."""
    device_name, backend_version = _COMBO
    factory = _make_aic_factory(device_name, backend_version)
    backend = BackendB(
        bundle=_empty_bundle(),
        prefill_predictor_factory=factory,
        decode_predictor_factory=factory,
    )
    backend.start()
    try:
        req = PDRequestState(rid="smoke", arrival_time=0.0, input_length=1024)
        idx, end_t = backend.try_admit_prefill(req, now=0.0)
        assert idx == 0
        assert end_t > 0.0
        assert end_t < 1.0, f"prefill duration {end_t} s suspiciously large"
    finally:
        backend.shutdown(timeout=10.0)
