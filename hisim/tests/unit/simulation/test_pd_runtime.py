"""Phase 2b.3 — tests for build_backend_a (PD runtime builder).

`build_backend_a(model, base_sched_config, disagg_config)` is the helper the
SGLang hook uses at init time. By default it wraps real
AIConfiguratorTimePredictor instances in AICPredictorAdapter, but it accepts
injected `predictor_factory` / `hw_factory` for unit testing.
"""
import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_runtime import build_backend_a
from hisim.simulation.types import SchedulerConfig


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
    def __init__(self, model, hw, config, **kwargs):
        self.hw = hw

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return 1e-6 * batch_tokens

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        return 5e-6 * batch_size


class _StubHW:
    def __init__(self, name):
        self.name = name


def _cfg():
    return DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(device_name="p", tp_size=2, replicas=2),
        decode=RolePredictorConfig(device_name="d", tp_size=2, replicas=4),
        kv_transfer=BandwidthTransferConfig(bw_gbps=50.0, latency_us=20.0),
    )


def test_build_backend_a_returns_backend_with_correct_pools():
    be = build_backend_a(
        model=_model(),
        base_sched_config=_base(),
        disagg_config=_cfg(),
        predictor_factory=_StubPredictor,
        hw_factory=lambda n: _StubHW(n),
    )
    assert isinstance(be, BackendA)
    assert be.prefill_pool_size() == 2
    assert be.decode_pool_size() == 4


def test_build_backend_a_raises_when_disagg_disabled():
    cfg = _cfg()
    cfg.enabled = False
    with pytest.raises(ValueError):
        build_backend_a(
            model=_model(),
            base_sched_config=_base(),
            disagg_config=cfg,
            predictor_factory=_StubPredictor,
            hw_factory=lambda n: _StubHW(n),
        )


def test_build_backend_a_predictor_protocol_works_end_to_end():
    """Smoke: the stub predictor's predict_decode_seconds is callable via the
    batch-aware path BackendA uses in 2b.2.
    """
    be = build_backend_a(
        model=_model(),
        base_sched_config=_base(),
        disagg_config=_cfg(),
        predictor_factory=_StubPredictor,
        hw_factory=lambda n: _StubHW(n),
    )
    from hisim.simulation.pd_types import PDRequestState, RequestPhase

    reqs = [
        PDRequestState(
            rid=f"r{i}",
            arrival_time=0.0,
            phase=RequestPhase.RUNNING_DECODE,
            input_length=10,
            output_length=1,
            current_past_kv_length=10,
        )
        for i in range(3)
    ]
    _, end_t = be.try_admit_decode_batch(reqs, now=0.0)
    # stub: 5us * 3 slots = 15us
    assert end_t == pytest.approx(15e-6)
