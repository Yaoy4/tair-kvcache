"""Phase 5b — AICPredictorFactory unit tests.

Validates the factory is picklable, captures only serializable inputs, and
constructs the predictor + adapter lazily inside ``__call__`` using injectable
hw/predictor factories so we can run without a real AIConfigurator DB.
"""

from __future__ import annotations

import pickle

import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_aic_adapter import AICPredictorAdapter
from hisim.simulation.pd_aic_factory import AICPredictorFactory
from hisim.simulation.pd_config import RolePredictorConfig
from hisim.simulation.types import SchedulerConfig


# ---------------------------------------------------------------------------
# Top-level stubs (must be picklable; AICPredictorFactory holds dotted names).
# ---------------------------------------------------------------------------


class StubHW:
    def __init__(self, name: str):
        self.name = name


def stub_hw_factory(name: str) -> StubHW:
    return StubHW(name)


class StubPredictor:
    def __init__(self, model, hw, config, **kwargs):
        self.model = model
        self.hw = hw
        self.config = config
        self.kwargs = kwargs

    def predict_infer_time(self, batch) -> float:  # pragma: no cover - unused here
        return 1e-6


def stub_predictor_factory(model, hw, config, **kwargs):
    return StubPredictor(model, hw, config, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model() -> ModelInfo:
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="aic-factory-stub",
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


def _role() -> RolePredictorConfig:
    return RolePredictorConfig(
        device_name="my-device",
        tp_size=2,
        ep_size=1,
        dp_size=1,
        pp_size=1,
        replicas=1,
        max_running_per_replica=8,
        database_path="/tmp/fake",
        prefill_scale_factor=1.5,
        decode_scale_factor=2.0,
    )


def _factory() -> AICPredictorFactory:
    mod = __name__
    return AICPredictorFactory(
        model=_model(),
        base_sched_config=_base(),
        role=_role(),
        hw_factory_dotted=f"{mod}.stub_hw_factory",
        predictor_factory_dotted=f"{mod}.stub_predictor_factory",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_is_picklable():
    f = _factory()
    blob = pickle.dumps(f)
    f2 = pickle.loads(blob)
    assert isinstance(f2, AICPredictorFactory)
    assert f2.role.device_name == "my-device"
    assert f2.role.tp_size == 2


def test_call_returns_adapter_wrapping_predictor():
    adapter = _factory()()
    assert isinstance(adapter, AICPredictorAdapter)
    # AICPredictorAdapter holds a `_base` reference to the wrapped predictor.
    assert isinstance(adapter._base, StubPredictor)


def test_call_passes_role_kwargs_to_predictor():
    adapter = _factory()()
    base = adapter._base
    assert base.kwargs.get("database_path") == "/tmp/fake"
    assert base.kwargs.get("prefill_scale_factor") == 1.5
    assert base.kwargs.get("decode_scale_factor") == 2.0


def test_call_applies_role_overrides_to_sched_config():
    adapter = _factory()()
    sched = adapter._base.config
    assert sched.tp_size == 2  # from role
    assert sched.max_running_requests == 8


def test_call_resolves_device_name_via_hw_factory():
    adapter = _factory()()
    assert isinstance(adapter._base.hw, StubHW)
    assert adapter._base.hw.name == "my-device"


def test_invalid_dotted_path_raises():
    f = AICPredictorFactory(
        model=_model(),
        base_sched_config=_base(),
        role=_role(),
        hw_factory_dotted="not_a_dotted_path",
    )
    with pytest.raises(ValueError, match="dotted"):
        f()
