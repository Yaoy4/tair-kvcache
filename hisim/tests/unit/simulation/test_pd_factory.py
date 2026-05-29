"""Phase 1c: tests for PD predictor/transfer-model factory.

Verifies that, given a DisaggConfig, we can build two role-specific predictors
and derive kv_bytes_per_token via the existing utils helpers — never
hand-picked. Real AIConfiguratorTimePredictor construction (which loads perf
databases) is mocked behind an injectable factory.
"""

import pytest

from hisim.spec import ModelInfo, DataType
from hisim.simulation.types import SchedulerConfig
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_factory import (
    DisaggPredictors,
    build_disagg,
    calc_kv_bytes_per_token,
)
from hisim.simulation.pd_transfer import BandwidthTransferModel


# --- helpers ----------------------------------------------------------------


def _mha_model(num_kv_heads: int = 8, head_dim: int = 128, num_layers: int = 32):
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=num_layers,
        vocab_size=32000,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_full_attention=num_layers,
        name="test-mha",
    )


def _mla_model(kv_lora_rank: int = 512, qk_rope_head_dim: int = 64, num_layers: int = 60):
    return ModelInfo(
        hidden_size=5120,
        num_attention_heads=128,
        num_hidden_layers=num_layers,
        vocab_size=32000,
        num_key_value_heads=128,
        head_dim=128,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_full_attention=num_layers,
        name="test-mla",
    )


def _base_sched(model):
    return SchedulerConfig(
        model=model,
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )


class _StubPredictor:
    """Lightweight stand-in for AIConfiguratorTimePredictor."""

    def __init__(self, model, hw, config, **kwargs):
        self.model = model
        self.hw = hw
        self.config = config
        self.kwargs = kwargs


class _StubHW:
    def __init__(self, name):
        self.name = name


def _hw_factory(name):
    return _StubHW(name)


# --- calc_kv_bytes_per_token ------------------------------------------------


def test_calc_kv_bytes_per_token_mha_fp16_tp1():
    model = _mha_model(num_kv_heads=8, head_dim=128, num_layers=32)
    # MHA cell elems = num_kv_heads * head_dim * num_layers * 2
    # = 8 * 128 * 32 * 2 = 65536 elems; FP16 = 2 bytes → 131072 bytes
    assert calc_kv_bytes_per_token(model, tp_size=1, pp_size=1,
                                   kv_cache_dtype=DataType.FP16) == 131072


def test_calc_kv_bytes_per_token_mha_with_tp():
    model = _mha_model(num_kv_heads=8, head_dim=128, num_layers=32)
    # tp=4 → num_kv_heads becomes max(8//4, 1) = 2
    # cell elems = 2 * 128 * 32 * 2 = 16384; FP16 = 2 bytes → 32768
    assert calc_kv_bytes_per_token(model, tp_size=4, pp_size=1,
                                   kv_cache_dtype=DataType.FP16) == 32768


def test_calc_kv_bytes_per_token_mla_ignores_tp():
    model = _mla_model(kv_lora_rank=512, qk_rope_head_dim=64, num_layers=60)
    # MLA cell elems = (512 + 64) * 60 = 34560; FP8 = 1 byte → 34560
    assert calc_kv_bytes_per_token(model, tp_size=8, pp_size=1,
                                   kv_cache_dtype=DataType.FP8) == 34560


# --- build_disagg -----------------------------------------------------------


def _disagg_cfg(prefill_device="h100_sxm", decode_device="h100_sxm"):
    return DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(
            device_name=prefill_device,
            tp_size=4,
            ep_size=1,
            replicas=2,
            max_running_per_replica=128,
            database_path="/aic/prefill.db",
        ),
        decode=RolePredictorConfig(
            device_name=decode_device,
            tp_size=2,
            dp_size=2,
            replicas=4,
            max_running_per_replica=64,
            database_path="/aic/decode.db",
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=100.0, latency_us=10.0),
    )


def test_build_disagg_disabled_raises():
    model = _mha_model()
    with pytest.raises(ValueError):
        build_disagg(
            model=model,
            base_sched_config=_base_sched(model),
            disagg_config=DisaggConfig(),  # disabled
            predictor_factory=_StubPredictor,
            hw_factory=_hw_factory,
        )


def test_build_disagg_returns_two_predictors_with_role_configs():
    model = _mha_model()
    out = build_disagg(
        model=model,
        base_sched_config=_base_sched(model),
        disagg_config=_disagg_cfg(),
        predictor_factory=_StubPredictor,
        hw_factory=_hw_factory,
    )
    assert isinstance(out, DisaggPredictors)
    # Two predictors, distinct objects
    assert out.prefill is not out.decode
    # Prefill role
    assert out.prefill.hw.name == "h100_sxm"
    assert out.prefill.config.tp_size == 4
    assert out.prefill.config.ep_size == 1
    assert out.prefill.config.max_running_requests == 128
    assert out.prefill.kwargs.get("database_path") == "/aic/prefill.db"
    # Decode role
    assert out.decode.config.tp_size == 2
    assert out.decode.config.dp_size == 2
    assert out.decode.config.max_running_requests == 64
    assert out.decode.kwargs.get("database_path") == "/aic/decode.db"


def test_build_disagg_supports_heterogeneous_devices():
    model = _mha_model()
    out = build_disagg(
        model=model,
        base_sched_config=_base_sched(model),
        disagg_config=_disagg_cfg(prefill_device="h100_sxm", decode_device="h20"),
        predictor_factory=_StubPredictor,
        hw_factory=_hw_factory,
    )
    assert out.prefill.hw.name == "h100_sxm"
    assert out.decode.hw.name == "h20"


def test_build_disagg_derives_kv_bytes_per_token_from_prefill_tp():
    model = _mha_model(num_kv_heads=8, head_dim=128, num_layers=32)
    out = build_disagg(
        model=model,
        base_sched_config=_base_sched(model),
        disagg_config=_disagg_cfg(),
        predictor_factory=_StubPredictor,
        hw_factory=_hw_factory,
    )
    # prefill.tp_size = 4 → num_kv_heads = 2; FP16 → 2*128*32*2*2 = 32768
    assert out.kv_bytes_per_token == 32768


def test_build_disagg_transfer_model_matches_kv_transfer_cfg():
    model = _mha_model()
    out = build_disagg(
        model=model,
        base_sched_config=_base_sched(model),
        disagg_config=_disagg_cfg(),
        predictor_factory=_StubPredictor,
        hw_factory=_hw_factory,
    )
    assert isinstance(out.transfer_model, BandwidthTransferModel)
    # estimate at seq_len=0 must equal latency_us in seconds
    assert out.transfer_model.estimate(0, out.kv_model_config) == pytest.approx(10e-6)
    # estimate at seq_len=1 with bw=100 GB/s and 32768 bytes/token:
    # 10e-6 + 32768 / 100e9 = 10e-6 + 3.2768e-7
    expected = 10e-6 + 32768 / 100e9
    assert out.transfer_model.estimate(1, out.kv_model_config) == pytest.approx(expected)


def test_build_disagg_uses_role_kv_cache_dtype_when_set():
    model = _mha_model(num_kv_heads=8, head_dim=128, num_layers=32)
    cfg = _disagg_cfg()
    cfg.prefill.kv_cache_data_type = "FP8"  # 1 byte
    out = build_disagg(
        model=model,
        base_sched_config=_base_sched(model),
        disagg_config=cfg,
        predictor_factory=_StubPredictor,
        hw_factory=_hw_factory,
    )
    # tp=4 → 2 kv_heads; FP8 → 2*128*32*2*1 = 16384
    assert out.kv_bytes_per_token == 16384
