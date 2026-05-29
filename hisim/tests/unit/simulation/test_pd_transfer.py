import inspect
import math

import pytest

from hisim.simulation.pd_transfer import (
    BandwidthTransferModel,
    KVModelConfig,
    TransferModel,
)


def test_transfer_model_is_abstract():
    assert inspect.isabstract(TransferModel)
    with pytest.raises(TypeError):
        TransferModel()  # type: ignore[abstract]


def test_bandwidth_transfer_model_deterministic_duration():
    model = BandwidthTransferModel(bw_gbps=100.0, latency_us=10.0)
    cfg = KVModelConfig(kv_bytes_per_token=1024)

    # bytes = 2048 * 1024 = 2_097_152
    # bw    = 100e9 bytes/s
    # xfer  = 2_097_152 / 1e11 = 2.097152e-5 s
    # total = 10e-6 + 2.097152e-5 = 3.097152e-5 s
    expected = 10e-6 + (2048 * 1024) / (100.0 * 1e9)
    assert math.isclose(model.estimate(2048, cfg), expected, rel_tol=1e-12)


def test_bandwidth_transfer_model_zero_latency():
    model = BandwidthTransferModel(bw_gbps=50.0, latency_us=0.0)
    cfg = KVModelConfig(kv_bytes_per_token=512)
    expected = (1024 * 512) / (50.0 * 1e9)
    assert math.isclose(model.estimate(1024, cfg), expected, rel_tol=1e-12)


def test_bandwidth_transfer_model_very_high_bw_approaches_latency():
    model = BandwidthTransferModel(bw_gbps=1e9, latency_us=5.0)
    cfg = KVModelConfig(kv_bytes_per_token=1024)
    duration = model.estimate(4096, cfg)
    assert math.isclose(duration, 5e-6, rel_tol=1e-3)


def test_bandwidth_transfer_model_zero_seq_len_is_just_latency():
    model = BandwidthTransferModel(bw_gbps=100.0, latency_us=7.5)
    cfg = KVModelConfig(kv_bytes_per_token=1024)
    assert math.isclose(model.estimate(0, cfg), 7.5e-6, rel_tol=1e-12)


def test_bandwidth_transfer_model_rejects_invalid_params():
    with pytest.raises(ValueError):
        BandwidthTransferModel(bw_gbps=0.0, latency_us=1.0)
    with pytest.raises(ValueError):
        BandwidthTransferModel(bw_gbps=-1.0, latency_us=1.0)
    with pytest.raises(ValueError):
        BandwidthTransferModel(bw_gbps=10.0, latency_us=-1.0)


def test_bandwidth_transfer_model_rejects_negative_seq_len():
    model = BandwidthTransferModel(bw_gbps=100.0, latency_us=1.0)
    cfg = KVModelConfig(kv_bytes_per_token=1024)
    with pytest.raises(ValueError):
        model.estimate(-1, cfg)


def test_pd_transfer_has_no_sglang_dependency():
    import ast

    module = inspect.getmodule(BandwidthTransferModel)
    tree = ast.parse(inspect.getsource(module))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name.startswith("sglang") for name in imported)
    assert not any(name.startswith("hisim.simulation.sglang") for name in imported)
