import ast
import dataclasses
import inspect

import pytest

from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)


def test_role_predictor_config_defaults_are_sane():
    cfg = RolePredictorConfig(device_name="h100_sxm")
    assert cfg.device_name == "h100_sxm"
    assert cfg.tp_size == 1
    assert cfg.ep_size == 1
    assert cfg.dp_size == 1
    assert cfg.pp_size == 1
    assert cfg.data_type is None
    assert cfg.kv_cache_data_type is None
    assert cfg.database_path is None
    assert cfg.backend_version is None
    assert cfg.replicas == 1
    assert cfg.max_running_per_replica > 0
    assert cfg.prefill_scale_factor == 1.0
    assert cfg.decode_scale_factor == 1.0


def test_role_predictor_config_validates_positive_sizes():
    with pytest.raises(ValueError):
        RolePredictorConfig(device_name="h100_sxm", tp_size=0)
    with pytest.raises(ValueError):
        RolePredictorConfig(device_name="h100_sxm", replicas=0)
    with pytest.raises(ValueError):
        RolePredictorConfig(device_name="h100_sxm", max_running_per_replica=0)


def test_bandwidth_transfer_config_validates():
    cfg = BandwidthTransferConfig(bw_gbps=100.0, latency_us=10.0)
    assert cfg.bw_gbps == 100.0
    assert cfg.latency_us == 10.0
    with pytest.raises(ValueError):
        BandwidthTransferConfig(bw_gbps=0.0, latency_us=10.0)
    with pytest.raises(ValueError):
        BandwidthTransferConfig(bw_gbps=100.0, latency_us=-1.0)


def test_disagg_config_disabled_by_default():
    cfg = DisaggConfig()
    assert cfg.enabled is False
    assert cfg.backend == "single_process"
    assert cfg.prefill is None
    assert cfg.decode is None
    assert cfg.kv_transfer is None


def test_disagg_config_enabled_requires_roles_and_transfer():
    prefill = RolePredictorConfig(device_name="h100_sxm")
    decode = RolePredictorConfig(device_name="h100_sxm")
    transfer = BandwidthTransferConfig(bw_gbps=100.0, latency_us=10.0)

    # Missing prefill role.
    with pytest.raises(ValueError):
        DisaggConfig(enabled=True, decode=decode, kv_transfer=transfer)
    # Missing decode role.
    with pytest.raises(ValueError):
        DisaggConfig(enabled=True, prefill=prefill, kv_transfer=transfer)
    # Missing kv_transfer.
    with pytest.raises(ValueError):
        DisaggConfig(enabled=True, prefill=prefill, decode=decode)
    # Unknown backend.
    with pytest.raises(ValueError):
        DisaggConfig(
            enabled=True,
            backend="three_process",
            prefill=prefill,
            decode=decode,
            kv_transfer=transfer,
        )

    # Valid construction.
    cfg = DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=prefill,
        decode=decode,
        kv_transfer=transfer,
    )
    assert cfg.enabled is True
    assert cfg.prefill is prefill
    assert cfg.decode is decode


def test_disagg_config_supports_heterogeneous_per_role_device():
    prefill = RolePredictorConfig(
        device_name="h100_sxm", tp_size=4, database_path="/aic/h100.db"
    )
    decode = RolePredictorConfig(
        device_name="h20", tp_size=2, database_path="/aic/h20.db"
    )
    transfer = BandwidthTransferConfig(bw_gbps=100.0, latency_us=10.0)
    cfg = DisaggConfig(
        enabled=True, prefill=prefill, decode=decode, kv_transfer=transfer
    )
    assert cfg.prefill.device_name != cfg.decode.device_name
    assert cfg.prefill.database_path != cfg.decode.database_path


def test_disagg_config_round_trips_through_asdict():
    cfg = DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(device_name="h100_sxm", tp_size=4),
        decode=RolePredictorConfig(device_name="h100_sxm", tp_size=2, dp_size=2),
        kv_transfer=BandwidthTransferConfig(bw_gbps=100.0, latency_us=10.0),
    )
    d = dataclasses.asdict(cfg)
    assert d["enabled"] is True
    assert d["prefill"]["tp_size"] == 4
    assert d["decode"]["dp_size"] == 2
    assert d["kv_transfer"]["bw_gbps"] == 100.0


def test_pd_config_has_no_sglang_dependency():
    module = inspect.getmodule(DisaggConfig)
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
