"""Phase 2b.3 — tests for ConfigManager.get_disagg_config().

Round-trips a DisaggConfig through the HISIM_CONFIG_PATH JSON file that the
hook reads at init time.
"""
import json
import os
from pathlib import Path

import pytest

from hisim.simulation.manager import ConfigManager
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)


def _write_cfg(tmp_path: Path, payload: dict) -> str:
    p = tmp_path / "hisim.json"
    p.write_text(json.dumps(payload))
    os.environ["HISIM_CONFIG_PATH"] = str(p)
    return str(p)


def test_get_disagg_config_returns_disabled_when_missing(tmp_path, monkeypatch):
    _write_cfg(tmp_path, {})
    cfg = ConfigManager.get_disagg_config()
    assert isinstance(cfg, DisaggConfig)
    assert cfg.enabled is False


def test_get_disagg_config_round_trip(tmp_path, monkeypatch):
    payload = {
        "disagg": {
            "enabled": True,
            "backend": "single_process",
            "prefill": {
                "device_name": "h100_sxm",
                "tp_size": 4,
                "replicas": 2,
                "max_running_per_replica": 32,
                "data_type": "FP16",
                "kv_cache_data_type": "FP16",
                "database_path": "/aic/db",
                "backend_version": "0.5.6.post2",
            },
            "decode": {
                "device_name": "h20",
                "tp_size": 2,
                "dp_size": 2,
                "replicas": 4,
                "max_running_per_replica": 256,
                "data_type": "FP16",
                "kv_cache_data_type": "FP16",
                "database_path": "/aic/db",
                "backend_version": "0.5.6.post2",
            },
            "kv_transfer": {"bw_gbps": 50.0, "latency_us": 20.0},
        }
    }
    _write_cfg(tmp_path, payload)
    cfg = ConfigManager.get_disagg_config()
    assert cfg.enabled is True
    assert cfg.backend == "single_process"
    assert isinstance(cfg.prefill, RolePredictorConfig)
    assert cfg.prefill.device_name == "h100_sxm"
    assert cfg.prefill.tp_size == 4
    assert cfg.prefill.replicas == 2
    assert cfg.decode.device_name == "h20"
    assert cfg.decode.tp_size == 2
    assert cfg.decode.dp_size == 2
    assert cfg.decode.replicas == 4
    assert isinstance(cfg.kv_transfer, BandwidthTransferConfig)
    assert cfg.kv_transfer.bw_gbps == 50.0
    assert cfg.kv_transfer.latency_us == 20.0
