import argparse
import json

import pytest

from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.sim_args import SimulationArgs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    SimulationArgs.add_cli_args(parser)
    return parser


def test_disagg_defaults_disabled_when_no_flags_or_json():
    parser = _build_parser()
    ns = parser.parse_args([])
    args = SimulationArgs.from_cli_args(ns)
    assert args.disagg.enabled is False
    assert args.disagg.prefill is None
    assert args.disagg.decode is None
    assert args.disagg.kv_transfer is None


def test_disagg_cli_full_round_trip():
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "--sim-disagg-enable",
            "--sim-disagg-backend", "single_process",
            "--sim-disagg-prefill-device-name", "h100_sxm",
            "--sim-disagg-prefill-tp", "4",
            "--sim-disagg-prefill-ep", "1",
            "--sim-disagg-prefill-dp", "1",
            "--sim-disagg-prefill-replicas", "2",
            "--sim-disagg-prefill-max-running", "128",
            "--sim-disagg-prefill-database-path", "/aic/h100.db",
            "--sim-disagg-decode-device-name", "h20",
            "--sim-disagg-decode-tp", "2",
            "--sim-disagg-decode-ep", "1",
            "--sim-disagg-decode-dp", "2",
            "--sim-disagg-decode-replicas", "1",
            "--sim-disagg-decode-max-running", "256",
            "--sim-disagg-decode-database-path", "/aic/h20.db",
            "--sim-kv-transfer-bw-gbps", "100.0",
            "--sim-kv-transfer-latency-us", "10.0",
        ]
    )
    args = SimulationArgs.from_cli_args(ns)

    assert args.disagg.enabled is True
    assert args.disagg.backend == "single_process"

    assert isinstance(args.disagg.prefill, RolePredictorConfig)
    assert args.disagg.prefill.device_name == "h100_sxm"
    assert args.disagg.prefill.tp_size == 4
    assert args.disagg.prefill.replicas == 2
    assert args.disagg.prefill.max_running_per_replica == 128
    assert args.disagg.prefill.database_path == "/aic/h100.db"

    assert isinstance(args.disagg.decode, RolePredictorConfig)
    assert args.disagg.decode.device_name == "h20"
    assert args.disagg.decode.tp_size == 2
    assert args.disagg.decode.dp_size == 2
    assert args.disagg.decode.database_path == "/aic/h20.db"

    assert isinstance(args.disagg.kv_transfer, BandwidthTransferConfig)
    assert args.disagg.kv_transfer.bw_gbps == 100.0
    assert args.disagg.kv_transfer.latency_us == 10.0


def test_disagg_supports_heterogeneous_devices_via_cli():
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "--sim-disagg-enable",
            "--sim-disagg-prefill-device-name", "h100_sxm",
            "--sim-disagg-decode-device-name", "h20",
            "--sim-kv-transfer-bw-gbps", "50.0",
            "--sim-kv-transfer-latency-us", "5.0",
        ]
    )
    args = SimulationArgs.from_cli_args(ns)
    assert args.disagg.enabled is True
    assert args.disagg.prefill.device_name == "h100_sxm"
    assert args.disagg.decode.device_name == "h20"


def test_disagg_enable_without_required_flags_raises():
    parser = _build_parser()
    ns = parser.parse_args(["--sim-disagg-enable"])
    with pytest.raises(ValueError):
        SimulationArgs.from_cli_args(ns)


def test_disagg_json_round_trip(tmp_path):
    cfg = {
        "predictor": {"name": "aiconfigurator"},
        "scheduler": {"tp_size": 1},
        "disagg": {
            "enabled": True,
            "backend": "single_process",
            "prefill": {
                "device_name": "h100_sxm",
                "tp_size": 4,
                "ep_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "replicas": 2,
                "max_running_per_replica": 128,
                "database_path": "/aic/h100.db",
            },
            "decode": {
                "device_name": "h20",
                "tp_size": 2,
                "dp_size": 2,
                "database_path": "/aic/h20.db",
            },
            "kv_transfer": {"bw_gbps": 100.0, "latency_us": 10.0},
        },
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    args = SimulationArgs.from_json(str(p))
    assert args.disagg.enabled is True
    assert args.disagg.backend == "single_process"
    assert args.disagg.prefill.device_name == "h100_sxm"
    assert args.disagg.prefill.tp_size == 4
    assert args.disagg.prefill.replicas == 2
    assert args.disagg.decode.device_name == "h20"
    assert args.disagg.decode.tp_size == 2
    assert args.disagg.decode.dp_size == 2
    assert args.disagg.kv_transfer.bw_gbps == 100.0

    # to_dict must include disagg block.
    d = args.to_dict()
    assert "disagg" in d
    assert d["disagg"]["enabled"] is True
    assert d["disagg"]["prefill"]["device_name"] == "h100_sxm"
    assert d["disagg"]["kv_transfer"]["latency_us"] == 10.0


def test_disagg_json_disabled_is_default_when_block_missing(tmp_path):
    cfg = {"predictor": {"name": "aiconfigurator"}, "scheduler": {"tp_size": 1}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    args = SimulationArgs.from_json(str(p))
    assert isinstance(args.disagg, DisaggConfig)
    assert args.disagg.enabled is False
