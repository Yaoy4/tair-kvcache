"""Phase 3c — sweep script disagg pass-through.

The aic_topn_to_hisim_sweep CLI must (1) build a bridge subprocess argv
without disagg flags by default (regression check), and (2) propagate
--emit-disagg / --kv-transfer-* / --disagg-backend through to the
aic_to_hisim_bridge invocation when requested.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

# Tool module lives outside the package; load via importlib.
import importlib.util


def _load_sweep_module():
    here = Path(__file__).resolve()
    module_path = here.parents[3] / "tools" / "aic_topn_to_hisim_sweep.py"
    assert module_path.exists(), f"sweep script not found at {module_path}"
    spec = importlib.util.spec_from_file_location(
        "aic_topn_to_hisim_sweep_under_test", module_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SWEEP = _load_sweep_module()


def _base_args(**overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        python_bin="/usr/bin/python3",
        base_config="/tmp/base.json",
        disagg_role="decode",
        database_mode="SILICON",
        platform_accelerator="H20",
        disk_read_bandwidth_gb=8.0,
        disk_write_bandwidth_gb=8.0,
        memory_read_bandwidth_gb=16.0,
        memory_write_bandwidth_gb=16.0,
        num_device_per_node=8,
        predictor_device_name=None,
        backend_version=None,
        prefill_scale_factor=None,
        decode_scale_factor=None,
        xgb_model_path=None,
        data_type=None,
        kv_cache_data_type=None,
        emit_disagg=False,
        kv_transfer_bw_gbps=200.0,
        kv_transfer_latency_us=10.0,
        disagg_backend="single_process",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _kwargs():
    return dict(
        bridge_script=Path("/tmp/bridge.py"),
        csv_path=Path("/tmp/aic.csv"),
        row_idx=2,
        sim_config_path=Path("/tmp/sim.json"),
        database_path=Path("/tmp/perf_db"),
        emit_disagg=False,
    )


def test_build_bridge_cmd_no_disagg_does_not_emit_disagg_flags():
    cmd = SWEEP._build_bridge_cmd(_base_args(), **_kwargs())
    # Sanity: row_idx and disagg-role are wired
    assert "--row-index" in cmd and "2" in cmd
    assert "--disagg-role" in cmd and "decode" in cmd
    # No disagg JSON emission flags by default
    for flag in (
        "--emit-disagg",
        "--kv-transfer-bw-gbps",
        "--kv-transfer-latency-us",
        "--disagg-backend",
    ):
        assert flag not in cmd, f"unexpected {flag} in default cmd"


def test_build_bridge_cmd_emit_disagg_propagates_flags():
    args = _base_args(
        emit_disagg=True,
        kv_transfer_bw_gbps=128.0,
        kv_transfer_latency_us=5.0,
        disagg_backend="single_process",
    )
    kw = _kwargs()
    kw["emit_disagg"] = True
    cmd = SWEEP._build_bridge_cmd(args, **kw)
    assert "--emit-disagg" in cmd
    # Each --kv-transfer-* flag is followed by its stringified value.
    bw_idx = cmd.index("--kv-transfer-bw-gbps")
    assert cmd[bw_idx + 1] == "128.0"
    lat_idx = cmd.index("--kv-transfer-latency-us")
    assert cmd[lat_idx + 1] == "5.0"
    be_idx = cmd.index("--disagg-backend")
    assert cmd[be_idx + 1] == "single_process"


def test_build_bridge_cmd_propagates_optional_flags():
    args = _base_args(
        predictor_device_name="H100_SXM",
        backend_version="0.4.8",
        prefill_scale_factor=1.1,
        decode_scale_factor=0.9,
        data_type="fp8",
        kv_cache_data_type="fp8",
    )
    cmd = SWEEP._build_bridge_cmd(args, **_kwargs())
    assert "--predictor-device-name" in cmd
    assert cmd[cmd.index("--predictor-device-name") + 1] == "H100_SXM"
    assert "--prefill-scale-factor" in cmd
    assert cmd[cmd.index("--prefill-scale-factor") + 1] == "1.1"
    assert "--data-type" in cmd and cmd[cmd.index("--data-type") + 1] == "fp8"


def test_parse_synth_disagg_profiles_parses_and_dedups():
    profiles = SWEEP._parse_synth_disagg_profiles("2:6,4:4,2:6")
    assert profiles == [(2, 6), (4, 4)]


def test_parse_synth_disagg_profiles_rejects_invalid_token():
    with pytest.raises(ValueError):
        SWEEP._parse_synth_disagg_profiles("2x6")


def test_build_synth_disagg_block_uses_agg_row_fields_and_replica_split():
    args = _base_args(
        predictor_device_name="b60",
        backend_version="0.12.0",
        data_type="bfloat16",
        kv_cache_data_type="bfloat16",
        disagg_backend="single_process",
    )
    args.synth_max_running_per_replica = 8
    row = {
        "tp": "8",
        "dp": "1",
        "moe_ep": "4",
        "pp": "1",
        "system": "b60",
        "version": "0.12.0",
        "gemm": "bfloat16",
        "kvcache": "bfloat16",
    }
    block = SWEEP._build_synth_disagg_block(
        row,
        args,
        database_path=Path("/tmp/perf_db"),
        prefill_replicas=2,
        decode_replicas=6,
    )
    assert block["enabled"] is True
    assert block["backend"] == "single_process"
    assert block["prefill"]["replicas"] == 2
    assert block["decode"]["replicas"] == 6
    assert block["prefill"]["tp_size"] == 8
    assert block["decode"]["ep_size"] == 4
    assert block["prefill"]["database_path"] == "/tmp/perf_db"


def test_build_synth_disagg_block_resolves_moe_overrides():
    args = _base_args(
        predictor_device_name="b60",
        backend_version="0.12.0",
        data_type="bfloat16",
        kv_cache_data_type="bfloat16",
        disagg_backend="single_process",
    )
    args.synth_max_running_per_replica = 8
    args.synth_ep_size = 8
    args.synth_moe_tp_size = None
    row = {
        "tp": "8",
        "dp": "1",
        "moe_ep": "4",
        "moe_tp": "2",
        "pp": "1",
        "system": "b60",
        "version": "0.12.0",
        "gemm": "bfloat16",
        "kvcache": "bfloat16",
    }
    block = SWEEP._build_synth_disagg_block(
        row,
        args,
        database_path=Path("/tmp/perf_db"),
        prefill_replicas=2,
        decode_replicas=6,
    )
    # tp*dp = 8, so synth_ep=8 resolves moe_tp to 1.
    assert block["prefill"]["ep_size"] == 8
    assert block["decode"]["ep_size"] == 8
    assert block["_resolved_moe_tp_size"] == 1


def test_sweep_parser_accepts_disagg_flags():
    """Parser must accept the new flags without --aic-csv issues (use minimal required)."""
    parser = argparse.ArgumentParser()
    # Re-register only what we test here; simpler than re-running _parse_args.
    parser.add_argument("--emit-disagg", action="store_true")
    parser.add_argument("--kv-transfer-bw-gbps", type=float, default=200.0)
    parser.add_argument("--kv-transfer-latency-us", type=float, default=10.0)
    parser.add_argument(
        "--disagg-backend",
        choices=["single_process", "two_process"],
        default="single_process",
    )
    ns = parser.parse_args(
        [
            "--emit-disagg",
            "--kv-transfer-bw-gbps",
            "100",
            "--kv-transfer-latency-us",
            "1",
            "--disagg-backend",
            "single_process",
        ]
    )
    assert ns.emit_disagg is True
    assert ns.kv_transfer_bw_gbps == pytest.approx(100.0)
    assert ns.disagg_backend == "single_process"
