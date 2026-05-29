"""Phase 1d: tests for the AIC→Hisim bridge emitting a `disagg` block.

When the AIC CSV is disagg-shaped ((p)tp / (d)tp columns), the bridge can emit
a top-level `disagg` block in the generated Hisim JSON. The shape must round-
trip through SimulationArgs.from_json's _disagg_from_dict helper.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hisim.simulation.sim_args import _disagg_from_dict


BRIDGE = Path(__file__).resolve().parents[3] / "tools" / "aic_to_hisim_bridge.py"


_DISAGG_HEADER = [
    "(p)tp", "(p)pp", "(p)dp", "(p)moe_ep", "(p)gemm", "(p)kvcache",
    "(p)version", "(p)system", "(p)workers",
    "(d)tp", "(d)pp", "(d)dp", "(d)moe_ep", "(d)gemm", "(d)kvcache",
    "(d)version", "(d)system", "(d)workers",
]
_DISAGG_ROW = [
    "4", "1", "1", "1", "fp16", "fp16",
    "0.5.6.post2", "h100_sxm", "2",
    "2", "1", "2", "1", "fp16", "fp16",
    "0.5.6.post2", "h20", "4",
]
_AGG_HEADER = ["tp", "pp", "dp", "moe_ep", "gemm", "kvcache", "version", "system"]
_AGG_ROW = ["8", "1", "1", "1", "fp16", "fp16", "0.5.6.post2", "h100_sxm"]


def _write_csv(path: Path, header, row):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


def _run_bridge(args, *, expect_ok=True):
    result = subprocess.run(
        [sys.executable, str(BRIDGE), *args],
        capture_output=True, text=True,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"bridge failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    return result


def test_bridge_emits_disagg_block_when_csv_is_disagg(tmp_path):
    csv_path = tmp_path / "pareto.csv"
    out_path = tmp_path / "hisim.json"
    _write_csv(csv_path, _DISAGG_HEADER, _DISAGG_ROW)

    _run_bridge([
        "--aic-csv", str(csv_path),
        "--row-index", "0",
        "--output", str(out_path),
        "--database-path", "/aic/db",
        "--emit-disagg",
        "--kv-transfer-bw-gbps", "100",
        "--kv-transfer-latency-us", "10",
    ])

    cfg = json.loads(out_path.read_text())
    assert "disagg" in cfg
    d = cfg["disagg"]
    assert d["enabled"] is True
    assert d["backend"] == "single_process"
    assert d["prefill"]["device_name"] == "h100_sxm"
    assert d["prefill"]["tp_size"] == 4
    assert d["prefill"]["replicas"] == 2
    assert d["decode"]["device_name"] == "h20"
    assert d["decode"]["tp_size"] == 2
    assert d["decode"]["dp_size"] == 2
    assert d["decode"]["replicas"] == 4
    assert d["kv_transfer"]["bw_gbps"] == 100.0
    assert d["kv_transfer"]["latency_us"] == 10.0


def test_bridge_disagg_block_round_trips_through_sim_args(tmp_path):
    csv_path = tmp_path / "pareto.csv"
    out_path = tmp_path / "hisim.json"
    _write_csv(csv_path, _DISAGG_HEADER, _DISAGG_ROW)

    _run_bridge([
        "--aic-csv", str(csv_path),
        "--row-index", "0",
        "--output", str(out_path),
        "--database-path", "/aic/db",
        "--emit-disagg",
        "--kv-transfer-bw-gbps", "100",
        "--kv-transfer-latency-us", "10",
    ])
    cfg = json.loads(out_path.read_text())

    disagg = _disagg_from_dict(cfg["disagg"])
    assert disagg.enabled is True
    assert disagg.prefill.device_name == "h100_sxm"
    assert disagg.prefill.tp_size == 4
    assert disagg.prefill.replicas == 2
    assert disagg.decode.device_name == "h20"
    assert disagg.decode.tp_size == 2
    assert disagg.decode.dp_size == 2
    assert disagg.decode.replicas == 4
    assert disagg.kv_transfer.bw_gbps == 100.0


def test_bridge_emit_disagg_requires_disagg_csv(tmp_path):
    csv_path = tmp_path / "agg.csv"
    out_path = tmp_path / "hisim.json"
    _write_csv(csv_path, _AGG_HEADER, _AGG_ROW)

    result = _run_bridge([
        "--aic-csv", str(csv_path),
        "--row-index", "0",
        "--output", str(out_path),
        "--database-path", "/aic/db",
        "--emit-disagg",
        "--kv-transfer-bw-gbps", "100",
        "--kv-transfer-latency-us", "10",
    ], expect_ok=False)
    assert result.returncode != 0
    assert "disagg" in (result.stderr + result.stdout).lower()


def test_bridge_without_emit_disagg_keeps_single_role_behavior(tmp_path):
    """Default (no --emit-disagg) on a disagg CSV still produces single-role
    scheduler block — preserves backward compatibility."""
    csv_path = tmp_path / "pareto.csv"
    out_path = tmp_path / "hisim.json"
    _write_csv(csv_path, _DISAGG_HEADER, _DISAGG_ROW)

    _run_bridge([
        "--aic-csv", str(csv_path),
        "--row-index", "0",
        "--output", str(out_path),
        "--database-path", "/aic/db",
        "--disagg-role", "decode",
    ])
    cfg = json.loads(out_path.read_text())
    assert "disagg" not in cfg
    assert cfg["scheduler"]["tp_size"] == 2  # decode role


def test_bridge_emit_disagg_requires_kv_transfer_flags(tmp_path):
    csv_path = tmp_path / "pareto.csv"
    out_path = tmp_path / "hisim.json"
    _write_csv(csv_path, _DISAGG_HEADER, _DISAGG_ROW)

    result = _run_bridge([
        "--aic-csv", str(csv_path),
        "--row-index", "0",
        "--output", str(out_path),
        "--database-path", "/aic/db",
        "--emit-disagg",
    ], expect_ok=False)
    assert result.returncode != 0
    assert "kv" in (result.stderr + result.stdout).lower()
