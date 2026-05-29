"""Integration test for hisim/tools/sweep_plot.py CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_HISIM_SRC = _REPO_ROOT / "hisim" / "src"
_AIC_SRC = _REPO_ROOT.parent / "aiconfigurator" / "src"
_CLI = _REPO_ROOT / "hisim" / "tools" / "sweep_plot.py"


def _write_summary(tmp_path: Path) -> Path:
    # disagg row with sim_config + result_file
    result_json = tmp_path / "row0_result.json"
    result_json.write_text(json.dumps({
        "mean_ttft_ms": 50.0,
        "output_throughput": 120.0,
        "p99_e2e_latency_ms": 300.0,
        "mean_prefill_queue_ms": 5.0,
        "mean_kv_transfer_ms": 2.0,
        "mean_decode_queue_ms": 7.0,
    }))
    sim_json = tmp_path / "row0_sim.json"
    sim_json.write_text(json.dumps({
        "disagg": {
            "enabled": True,
            "backend": "single_process",
            "prefill": {"tp_size": 2, "replicas": 1},
            "decode": {"tp_size": 2, "replicas": 1},
            "kv_transfer": {"bw_gbps": 200, "latency_us": 5},
        }
    }))

    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "run_id,result_file,sim_config\n"
        f"run0,{result_json},{sim_json}\n"
    )
    return csv_path


@pytest.mark.integration
def test_sweep_plot_cli_writes_html(tmp_path):
    csv_path = _write_summary(tmp_path)
    out_path = tmp_path / "report.html"

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_HISIM_SRC), str(_AIC_SRC), existing_pp]))

    proc = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--summary-csv", str(csv_path),
            "--output", str(out_path),
            "--title", "Test Report",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "Test Report" in html
    # plotly was loaded from CDN -> embedded <script> tag from plotly
    assert "plotly" in html.lower()
    # The pareto / heatmap sections should appear
    assert "Pareto" in html
    assert "PD replica heatmap" in html


@pytest.mark.integration
def test_sweep_plot_cli_missing_csv_returns_2(tmp_path):
    out_path = tmp_path / "report.html"
    proc = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--summary-csv", str(tmp_path / "nope.csv"),
            "--output", str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
    assert not out_path.exists()
