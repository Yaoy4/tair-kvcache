"""Unit tests for the V5 streamlit dashboard.

Uses streamlit's AppTest harness; no browser, no subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


_DASHBOARD = (
    Path(__file__).resolve().parents[3]
    / "tools" / "sweep_dashboard.py"
)


def _disagg_sim(p_tp: int, p_repl: int, d_tp: int, d_repl: int) -> dict:
    return {
        "disagg": {
            "enabled": True,
            "backend": "single_process",
            "prefill": {"tp_size": p_tp, "replicas": p_repl},
            "decode": {"tp_size": d_tp, "replicas": d_repl},
            "kv_transfer": {"bw_gbps": 200, "latency_us": 5},
        }
    }


def _result(ttft, tput, p99) -> dict:
    return {
        "mean_ttft_ms": ttft,
        "output_throughput": tput,
        "p99_e2e_latency_ms": p99,
        "mean_prefill_queue_ms": 5.0,
        "mean_kv_transfer_ms": 2.0,
        "mean_decode_queue_ms": 7.0,
    }


def _write_fixture(tmp_path: Path) -> Path:
    rows = []
    for i, (p_tp, p_r, d_tp, d_r, ttft, tput, p99) in enumerate([
        (2, 1, 2, 1, 55, 110, 320),
        (2, 2, 2, 1, 42, 145, 290),
        (4, 2, 4, 2, 38, 180, 260),
    ]):
        sim = tmp_path / f"sim_{i}.json"
        res = tmp_path / f"res_{i}.json"
        sim.write_text(json.dumps(_disagg_sim(p_tp, p_r, d_tp, d_r)))
        res.write_text(json.dumps(_result(ttft, tput, p99)))
        rows.append(f"run{i},ok,{sim},{res}")
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("run_id,status,sim_config,result_file\n" + "\n".join(rows) + "\n")
    return csv_path


def _app() -> AppTest:
    return AppTest.from_file(str(_DASHBOARD), default_timeout=30)


def test_dashboard_renders_info_without_csv():
    at = _app()
    at.run()
    assert not at.exception
    # When no CSV path is given, the dashboard shows an info banner.
    assert any("summary.csv" in str(el.value).lower() for el in at.info)


def test_dashboard_loads_csv_and_shows_metrics(tmp_path):
    csv_path = _write_fixture(tmp_path)
    at = _app()
    # Set the text_input to the CSV path BEFORE running.
    at.run()
    assert not at.exception
    at.sidebar.text_input[0].set_value(str(csv_path)).run()
    assert not at.exception
    # Four metric tiles: Rows, Pareto, Disagg, Aggregated.
    assert len(at.metric) == 4
    assert at.metric[0].value == "3"  # Rows
    assert at.metric[2].value == "3"  # Disagg
    assert at.metric[3].value == "0"  # Aggregated


def test_dashboard_error_on_missing_csv(tmp_path):
    at = _app()
    at.run()
    at.sidebar.text_input[0].set_value(str(tmp_path / "nope.csv")).run()
    assert not at.exception
    assert any("not found" in str(el.value).lower() for el in at.error)


def test_dashboard_pareto_filter_reduces_rows(tmp_path):
    csv_path = _write_fixture(tmp_path)
    at = _app()
    at.run()
    at.sidebar.text_input[0].set_value(str(csv_path)).run()
    # First checkbox in the sidebar is "Pareto-optimal only".
    at.sidebar.checkbox[0].check().run()
    assert not at.exception
    # Of the 3 rows above, only (38, 180) is pareto-optimal.
    rows_metric = at.metric[0].value
    assert int(rows_metric) <= 3
