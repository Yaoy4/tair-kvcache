"""Unit tests for sweep_report (V3)."""

from __future__ import annotations

import pandas as pd

from hisim.visualization import sweep_report


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "topology": "pd-p1tp2-d1tp2",
            "total_gpus": 4,
            "mean_ttft_ms": 50.0,
            "output_throughput": 120.0,
            "p99_e2e_latency_ms": 300.0,
            "is_pareto_optimal": True,
            "disagg_enabled": True,
            "prefill_replicas": 1,
            "decode_replicas": 1,
            "mean_prefill_queue_ms": 5.0,
            "mean_kv_transfer_ms": 2.0,
            "mean_decode_queue_ms": 7.0,
        },
        {
            "topology": "agg-tp4-dp1",
            "total_gpus": 4,
            "mean_ttft_ms": 60.0,
            "output_throughput": 130.0,
            "p99_e2e_latency_ms": 350.0,
            "is_pareto_optimal": True,
            "disagg_enabled": False,
            "prefill_replicas": None,
            "decode_replicas": None,
            "mean_prefill_queue_ms": 4.0,
            "mean_kv_transfer_ms": 0.0,
            "mean_decode_queue_ms": 6.0,
        },
    ])


def test_render_report_writes_html_with_expected_content(tmp_path):
    out = sweep_report.render_report(
        _fixture_df(),
        tmp_path / "r.html",
        title="My Report",
        source="src.csv",
    )
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "My Report" in html
    assert "src.csv" in html
    assert "Pareto" in html
    assert "2 configurations" in html
    assert "2 on Pareto frontier" in html


def test_render_report_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "out.html"
    out = sweep_report.render_report(_fixture_df(), nested)
    assert out.exists()
    assert out.parent.is_dir()


def test_render_report_empty_dataframe(tmp_path):
    out = sweep_report.render_report(pd.DataFrame(), tmp_path / "empty.html")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "0 configurations" in html
