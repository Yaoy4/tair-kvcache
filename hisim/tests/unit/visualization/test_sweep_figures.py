"""Tests for sweep_figures (V2)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from hisim.visualization import sweep_figures


def _disagg_row(
    *,
    topology: str,
    total_gpus: int,
    ttft: float,
    tput: float,
    p99: float,
    pareto: bool,
    p_repl: int | None,
    d_repl: int | None,
    prefill_q: float = 5.0,
    kv_tx: float = 2.0,
    decode_q: float = 7.0,
) -> dict:
    return {
        "topology": topology,
        "total_gpus": total_gpus,
        "mean_ttft_ms": ttft,
        "output_throughput": tput,
        "p99_e2e_latency_ms": p99,
        "is_pareto_optimal": pareto,
        "prefill_replicas": p_repl,
        "decode_replicas": d_repl,
        "mean_prefill_queue_ms": prefill_q,
        "mean_kv_transfer_ms": kv_tx,
        "mean_decode_queue_ms": decode_q,
    }


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    rows = [
        _disagg_row(topology="pd-p1tp2-d1tp2", total_gpus=4, ttft=50.0, tput=120.0, p99=300.0,
                    pareto=True, p_repl=1, d_repl=1),
        _disagg_row(topology="pd-p2tp2-d1tp2", total_gpus=6, ttft=40.0, tput=150.0, p99=280.0,
                    pareto=True, p_repl=2, d_repl=1),
        _disagg_row(topology="pd-p1tp2-d2tp2", total_gpus=6, ttft=80.0, tput=100.0, p99=400.0,
                    pareto=False, p_repl=1, d_repl=2),
        _disagg_row(topology="agg-tp4-dp1", total_gpus=4, ttft=60.0, tput=130.0, p99=350.0,
                    pareto=True, p_repl=None, d_repl=None),
    ]
    return pd.DataFrame(rows)


# ---------------- pareto_scatter ----------------

def test_pareto_scatter_returns_figure_with_traces(mixed_df):
    fig = sweep_figures.pareto_scatter(mixed_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    names = [t.name for t in fig.data]
    assert "Pareto frontier" in names


def test_pareto_scatter_empty_df_returns_figure_with_annotation():
    fig = sweep_figures.pareto_scatter(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) >= 1


# ---------------- stage_breakdown_bar ----------------

def test_stage_breakdown_bar_stacks_three_stages(mixed_df):
    fig = sweep_figures.stage_breakdown_bar(mixed_df, top_k=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    assert fig.layout.barmode == "stack"


def test_stage_breakdown_bar_respects_top_k(mixed_df):
    fig = sweep_figures.stage_breakdown_bar(mixed_df, top_k=2)
    # each of the 3 traces should have exactly 2 x entries
    for trace in fig.data:
        assert len(trace.x) == 2


# ---------------- pd_resource_heatmap ----------------

def test_pd_resource_heatmap_disagg_rows(mixed_df):
    fig = sweep_figures.pd_resource_heatmap(mixed_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    heat = fig.data[0]
    # 2 unique prefill_replicas (1, 2), 2 unique decode_replicas (1, 2)
    assert len(heat.x) == 2
    assert len(heat.y) == 2


def test_pd_resource_heatmap_agg_only_degrades():
    agg_only = pd.DataFrame([
        _disagg_row(topology="agg-tp4-dp1", total_gpus=4, ttft=60.0, tput=130.0, p99=350.0,
                    pareto=True, p_repl=None, d_repl=None),
    ])
    fig = sweep_figures.pd_resource_heatmap(agg_only)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) >= 1
