"""Sweep figure factories (V2).

Each public function takes a ``pandas.DataFrame`` produced by
``sweep_data.add_derived_columns`` and returns a
``plotly.graph_objects.Figure``. No function in this module performs
file IO. Empty / degenerate inputs return an empty Figure with an
annotation so downstream report rendering never crashes.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


__all__ = (
    "pareto_scatter",
    "stage_breakdown_bar",
    "pd_resource_heatmap",
)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def pareto_scatter(
    df: pd.DataFrame,
    *,
    x: str = "mean_ttft_ms",
    y: str = "output_throughput",
) -> go.Figure:
    """Scatter of (x, y) colored by topology; Pareto points outlined.

    Marker size encodes ``total_gpus``. Expects derived columns
    produced by :func:`sweep_data.add_derived_columns`.
    """

    if df.empty:
        return _empty_figure("No rows to plot")

    missing = _require_columns(df, [x, y, "topology", "total_gpus", "is_pareto_optimal"])
    if missing:
        return _empty_figure(f"Missing columns: {', '.join(missing)}")

    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return _empty_figure(f"No finite ({x}, {y}) pairs")

    plot_df["_size"] = plot_df["total_gpus"].fillna(1).clip(lower=1)

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color="topology",
        size="_size",
        size_max=24,
        hover_data={
            "topology": True,
            "total_gpus": True,
            "is_pareto_optimal": True,
            "_size": False,
        },
    )

    pareto_df = plot_df[plot_df["is_pareto_optimal"] == True]  # noqa: E712
    if not pareto_df.empty:
        pareto_sorted = pareto_df.sort_values(by=x)
        fig.add_trace(
            go.Scatter(
                x=pareto_sorted[x],
                y=pareto_sorted[y],
                mode="lines+markers",
                name="Pareto frontier",
                line={"color": "black", "dash": "dash"},
                marker={"symbol": "circle-open", "size": 14, "color": "black"},
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Pareto: {y} vs {x}",
        xaxis_title=x,
        yaxis_title=y,
        legend_title="Topology",
    )
    return fig


def stage_breakdown_bar(
    df: pd.DataFrame,
    *,
    top_k: int = 10,
    sort_by: str = "output_throughput",
) -> go.Figure:
    """Stacked bar of mean prefill-queue / kv-transfer / decode-queue per config.

    Top-``top_k`` rows by ``sort_by`` descending. Bars stacked in the
    natural request lifecycle order.
    """

    stage_cols = [
        "mean_prefill_queue_ms",
        "mean_kv_transfer_ms",
        "mean_decode_queue_ms",
    ]

    if df.empty:
        return _empty_figure("No rows to plot")

    missing = _require_columns(df, ["topology", sort_by])
    if missing:
        return _empty_figure(f"Missing columns: {', '.join(missing)}")

    present_stages = [c for c in stage_cols if c in df.columns]
    if not present_stages:
        return _empty_figure("No stage timing columns present")

    sorted_df = df.sort_values(by=sort_by, ascending=False).head(top_k).copy()
    labels = sorted_df["topology"].astype(str).tolist()

    fig = go.Figure()
    for stage in present_stages:
        fig.add_trace(
            go.Bar(
                name=stage.replace("mean_", "").replace("_ms", ""),
                x=labels,
                y=sorted_df[stage].fillna(0.0).tolist(),
            )
        )

    fig.update_layout(
        barmode="stack",
        title=f"Stage breakdown (top {top_k} by {sort_by})",
        xaxis_title="Topology",
        yaxis_title="Latency (ms)",
        legend_title="Stage",
    )
    return fig


def pd_resource_heatmap(
    df: pd.DataFrame,
    *,
    metric: str = "p99_e2e_latency_ms",
) -> go.Figure:
    """Heatmap of disagg configs over (prefill_replicas, decode_replicas).

    Aggregated-only DataFrames degrade to an empty figure with an
    annotation. If multiple rows share a cell, the mean of ``metric``
    is plotted.
    """

    if df.empty:
        return _empty_figure("No rows to plot")

    missing = _require_columns(df, ["prefill_replicas", "decode_replicas", metric])
    if missing:
        return _empty_figure(f"Missing columns: {', '.join(missing)}")

    disagg_df = df.dropna(subset=["prefill_replicas", "decode_replicas", metric])
    if disagg_df.empty:
        return _empty_figure("No disagg rows available")

    pivot = disagg_df.pivot_table(
        index="decode_replicas",
        columns="prefill_replicas",
        values=metric,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="Viridis",
            colorbar={"title": metric},
        )
    )
    fig.update_layout(
        title=f"{metric} over PD replicas",
        xaxis_title="prefill_replicas",
        yaxis_title="decode_replicas",
    )
    return fig
