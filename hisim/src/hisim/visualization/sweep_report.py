"""Sweep HTML report composer (V3).

Combines plotly figures from :mod:`sweep_figures` with a jinja2
template into a single self-contained HTML document. ``render_report``
takes an already-derived DataFrame (output of
``sweep_data.add_derived_columns``) and writes the report to disk.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly import graph_objects as go

from . import sweep_figures


__all__ = ("render_report",)


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_NAME = "report.html.j2"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _fig_to_html(fig: go.Figure, *, include_plotlyjs: bool | str) -> str:
    return fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"displaylogo": False},
    )


def _build_sections(df: pd.DataFrame) -> list[dict]:
    figs: list[tuple[str, go.Figure]] = [
        ("Pareto: throughput vs TTFT", sweep_figures.pareto_scatter(df)),
        ("Stage breakdown (top configs)", sweep_figures.stage_breakdown_bar(df)),
        ("PD replica heatmap (p99 e2e latency)", sweep_figures.pd_resource_heatmap(df)),
    ]
    sections: list[dict] = []
    # Embed plotly.js once on the first figure, then reference for the rest.
    first = True
    for title, fig in figs:
        sections.append(
            {
                "title": title,
                "html": _fig_to_html(fig, include_plotlyjs="cdn" if first else False),
            }
        )
        first = False
    return sections


def render_report(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "HiSim Sweep Report",
    source: str = "",
) -> Path:
    """Render ``df`` to a self-contained HTML report at ``output_path``.

    Returns the resolved output path. Parent directory is created if
    missing. The DataFrame is expected to already contain the columns
    produced by :func:`sweep_data.add_derived_columns`; missing
    optional columns are tolerated by the figure layer.
    """

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    n_rows = int(len(df))
    if "is_pareto_optimal" in df.columns:
        n_pareto = int(df["is_pareto_optimal"].fillna(False).astype(bool).sum())
    else:
        n_pareto = 0
    if "disagg_enabled" in df.columns:
        n_disagg = int(df["disagg_enabled"].fillna(False).astype(bool).sum())
    else:
        n_disagg = 0
    n_agg = n_rows - n_disagg

    context = {
        "title": title,
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": n_rows,
        "n_pareto": n_pareto,
        "n_disagg": n_disagg,
        "n_agg": n_agg,
        "source": source or "(not specified)",
        "sections": _build_sections(df),
    }

    template = _env().get_template(_TEMPLATE_NAME)
    output.write_text(template.render(**context), encoding="utf-8")
    return output
