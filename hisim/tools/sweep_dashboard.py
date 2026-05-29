"""HiSim Sweep Dashboard (V5: passive).

Streamlit app that reads an existing summary.csv produced by
``hisim/tools/aic_topn_to_hisim_sweep.py`` and renders the same three
figures as the static report, plus sidebar filters and a row table.

Usage:
    streamlit run hisim/tools/sweep_dashboard.py -- --summary-csv path/to/summary.csv

The dashboard does NOT launch new HiSim runs in this phase (V6 adds
that). It is intentionally a thin viewer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from hisim.visualization import sweep_data, sweep_figures
from hisim.visualization.single_run import SingleRunConfig, run_single


# ---------------------------------------------------------------------------
# Args (parsed from the trailing argv after `--` in `streamlit run ... -- ...`)
# ---------------------------------------------------------------------------


def _parse_dashboard_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiSim sweep dashboard")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Default summary CSV to load on startup.",
    )
    # streamlit injects unknown args; ignore them gracefully.
    args, _ = parser.parse_known_args(argv)
    return args


@st.cache_data(show_spinner=False)
def _load(csv_path: str) -> pd.DataFrame:
    df = sweep_data.load_sweep(Path(csv_path))
    return sweep_data.add_derived_columns(df)


@st.cache_data(show_spinner=False)
def _run_single_cached(
    prefill_device: str,
    decode_device: str,
    prefill_tp: int,
    decode_tp: int,
    prefill_replicas: int,
    decode_replicas: int,
    bw_gbps: float,
    latency_us: float,
    num_requests: int,
    burst: bool,
) -> dict[str, Any]:
    cfg = SingleRunConfig(
        prefill_device=prefill_device,
        decode_device=decode_device,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
        prefill_replicas=prefill_replicas,
        decode_replicas=decode_replicas,
        bw_gbps=bw_gbps,
        latency_us=latency_us,
        num_requests=num_requests,
        burst=burst,
    )
    return run_single(cfg)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar(default_csv: Path | None) -> tuple[Path | None, dict[str, Any]]:
    st.sidebar.title("HiSim Sweep Viewer")
    st.sidebar.caption("V5 · passive: reads an existing summary.csv")

    csv_text = st.sidebar.text_input(
        "Summary CSV path",
        value=str(default_csv) if default_csv else "",
        placeholder="/path/to/output/.../summary.csv",
    )
    csv_path: Path | None = Path(csv_text) if csv_text else None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    filters: dict[str, Any] = {}
    # Filters are applied lazily in render after we know what columns exist.
    filters["only_pareto"] = st.sidebar.checkbox("Pareto-optimal only", value=False)
    filters["only_disagg"] = st.sidebar.checkbox("Disagg only", value=False)
    filters["only_agg"] = st.sidebar.checkbox("Aggregated only", value=False)
    return csv_path, filters


_DEVICE_OPTIONS = ("h100_sxm", "h200_sxm", "h20", "a100_sxm")


def _single_run_section() -> dict[str, Any] | None:
    """Render the Single Run sidebar section. Returns a config dict if the
    Run button was just clicked, else None."""
    st.sidebar.markdown("---")
    with st.sidebar.expander("Single Run (BackendA)", expanded=False):
        st.caption("Backend locked to single_process (BackendA). Analytic predictor.")
        col1, col2 = st.columns(2)
        with col1:
            p_dev = st.selectbox("Prefill device", _DEVICE_OPTIONS, key="sr_p_dev")
            p_tp = st.number_input("Prefill TP", 1, 16, 1, key="sr_p_tp")
            p_repl = st.number_input("Prefill replicas", 1, 16, 1, key="sr_p_repl")
        with col2:
            d_dev = st.selectbox("Decode device", _DEVICE_OPTIONS, key="sr_d_dev")
            d_tp = st.number_input("Decode TP", 1, 16, 1, key="sr_d_tp")
            d_repl = st.number_input("Decode replicas", 1, 16, 1, key="sr_d_repl")
        bw = st.number_input("KV bw (GB/s)", 1.0, 1000.0, 100.0, key="sr_bw")
        lat = st.number_input("KV latency (us)", 0.0, 1000.0, 10.0, key="sr_lat")
        nreq = st.number_input("Requests", 1, 64, 8, key="sr_nreq")
        burst = st.checkbox("Burst (all arrive at t=0)", value=False, key="sr_burst")
        run = st.button("Run simulation", key="sr_run", use_container_width=True)
        if run:
            return {
                "prefill_device": str(p_dev),
                "decode_device": str(d_dev),
                "prefill_tp": int(p_tp),
                "decode_tp": int(d_tp),
                "prefill_replicas": int(p_repl),
                "decode_replicas": int(d_repl),
                "bw_gbps": float(bw),
                "latency_us": float(lat),
                "num_requests": int(nreq),
                "burst": bool(burst),
            }
    return None


def _column_filter(df: pd.DataFrame, col: str, label: str) -> Any:
    if col not in df.columns:
        return None
    options = sorted({v for v in df[col].dropna().tolist()})
    if not options:
        return None
    return st.sidebar.multiselect(label, options=options, default=[])


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df
    if filters.get("only_pareto") and "is_pareto_optimal" in out.columns:
        out = out[out["is_pareto_optimal"].fillna(False).astype(bool)]
    if filters.get("only_disagg") and "disagg_enabled" in out.columns:
        out = out[out["disagg_enabled"].fillna(False).astype(bool)]
    if filters.get("only_agg") and "disagg_enabled" in out.columns:
        out = out[~out["disagg_enabled"].fillna(False).astype(bool)]
    for key in ("cache_scenario", "request_rate", "topology"):
        selected = filters.get(key)
        if selected and key in out.columns:
            out = out[out[key].isin(selected)]
    return out


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render(default_csv: Path | None) -> None:
    st.set_page_config(page_title="HiSim Sweep", layout="wide")
    csv_path, filters = _sidebar(default_csv)
    pending_run = _single_run_section()

    # Initialize session_state container for ad-hoc single runs.
    if "single_runs" not in st.session_state:
        st.session_state["single_runs"] = []

    if pending_run is not None:
        with st.spinner("Running BackendA simulation..."):
            row = _run_single_cached(**pending_run)
        st.session_state["single_runs"].append(row)

    st.title("HiSim Sweep Dashboard")

    # Show single-run results (if any) above the sweep section.
    single_rows = st.session_state.get("single_runs", [])
    if single_rows:
        with st.expander(f"Single runs ({len(single_rows)})", expanded=True):
            sr_df = pd.DataFrame(single_rows)
            st.dataframe(sr_df, use_container_width=True)
            cols = st.columns([1, 4])
            if cols[0].button("Clear", key="sr_clear"):
                st.session_state["single_runs"] = []
                st.rerun()

    if not csv_path:
        st.info("Enter a summary.csv path in the sidebar to begin.")
        return
    if not csv_path.exists():
        st.error(f"Summary CSV not found: {csv_path}")
        return

    df = _load(str(csv_path))

    # Late-bound filters that need columns from the loaded df.
    for key, label in (
        ("cache_scenario", "Cache scenario"),
        ("request_rate", "Request rate"),
        ("topology", "Topology"),
    ):
        sel = _column_filter(df, key, label)
        if sel is not None:
            filters[key] = sel

    filtered = _apply_filters(df, filters)

    # ---- header metrics ----
    n_rows = int(len(filtered))
    n_pareto = (
        int(filtered["is_pareto_optimal"].fillna(False).astype(bool).sum())
        if "is_pareto_optimal" in filtered.columns
        else 0
    )
    n_disagg = (
        int(filtered["disagg_enabled"].fillna(False).astype(bool).sum())
        if "disagg_enabled" in filtered.columns
        else 0
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", n_rows)
    col2.metric("Pareto", n_pareto)
    col3.metric("Disagg", n_disagg)
    col4.metric("Aggregated", n_rows - n_disagg)

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    # ---- figures ----
    tab_pareto, tab_stages, tab_heat, tab_table = st.tabs(
        ["Pareto", "Stage breakdown", "PD heatmap", "Table"]
    )
    with tab_pareto:
        st.plotly_chart(sweep_figures.pareto_scatter(filtered), use_container_width=True)
    with tab_stages:
        top_k = st.slider("Top K by throughput", min_value=1, max_value=50, value=10)
        st.plotly_chart(
            sweep_figures.stage_breakdown_bar(filtered, top_k=top_k),
            use_container_width=True,
        )
    with tab_heat:
        st.plotly_chart(sweep_figures.pd_resource_heatmap(filtered), use_container_width=True)
    with tab_table:
        st.dataframe(filtered, use_container_width=True)


def main() -> None:
    args = _parse_dashboard_args(sys.argv[1:])
    render(args.summary_csv)


# Streamlit imports this module top-level; running main() unconditionally
# at import is the canonical pattern for `streamlit run script.py`.
main()
