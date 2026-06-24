"""Sweep data layer.

Reads the ``summary.csv`` produced by ``aic_topn_to_hisim_sweep.py``
into a :class:`pandas.DataFrame` and, optionally, enriches it with
per-row metrics from each run's ``result_file`` (bench_serving JSON)
and disagg topology fields from each run's ``sim_config`` JSON.

Layering rule: this module uses only pandas + the standard library. It
must not import plotly, jinja2, streamlit, or any rendering tool.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Per-run bench_serving JSON keys that we promote into the dataframe
# when present. Anything missing is left as NaN.
_RESULT_FILE_KEYS: tuple[str, ...] = (
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "median_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_prefill_queue_ms",
    "p50_prefill_queue_ms",
    "p95_prefill_queue_ms",
    "p99_prefill_queue_ms",
    "mean_kv_transfer_ms",
    "p50_kv_transfer_ms",
    "p95_kv_transfer_ms",
    "p99_kv_transfer_ms",
    "mean_decode_queue_ms",
    "p50_decode_queue_ms",
    "p95_decode_queue_ms",
    "p99_decode_queue_ms",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON or None if the file is missing / unreadable."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _extract_result_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    """Pull the PD-aware percentile keys we care about out of a result_file."""
    if not payload:
        return {}
    out: dict[str, float] = {}
    for key in _RESULT_FILE_KEYS:
        if key in payload and payload[key] is not None:
            try:
                out[key] = float(payload[key])
            except (TypeError, ValueError):
                continue
    return out


def _extract_disagg_topology(sim_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Pull a flat topology dict out of a sim_config JSON.

    Always returns the same set of keys so the resulting DataFrame has a
    stable schema. For non-disagg configs all disagg_* fields are NaN /
    False / None.
    """
    flat: dict[str, Any] = {
        "disagg_enabled": False,
        "disagg_backend": None,
        "prefill_tp": math.nan,
        "prefill_replicas": math.nan,
        "decode_tp": math.nan,
        "decode_replicas": math.nan,
        "kv_bw_gbps": math.nan,
        "kv_latency_us": math.nan,
        "agg_tp": math.nan,
        "agg_dp": math.nan,
    }
    if not sim_cfg:
        return flat
    # Aggregated tp/dp live at the top of SchedulerConfig.
    for src, dst in (("tp_size", "agg_tp"), ("dp_size", "agg_dp")):
        if src in sim_cfg and sim_cfg[src] is not None:
            try:
                flat[dst] = int(sim_cfg[src])
            except (TypeError, ValueError):
                pass
    disagg = sim_cfg.get("disagg")
    if not disagg or not disagg.get("enabled"):
        return flat
    flat["disagg_enabled"] = True
    flat["disagg_backend"] = disagg.get("backend")
    for role, prefix in (("prefill", "prefill"), ("decode", "decode")):
        block = disagg.get(role) or {}
        if "tp_size" in block:
            try:
                flat[f"{prefix}_tp"] = int(block["tp_size"])
            except (TypeError, ValueError):
                pass
        if "replicas" in block:
            try:
                flat[f"{prefix}_replicas"] = int(block["replicas"])
            except (TypeError, ValueError):
                pass
    kv = disagg.get("kv_transfer") or {}
    if "bw_gbps" in kv:
        try:
            flat["kv_bw_gbps"] = float(kv["bw_gbps"])
        except (TypeError, ValueError):
            pass
    if "latency_us" in kv:
        try:
            flat["kv_latency_us"] = float(kv["latency_us"])
        except (TypeError, ValueError):
            pass
    return flat


def load_sweep(
    summary_csv: Path | str,
    *,
    enrich_with_result_files: bool = True,
    enrich_with_sim_configs: bool = True,
) -> pd.DataFrame:
    """Load a sweep ``summary.csv`` and optionally enrich it per-row.

    Parameters
    ----------
    summary_csv:
        Path to the ``summary.csv`` written by ``aic_topn_to_hisim_sweep``.
    enrich_with_result_files:
        If True, attempt to read each row's ``result_file`` column and
        merge in PD stage percentiles + per-run percentiles. Missing
        files are tolerated silently.
    enrich_with_sim_configs:
        If True, attempt to read each row's ``sim_config`` column and
        extract disagg topology (replicas, tp, kv bw/latency). Missing
        or non-disagg configs leave disagg_* columns as NaN / False.
    """
    csv_path = Path(summary_csv)
    df = pd.read_csv(csv_path)

    extra_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        bag: dict[str, Any] = {}
        if enrich_with_result_files and isinstance(row.get("result_file"), str):
            payload = _load_json(Path(row["result_file"]))
            bag.update(_extract_result_metrics(payload))
        if enrich_with_sim_configs and isinstance(row.get("sim_config"), str):
            cfg = _load_json(Path(row["sim_config"]))
            bag.update(_extract_disagg_topology(cfg))
        extra_rows.append(bag)

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        # Prefer existing summary columns over re-derived ones (the
        # summary.csv values are authoritative if a name clashes).
        for col in extra_df.columns:
            if col not in df.columns:
                df[col] = extra_df[col].values
    return df


def _pareto_mask(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Return a boolean Series marking Pareto-optimal rows.

    A row is dominated iff some other row has ``x_j <= x_i`` and
    ``y_j >= y_i`` with at least one strict. Rows where either column
    is NaN are False.
    """
    mask = pd.Series([False] * len(df), index=df.index)
    if x not in df.columns or y not in df.columns:
        return mask
    xs = df[x].to_numpy()
    ys = df[y].to_numpy()
    n = len(df)
    for i in range(n):
        xi, yi = xs[i], ys[i]
        if pd.isna(xi) or pd.isna(yi):
            continue
        dominated = False
        for j in range(n):
            if i == j:
                continue
            xj, yj = xs[j], ys[j]
            if pd.isna(xj) or pd.isna(yj):
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        mask.iloc[i] = not dominated
    return mask


def _topology_label(row: pd.Series) -> str:
    """Build a short topology label for one row."""
    if bool(row.get("disagg_enabled", False)):
        p_n = row.get("prefill_replicas")
        p_tp = row.get("prefill_tp")
        d_n = row.get("decode_replicas")
        d_tp = row.get("decode_tp")
        if not any(pd.isna(v) for v in (p_n, p_tp, d_n, d_tp)):
            return f"pd-p{int(p_n)}tp{int(p_tp)}-d{int(d_n)}tp{int(d_tp)}"
        return "pd-?"
    tp = row.get("agg_tp")
    dp = row.get("agg_dp")
    parts: list[str] = ["agg"]
    if not pd.isna(tp):
        parts.append(f"tp{int(tp)}")
    if not pd.isna(dp):
        parts.append(f"dp{int(dp)}")
    return "-".join(parts)


def _total_gpus(row: pd.Series) -> float:
    """Resource-normalization formula.

    Aggregated:  total_gpus = tp * dp  (dp defaults to 1 if missing)
    Disagg:      total_gpus = (p)replicas * (p)tp + (d)replicas * (d)tp
    Returns NaN if the relevant fields are missing.
    """
    if bool(row.get("disagg_enabled", False)):
        p_n = row.get("prefill_replicas")
        p_tp = row.get("prefill_tp")
        d_n = row.get("decode_replicas")
        d_tp = row.get("decode_tp")
        vals = (p_n, p_tp, d_n, d_tp)
        if any(pd.isna(v) for v in vals):
            return math.nan
        return float(int(p_n) * int(p_tp) + int(d_n) * int(d_tp))
    tp = row.get("agg_tp")
    dp = row.get("agg_dp")
    if pd.isna(tp):
        return math.nan
    dp_v = 1 if pd.isna(dp) else int(dp)
    return float(int(tp) * dp_v)


def add_derived_columns(
    df: pd.DataFrame,
    *,
    pareto_x: str = "mean_ttft_ms",
    pareto_y: str = "output_throughput",
) -> pd.DataFrame:
    """Return a copy of ``df`` with derived columns appended.

    Added columns:
      - ``total_gpus``: float per :func:`_total_gpus`
      - ``topology``: short label per :func:`_topology_label`
      - ``pd_ratio``: prefill_replicas / decode_replicas (NaN for agg
        rows or when either replica count is missing)
      - ``is_pareto_optimal``: bool on the (``pareto_x``, ``pareto_y``)
        plane; lower ``pareto_x`` and higher ``pareto_y`` is better.
    """
    out = df.copy()
    out["total_gpus"] = out.apply(_total_gpus, axis=1)
    out["topology"] = out.apply(_topology_label, axis=1)

    def _ratio(row: pd.Series) -> float:
        if not bool(row.get("disagg_enabled", False)):
            return math.nan
        p = row.get("prefill_replicas")
        d = row.get("decode_replicas")
        if pd.isna(p) or pd.isna(d) or d == 0:
            return math.nan
        return float(p) / float(d)

    out["pd_ratio"] = out.apply(_ratio, axis=1)
    out["is_pareto_optimal"] = _pareto_mask(out, pareto_x, pareto_y)
    return out


def build_comparison_candidates(
    sweep_df: pd.DataFrame, single_rows: list[dict[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Build labels for the dashboard comparison tray.

    Combines rows from the loaded sweep ``DataFrame`` with ad-hoc
    single-run rows (same schema) into a stable label list plus a
    label -> row-dict mapping. Pure pandas/stdlib so it is unit-testable
    without streamlit.
    """
    candidates: dict[str, dict[str, Any]] = {}
    labels: list[str] = []

    def _label(source: str, idx: int, row: dict[str, Any]) -> str:
        topo = str(row.get("topology", "?"))
        return f"{source}#{idx} \u00b7 {topo}"

    if sweep_df is not None and not sweep_df.empty:
        for i, row in enumerate(sweep_df.to_dict(orient="records")):
            lbl = _label("sweep", i, row)
            candidates[lbl] = row
            labels.append(lbl)
    for i, row in enumerate(single_rows or []):
        lbl = _label("single", i, row)
        candidates[lbl] = row
        labels.append(lbl)
    return labels, candidates


__all__: Iterable[str] = (
    "load_sweep",
    "add_derived_columns",
    "build_comparison_candidates",
)
