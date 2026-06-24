"""V1 tests for the sweep data layer.

Covers ``load_sweep`` enrichment (result_file + sim_config) and
``add_derived_columns`` (total_gpus / topology / pd_ratio / pareto).
Synthetic fixtures only — no dependency on a real sweep run.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from hisim.visualization.sweep_data import add_derived_columns, load_sweep


# --------------------------------------------------------------------- helpers


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _disagg_sim_cfg(
    *,
    p_tp: int,
    p_repl: int,
    d_tp: int,
    d_repl: int,
    bw: float = 200.0,
    lat: float = 5.0,
    backend: str = "single_process",
) -> dict[str, object]:
    return {
        "tp_size": 1,
        "dp_size": 1,
        "disagg": {
            "enabled": True,
            "backend": backend,
            "prefill": {"tp_size": p_tp, "replicas": p_repl},
            "decode": {"tp_size": d_tp, "replicas": d_repl},
            "kv_transfer": {"bw_gbps": bw, "latency_us": lat},
        },
    }


def _agg_sim_cfg(tp: int, dp: int) -> dict[str, object]:
    return {"tp_size": tp, "dp_size": dp}


# ------------------------------------------------------------------ load_sweep


def test_load_sweep_reads_csv_with_no_enrichment(tmp_path: Path) -> None:
    csv_path = _write_summary_csv(
        tmp_path / "summary.csv",
        [{"row_index": 0, "status": "ok", "mean_ttft_ms": 12.3}],
    )
    df = load_sweep(
        csv_path, enrich_with_result_files=False, enrich_with_sim_configs=False
    )
    assert list(df.columns) == ["row_index", "status", "mean_ttft_ms"]
    assert df.loc[0, "mean_ttft_ms"] == 12.3


def test_load_sweep_merges_result_file_pd_stage_keys(tmp_path: Path) -> None:
    result_path = _write_json(
        tmp_path / "run0" / "result.json",
        {
            "request_throughput": 10.0,
            "output_throughput": 320.0,
            "mean_ttft_ms": 50.0,
            "p99_e2e_latency_ms": 300.0,
            "mean_prefill_queue_ms": 7.5,
            "mean_kv_transfer_ms": 1.25,
            "mean_decode_queue_ms": 2.0,
        },
    )
    csv_path = _write_summary_csv(
        tmp_path / "summary.csv",
        [
            {
                "row_index": 0,
                "status": "ok",
                "result_file": str(result_path),
                "sim_config": "",
            }
        ],
    )
    df = load_sweep(csv_path)
    assert df.loc[0, "request_throughput"] == 10.0
    assert df.loc[0, "mean_prefill_queue_ms"] == 7.5
    assert df.loc[0, "mean_kv_transfer_ms"] == 1.25
    assert df.loc[0, "mean_decode_queue_ms"] == 2.0


def test_load_sweep_tolerates_missing_result_file(tmp_path: Path) -> None:
    csv_path = _write_summary_csv(
        tmp_path / "summary.csv",
        [
            {
                "row_index": 0,
                "status": "ok",
                "result_file": str(tmp_path / "does_not_exist.json"),
            }
        ],
    )
    df = load_sweep(csv_path)
    # No crash; PD stage columns are absent (no keys merged).
    assert "mean_prefill_queue_ms" not in df.columns


def test_load_sweep_extracts_disagg_topology_from_sim_config(tmp_path: Path) -> None:
    sim_cfg_path = _write_json(
        tmp_path / "run0" / "sim_config.json",
        _disagg_sim_cfg(p_tp=4, p_repl=2, d_tp=2, d_repl=4),
    )
    csv_path = _write_summary_csv(
        tmp_path / "summary.csv",
        [
            {
                "row_index": 0,
                "status": "ok",
                "sim_config": str(sim_cfg_path),
            }
        ],
    )
    df = load_sweep(csv_path)
    assert bool(df.loc[0, "disagg_enabled"]) is True
    assert int(df.loc[0, "prefill_tp"]) == 4
    assert int(df.loc[0, "prefill_replicas"]) == 2
    assert int(df.loc[0, "decode_tp"]) == 2
    assert int(df.loc[0, "decode_replicas"]) == 4
    assert df.loc[0, "kv_bw_gbps"] == 200.0
    assert df.loc[0, "kv_latency_us"] == 5.0


def test_load_sweep_marks_non_disagg_sim_config_as_aggregated(tmp_path: Path) -> None:
    sim_cfg_path = _write_json(
        tmp_path / "run0" / "sim_config.json", _agg_sim_cfg(tp=8, dp=1)
    )
    csv_path = _write_summary_csv(
        tmp_path / "summary.csv",
        [{"row_index": 0, "status": "ok", "sim_config": str(sim_cfg_path)}],
    )
    df = load_sweep(csv_path)
    assert bool(df.loc[0, "disagg_enabled"]) is False
    assert int(df.loc[0, "agg_tp"]) == 8
    assert int(df.loc[0, "agg_dp"]) == 1
    assert math.isnan(df.loc[0, "prefill_tp"])


# ----------------------------------------------------- add_derived_columns


def _df_with_topology(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_add_derived_columns_computes_total_gpus_for_agg_and_disagg() -> None:
    df = _df_with_topology(
        [
            {
                "disagg_enabled": True,
                "prefill_tp": 4,
                "prefill_replicas": 2,
                "decode_tp": 2,
                "decode_replicas": 4,
                "mean_ttft_ms": 10.0,
                "output_throughput": 100.0,
            },
            {
                "disagg_enabled": False,
                "agg_tp": 8,
                "agg_dp": 2,
                "mean_ttft_ms": 20.0,
                "output_throughput": 200.0,
            },
        ]
    )
    out = add_derived_columns(df)
    assert out.loc[0, "total_gpus"] == 4 * 2 + 2 * 4  # 16
    assert out.loc[1, "total_gpus"] == 8 * 2  # 16


def test_add_derived_columns_topology_labels() -> None:
    df = _df_with_topology(
        [
            {
                "disagg_enabled": True,
                "prefill_tp": 4,
                "prefill_replicas": 2,
                "decode_tp": 2,
                "decode_replicas": 4,
            },
            {"disagg_enabled": False, "agg_tp": 8, "agg_dp": 1},
        ]
    )
    out = add_derived_columns(df)
    assert out.loc[0, "topology"] == "pd-p2tp4-d4tp2"
    assert out.loc[1, "topology"] == "agg-tp8-dp1"


def test_add_derived_columns_pd_ratio_nan_for_agg_rows() -> None:
    df = _df_with_topology(
        [
            {
                "disagg_enabled": True,
                "prefill_replicas": 3,
                "decode_replicas": 6,
                "prefill_tp": 1,
                "decode_tp": 1,
            },
            {"disagg_enabled": False, "agg_tp": 4},
        ]
    )
    out = add_derived_columns(df)
    assert out.loc[0, "pd_ratio"] == pytest.approx(0.5)
    assert math.isnan(out.loc[1, "pd_ratio"])


def test_add_derived_columns_pareto_optimal_marks_frontier() -> None:
    # Lower ttft + higher throughput is better. With four points where
    # only one is strictly dominated by another, the other three are
    # Pareto-optimal.
    df = _df_with_topology(
        [
            {"mean_ttft_ms": 10.0, "output_throughput": 100.0},  # optimal
            {"mean_ttft_ms": 20.0, "output_throughput": 200.0},  # optimal
            {"mean_ttft_ms": 30.0, "output_throughput": 50.0},   # dominated by row 0
            {"mean_ttft_ms": 15.0, "output_throughput": 150.0},  # optimal (in between)
        ]
    )
    out = add_derived_columns(df)
    assert list(out["is_pareto_optimal"]) == [True, True, False, True]
