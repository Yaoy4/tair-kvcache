"""Unit tests for single_run (V6)."""

from __future__ import annotations

import pytest

from hisim.visualization.single_run import SingleRunConfig, run_single


def test_run_single_returns_expected_schema():
    row = run_single(SingleRunConfig(num_requests=4))
    # Required schema for downstream figures.
    for key in (
        "topology", "total_gpus", "pd_ratio", "disagg_enabled",
        "prefill_tp", "prefill_replicas", "decode_tp", "decode_replicas",
        "kv_bw_gbps", "kv_latency_us",
        "mean_ttft_ms", "output_throughput",
        "mean_prefill_queue_ms", "mean_kv_transfer_ms", "mean_decode_queue_ms",
    ):
        assert key in row, f"missing key: {key}"


def test_run_single_topology_and_gpus_match_inputs():
    cfg = SingleRunConfig(
        prefill_tp=4, prefill_replicas=2,
        decode_tp=2, decode_replicas=3,
        num_requests=4,
    )
    row = run_single(cfg)
    assert row["topology"] == "pd-p2tp4-d3tp2"
    assert row["total_gpus"] == 2 * 4 + 3 * 2  # 14


def test_run_single_completes_all_requests():
    cfg = SingleRunConfig(num_requests=6, prefill_replicas=2, decode_replicas=2)
    row = run_single(cfg)
    assert row["completed"] == 6
    assert row["mean_ttft_ms"] > 0
    assert row["output_throughput"] > 0


def test_run_single_kv_bandwidth_affects_transfer_time():
    fast = run_single(SingleRunConfig(num_requests=4, bw_gbps=400.0, latency_us=1.0))
    slow = run_single(SingleRunConfig(num_requests=4, bw_gbps=10.0, latency_us=1.0))
    assert slow["mean_kv_transfer_ms"] > fast["mean_kv_transfer_ms"]


def test_run_single_deterministic_same_config():
    cfg = SingleRunConfig(num_requests=4)
    a = run_single(cfg)
    b = run_single(cfg)
    assert a["mean_ttft_ms"] == pytest.approx(b["mean_ttft_ms"])
    assert a["output_throughput"] == pytest.approx(b["output_throughput"])
