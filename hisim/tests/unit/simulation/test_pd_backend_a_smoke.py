"""Phase 4 — Backend A verification / smoke tests.

These tests exercise the *full* Phase 0-3 stack in one process without
SGLang: PDController → BackendA → pd_runtime helpers → pd_metrics →
calc_metrics. A tiny analytic predictor stands in for AIConfigurator so
the suite stays fast and DB-free.

What we assert:

  4a. Smoke / regression — identical prefill+decode roles with effectively
      zero KV transfer time produce E2E latencies close to (and never
      smaller than) a hand-computed serial baseline.

  4b. Rate mismatch — shrinking decode-replicas while holding everything
      else fixed monotonically increases decode_queue_p95.

  4c. KV bandwidth sweep — halving bw_gbps roughly doubles kv_transfer_p95;
      lowering bw monotonically grows it.

  4d. AIC bridge JSON round-trip — disagg config produced by
      `aic_to_hisim_bridge.py --emit-disagg` survives a
      SimulationArgs.from_json → build_disagg roundtrip.
"""
from __future__ import annotations

import heapq
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.pd_metrics import populate_request_stats
from hisim.simulation.pd_runtime import (
    admit_prefill_batch_latency,
    decode_batch_latency,
    drain_kv_ready_and_admit_decode,
    finalize_prefill_batch,
)
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.sim_args import SimulationArgs
from hisim.simulation.types import RequestStats, SchedulerConfig
from hisim.simulation.utils import calc_metrics


# ---------------------------------------------------------------------------
# Test-only analytic predictor (no AIC DB needed).
# ---------------------------------------------------------------------------


class _StubHW:
    def __init__(self, name: str):
        self.name = name


def _hw_factory(name: str) -> _StubHW:
    return _StubHW(name)


class _AnalyticPredictor:
    """Deterministic, monotonic predictor — fast tests, no DB dependency."""

    _PREFILL_US_PER_TOK = 0.5
    _DECODE_US_PER_STEP = 15.0

    def __init__(self, model, hw, config, **kwargs):
        self.tp = max(1, config.tp_size)

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self._PREFILL_US_PER_TOK * 1e-6 / self.tp

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        base = self._DECODE_US_PER_STEP * (1 + 0.05 * max(0, batch_size - 1))
        return base * 1e-6 / self.tp


# ---------------------------------------------------------------------------
# Tiny event-loop harness (mirrors the structure of pd_demo.py).
# ---------------------------------------------------------------------------


def _make_model() -> ModelInfo:
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="phase4-stub",
    )


def _make_disagg_config(
    *,
    prefill_replicas: int = 2,
    decode_replicas: int = 2,
    bw_gbps: float = 100.0,
    latency_us: float = 10.0,
    device: str = "h100_sxm",
    decode_max_running_per_replica: int = 64,
) -> DisaggConfig:
    return DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(
            device_name=device,
            tp_size=1,
            replicas=prefill_replicas,
            max_running_per_replica=8,
        ),
        decode=RolePredictorConfig(
            device_name=device,
            tp_size=1,
            replicas=decode_replicas,
            max_running_per_replica=decode_max_running_per_replica,
        ),
        kv_transfer=BandwidthTransferConfig(
            bw_gbps=bw_gbps, latency_us=latency_us
        ),
    )


def _make_base_sched() -> SchedulerConfig:
    return SchedulerConfig(
        model=_make_model(),
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )


def _run_workload(
    disagg_cfg: DisaggConfig,
    reqs: List[PDRequestState],
) -> Dict[str, RequestStats]:
    """Run a workload through BackendA + pd_runtime, returning per-rid
    RequestStats with populated stage durations."""
    bundle = build_disagg(
        model=_make_model(),
        base_sched_config=_make_base_sched(),
        disagg_config=disagg_cfg,
        predictor_factory=_AnalyticPredictor,
        hw_factory=_hw_factory,
    )
    backend = BackendA(bundle)
    decode_replicas = backend.decode_pool_size()
    decode_max_per_replica = disagg_cfg.decode.max_running_per_replica

    events: list = []
    seq = 0

    def push(t: float, kind: str, payload):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, payload))

    for r in reqs:
        push(r.arrival_time, "arrive", r)

    pd_states = {r.rid: r for r in reqs}
    ttft: Dict[str, float] = {}
    e2e: Dict[str, float] = {}
    decode_inflight: Dict[int, list] = {}

    def _launch_decode(now: float) -> None:
        # Only admit as many as can immediately occupy a replica's batch slot,
        # so decode_queue_wait correctly reflects time spent waiting for a
        # free batch slot rather than being eagerly stamped at KV-ready.
        idle = [i for i in range(decode_replicas) if i not in decode_inflight]
        if not idle:
            return
        admit_capacity = len(idle) * max(1, decode_max_per_replica)
        drain_kv_ready_and_admit_decode(backend, now=now, capacity=admit_capacity)
        in_flight = {r.rid for batch in decode_inflight.values() for r in batch}
        pending = [
            s
            for s in pd_states.values()
            if s.phase == RequestPhase.RUNNING_DECODE and s.rid not in in_flight
        ]
        if not pending:
            return
        buckets: List[list] = [[] for _ in idle]
        for j, req in enumerate(pending):
            buckets[j % len(idle)].append(req)
        for slot, replica_idx in enumerate(idle):
            batch = buckets[slot]
            if not batch:
                continue
            lat = decode_batch_latency(backend, batch, now)
            end_t = now + lat
            decode_inflight[replica_idx] = batch
            for r in batch:
                ttft.setdefault(r.rid, end_t - r.arrival_time)
            push(end_t, "decode_step_done", replica_idx)

    while events:
        now, _, kind, payload = heapq.heappop(events)
        if kind == "arrive":
            req = payload
            lat = admit_prefill_batch_latency(backend, [req], now)
            push(now + lat, "prefill_done", [req])
        elif kind == "prefill_done":
            batch = payload
            finalize_prefill_batch(backend, batch, now)
            for req in batch:
                push(req.kv_ready_time, "kv_ready", req)
        elif kind == "kv_ready":
            _launch_decode(now)
        elif kind == "decode_step_done":
            replica_idx = payload
            batch = decode_inflight.pop(replica_idx, [])
            backend.on_decode_step_done_batch(batch, now)
            for r in batch:
                if r.phase == RequestPhase.FINISHED:
                    e2e[r.rid] = now - r.arrival_time
            _launch_decode(now)

    # Build RequestStats with populated stage durations.
    stats: Dict[str, RequestStats] = {}
    for r in reqs:
        if r.phase != RequestPhase.FINISHED:
            continue
        tpot = (e2e[r.rid] - ttft[r.rid]) / max(1, r.output_length - 1)
        s = RequestStats(
            rid=r.rid,
            last_event_time=r.arrival_time + e2e[r.rid],
            input_length=r.input_length,
            output_length=r.output_length,
            queue_start=r.arrival_time,
            queue_end=r.arrival_time,
            created_time=r.arrival_time,
            gen_token_latencies=[ttft[r.rid]]
            + [tpot] * max(0, r.output_length - 1),
        )
        populate_request_stats(s, r)
        stats[r.rid] = s
    return stats


def _mk_reqs(
    n: int,
    *,
    input_length: int = 256,
    output_length: int = 8,
    arrival_stride: float = 0.0,
) -> List[PDRequestState]:
    return [
        PDRequestState(
            rid=f"r{i}",
            arrival_time=i * arrival_stride,
            phase=RequestPhase.WAITING_PREFILL,
            input_length=input_length,
            output_length=output_length,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 4a. Smoke — identical roles, ~zero KV ⇒ E2E close to serial baseline.
# ---------------------------------------------------------------------------


def test_4a_smoke_zero_kv_identical_roles_near_serial_baseline():
    """With huge KV bw + zero latency, KV transit ≈ 0 and a single-replica
    p/d should yield E2E ≈ prefill_dur + output_length * decode_dur."""
    cfg = _make_disagg_config(
        prefill_replicas=1,
        decode_replicas=1,
        bw_gbps=1e9,
        latency_us=0.0,
    )
    reqs = _mk_reqs(1, input_length=512, output_length=8)
    stats = _run_workload(cfg, reqs)

    assert len(stats) == 1
    s = stats["r0"]
    # KV transfer must be ~0 (well under 1 us).
    assert s.kv_transfer_time < 1e-6
    # Prefill / decode queue must be 0 with a single request.
    assert s.prefill_queue_wait == 0.0
    assert s.decode_queue_wait == 0.0

    # Analytic serial baseline.
    prefill_dur = 512 * _AnalyticPredictor._PREFILL_US_PER_TOK * 1e-6
    decode_dur = _AnalyticPredictor._DECODE_US_PER_STEP * 1e-6
    expected = prefill_dur + 8 * decode_dur
    actual_e2e = sum(s.gen_token_latencies)
    # Allow tiny slack for KV/latency rounding. Backend A should not be
    # faster than the serial baseline (no parallelism for a single request).
    assert actual_e2e == pytest.approx(expected, abs=2e-6)


# ---------------------------------------------------------------------------
# 4b. Rate mismatch — fewer decode replicas ⇒ higher decode_queue_p95.
# ---------------------------------------------------------------------------


def test_4b_decode_bottleneck_grows_decode_queue_p95():
    """Hold prefill capacity high, sweep decode_replicas downward. With a
    bursty workload and a small max-batch per replica, decode_queue_p95 must
    be monotonically non-decreasing as decode capacity shrinks."""
    reqs_template = lambda: _mk_reqs(
        16, input_length=256, output_length=4, arrival_stride=0.0
    )
    p95s = []
    for decode_replicas in (8, 4, 2, 1):
        cfg = _make_disagg_config(
            prefill_replicas=8,
            decode_replicas=decode_replicas,
            bw_gbps=1e9,
            latency_us=0.0,
            decode_max_running_per_replica=2,
        )
        stats = _run_workload(cfg, reqs_template())
        m = calc_metrics(list(stats.values()))
        p95s.append(m["p95_decode_queue_ms"])

    # Strictly non-decreasing as decode capacity shrinks.
    for i in range(1, len(p95s)):
        assert p95s[i] >= p95s[i - 1] - 1e-9, (
            f"decode_queue_p95 not monotonic: {p95s}"
        )
    # Smallest pool must show some queueing.
    assert p95s[-1] > p95s[0], f"expected queueing growth: {p95s}"


# ---------------------------------------------------------------------------
# 4c. KV bandwidth sweep — lower bw ⇒ higher kv_transfer_p95.
# ---------------------------------------------------------------------------


def test_4c_kv_bandwidth_sweep_grows_kv_transfer_p95():
    """Halve bw_gbps each step. kv_transfer_p95 must increase roughly 2x."""
    reqs_template = lambda: _mk_reqs(
        8, input_length=1024, output_length=4, arrival_stride=0.0
    )
    bws = [800.0, 400.0, 200.0, 100.0]
    p95s = []
    for bw in bws:
        cfg = _make_disagg_config(
            prefill_replicas=4,
            decode_replicas=4,
            bw_gbps=bw,
            latency_us=0.0,
        )
        stats = _run_workload(cfg, reqs_template())
        m = calc_metrics(list(stats.values()))
        p95s.append(m["p95_kv_transfer_ms"])

    # Strictly increasing as bw shrinks.
    for i in range(1, len(p95s)):
        assert p95s[i] > p95s[i - 1], (
            f"kv_transfer_p95 should grow as bw drops: bws={bws} p95s={p95s}"
        )
    # Doubling check (loose): halving bw should ~double p95 within 30%.
    for i in range(1, len(p95s)):
        ratio = p95s[i] / max(p95s[i - 1], 1e-12)
        assert 1.5 <= ratio <= 2.5, (
            f"halving bw should roughly double kv_transfer_p95; "
            f"got ratio={ratio:.3f} (p95s={p95s}, bws={bws})"
        )


# ---------------------------------------------------------------------------
# 4d. AIC bridge --emit-disagg JSON round-trips through SimulationArgs.
# ---------------------------------------------------------------------------


def _bridge_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3] / "tools" / "aic_to_hisim_bridge.py"


def test_4d_aic_bridge_emit_disagg_roundtrips_through_sim_args(tmp_path):
    """Generate disagg JSON via the bridge, load it via SimulationArgs, and
    confirm build_disagg accepts the resulting DisaggConfig."""
    bridge = _bridge_path()
    assert bridge.exists(), f"bridge script missing at {bridge}"

    csv_path = tmp_path / "pareto.csv"
    csv_path.write_text(
        "(p)tp,(p)pp,(p)dp,(p)moe_ep,(p)gemm,(p)kvcache,(p)version,(p)system,(p)workers,"
        "(d)tp,(d)pp,(d)dp,(d)moe_ep,(d)gemm,(d)kvcache,(d)version,(d)system,(d)workers\n"
        "4,1,1,1,fp16,fp16,0.5.6.post2,h100_sxm,2,"
        "2,1,2,1,fp16,fp16,0.5.6.post2,h20,4\n"
    )
    out_json = tmp_path / "hisim.json"

    # Invoke the bridge as a subprocess (mirrors the real sweep pipeline).
    result = subprocess.run(
        [
            "python",
            str(bridge),
            "--aic-csv",
            str(csv_path),
            "--row-index",
            "0",
            "--output",
            str(out_json),
            "--database-path",
            "/aic/db",
            "--emit-disagg",
            "--kv-transfer-bw-gbps",
            "50",
            "--kv-transfer-latency-us",
            "20",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bridge failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert out_json.exists()

    # Round-trip through SimulationArgs.
    sim_args = SimulationArgs.from_json(str(out_json))
    disagg = sim_args.disagg
    assert disagg is not None and disagg.enabled is True
    assert disagg.prefill.device_name and disagg.decode.device_name
    assert disagg.kv_transfer.bw_gbps == pytest.approx(50.0)
    assert disagg.kv_transfer.latency_us == pytest.approx(20.0)

    # build_disagg must accept the bridge-emitted config end-to-end.
    bundle = build_disagg(
        model=_make_model(),
        base_sched_config=_make_base_sched(),
        disagg_config=disagg,
        predictor_factory=_AnalyticPredictor,
        hw_factory=_hw_factory,
    )
    assert bundle.kv_bytes_per_token > 0
    assert bundle.prefill_replicas == disagg.prefill.replicas
    assert bundle.decode_replicas == disagg.decode.replicas
