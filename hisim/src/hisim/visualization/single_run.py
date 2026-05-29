"""Single-run helper for the V6 dashboard.

Runs ONE PD-disagg simulation in-process using BackendA (single-process,
virtual-time) and returns a single summary row in the same schema as
``sweep_data.add_derived_columns`` produces. No file IO.

This module intentionally duplicates the event loop from
``hisim/tools/pd_demo.py`` rather than refactoring it, to keep the viz
layer decoupled from CLI-shaped code. Use this from the dashboard.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

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
from hisim.simulation.types import RequestStats, SchedulerConfig
from hisim.simulation.utils import calc_metrics


__all__ = ("SingleRunConfig", "run_single")


@dataclass(frozen=True)
class SingleRunConfig:
    prefill_device: str = "h100_sxm"
    decode_device: str = "h100_sxm"
    prefill_tp: int = 1
    decode_tp: int = 1
    prefill_replicas: int = 1
    decode_replicas: int = 1
    bw_gbps: float = 100.0
    latency_us: float = 10.0
    num_requests: int = 8
    burst: bool = False


# ---------------------------------------------------------------------------
# Minimal analytic predictor (same shape as pd_demo's _AnalyticPredictor).
# Kept private to this module to avoid coupling viz to demo internals.
# ---------------------------------------------------------------------------


class _AnalyticPredictor:
    _PREFILL_US_PER_TOK = {
        "h100_sxm": 0.4, "h200_sxm": 0.3, "h20": 1.2, "a100_sxm": 0.8,
    }
    _DECODE_US_PER_STEP = {
        "h100_sxm": 12.0, "h200_sxm": 10.0, "h20": 25.0, "a100_sxm": 18.0,
    }

    def __init__(self, model, hw, config, **_kwargs):
        device = hw.name
        tp = max(1, config.tp_size)
        self.prefill_us_per_tok = self._PREFILL_US_PER_TOK.get(device, 0.5) / tp
        self.decode_us_per_step = self._DECODE_US_PER_STEP.get(device, 15.0) / tp

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self.prefill_us_per_tok * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        if isinstance(past_kv_length, int):
            total = past_kv_length * batch_size
        else:
            total = sum(int(x) for x in past_kv_length)
        base = self.decode_us_per_step * (1 + 0.05 * max(0, batch_size - 1))
        return (base + 0.001 * total) * 1e-6


@dataclass
class _StubHW:
    name: str


def _model() -> ModelInfo:
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="dashboard-demo",
    )


# Fixed deterministic workload so reruns with the same config produce the
# same numbers — important for caching and reproducibility in the UI.
_INPUT_LENS = (256, 512, 1024, 768, 2048, 384, 1536, 640, 320, 1280)
_OUTPUT_LENS = (64, 128, 32, 96, 48, 80, 64, 128, 96, 48)


def run_single(cfg: SingleRunConfig) -> dict[str, Any]:
    """Execute one BackendA simulation and return a single summary row.

    Returned keys match the schema downstream (``sweep_data`` /
    ``sweep_figures``) expects.
    """
    model = _model()
    base_sched = SchedulerConfig(
        model=model,
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )
    disagg_cfg = DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(
            device_name=cfg.prefill_device,
            tp_size=cfg.prefill_tp,
            replicas=cfg.prefill_replicas,
            max_running_per_replica=8,
        ),
        decode=RolePredictorConfig(
            device_name=cfg.decode_device,
            tp_size=cfg.decode_tp,
            replicas=cfg.decode_replicas,
            max_running_per_replica=64,
        ),
        kv_transfer=BandwidthTransferConfig(
            bw_gbps=cfg.bw_gbps, latency_us=cfg.latency_us
        ),
    )
    bundle = build_disagg(
        model=model,
        base_sched_config=base_sched,
        disagg_config=disagg_cfg,
        predictor_factory=_AnalyticPredictor,
        hw_factory=lambda name: _StubHW(name),
    )
    backend = BackendA(bundle)
    decode_replicas = backend.decode_pool_size()

    reqs: list[PDRequestState] = []
    for i in range(cfg.num_requests):
        reqs.append(PDRequestState(
            rid=f"r{i}",
            arrival_time=0.0 if cfg.burst else i * 0.002,
            phase=RequestPhase.WAITING_PREFILL,
            input_length=_INPUT_LENS[i % len(_INPUT_LENS)],
            output_length=_OUTPUT_LENS[i % len(_OUTPUT_LENS)],
        ))
    pd_states = {r.rid: r for r in reqs}

    events: list[tuple[float, int, str, Any]] = []
    seq = 0

    def push(t: float, kind: str, payload: Any) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, payload))

    ttft: dict[str, float] = {}
    e2e: dict[str, float] = {}
    decode_inflight: dict[int, list[PDRequestState]] = {}

    def _tick_decode(now: float) -> None:
        drain_kv_ready_and_admit_decode(backend, now=now, capacity=10**6)
        in_flight = {r.rid for batch in decode_inflight.values() for r in batch}
        pending = [
            s for s in pd_states.values()
            if s.phase == RequestPhase.RUNNING_DECODE and s.rid not in in_flight
        ]
        if not pending:
            return
        idle = [i for i in range(decode_replicas) if i not in decode_inflight]
        if not idle:
            return
        buckets: list[list[PDRequestState]] = [[] for _ in idle]
        for j, r in enumerate(pending):
            buckets[j % len(idle)].append(r)
        for slot, ridx in enumerate(idle):
            batch = buckets[slot]
            if not batch:
                continue
            lat = decode_batch_latency(backend, batch, now)
            end_t = now + lat
            decode_inflight[ridx] = batch
            for r in batch:
                if r.rid not in ttft:
                    ttft[r.rid] = end_t - r.arrival_time
            push(end_t, "decode_step_done", ridx)

    for r in reqs:
        push(r.arrival_time, "arrive", r)

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
            _tick_decode(now)
        elif kind == "decode_step_done":
            ridx = payload
            batch = decode_inflight.pop(ridx, [])
            backend.on_decode_step_done_batch(batch, now)
            for r in batch:
                if r.phase == RequestPhase.FINISHED:
                    e2e[r.rid] = now - r.arrival_time
            _tick_decode(now)

    completed = [r for r in reqs if r.phase == RequestPhase.FINISHED]
    n_done = len(completed)
    mean_ttft = (sum(ttft.values()) / max(1, len(ttft))) * 1000.0 if ttft else math.nan
    sim_end = max(e2e.values()) if e2e else 0.0
    out_tokens = sum(r.output_length for r in completed)
    output_throughput = (out_tokens / sim_end) if sim_end > 0 else 0.0

    metrics: dict[str, Any] = {}
    if completed:
        stats_list: list[RequestStats] = []
        for r in completed:
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
            stats_list.append(s)
        metrics = calc_metrics(stats_list)

    total_gpus = (
        cfg.prefill_replicas * cfg.prefill_tp + cfg.decode_replicas * cfg.decode_tp
    )
    topology = (
        f"pd-p{cfg.prefill_replicas}tp{cfg.prefill_tp}"
        f"-d{cfg.decode_replicas}tp{cfg.decode_tp}"
    )
    pd_ratio = (
        cfg.prefill_replicas / cfg.decode_replicas
        if cfg.decode_replicas > 0 else math.nan
    )

    row: dict[str, Any] = {
        "topology": topology,
        "total_gpus": total_gpus,
        "pd_ratio": pd_ratio,
        "is_pareto_optimal": False,  # caller can recompute over the set
        "disagg_enabled": True,
        "disagg_backend": "single_process",
        "prefill_tp": cfg.prefill_tp,
        "prefill_replicas": cfg.prefill_replicas,
        "decode_tp": cfg.decode_tp,
        "decode_replicas": cfg.decode_replicas,
        "kv_bw_gbps": cfg.bw_gbps,
        "kv_latency_us": cfg.latency_us,
        "num_requests": cfg.num_requests,
        "completed": n_done,
        "mean_ttft_ms": mean_ttft,
        "output_throughput": output_throughput,
    }
    # Stage percentiles + per-metric keys produced by calc_metrics.
    for k, v in metrics.items():
        row.setdefault(k, v)
    return row
