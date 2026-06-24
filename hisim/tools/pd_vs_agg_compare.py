"""Apples-to-apples HiSim comparison: aggregated TP=8 vs PD 4P+4D.

Both layouts use 8 H100s and the exact same workload + the exact same
`_AnalyticPredictor` calibration from pd_demo.py, so any throughput / latency
delta is purely the architectural effect of disaggregation in HiSim.

AGG mode: ONE replica at TP=8 that, per request, runs prefill THEN decode
serially (mirrors AIC row 48: tp=8 dp=1, one worker). This is a fair stand-in
for the co-located scheduler that pd_demo's pool model cannot otherwise
express.

PD mode:  prefill_tp=4 replicas=1 + decode_tp=4 replicas=1 (mirrors AIC
disagg row 19), driven by BackendA exactly as in pd_demo.

Usage:
  PYTHONPATH=hisim/src:../aiconfigurator/src \
    venv/bin/python hisim/tools/pd_vs_agg_compare.py --requests 32 --burst
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

from hisim.spec import DataType, ModelInfo
from hisim.simulation.types import SchedulerConfig
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_factory import DisaggPredictors, build_disagg
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_runtime import (
    admit_prefill_batch_latency,
    decode_batch_latency,
    drain_kv_ready_and_admit_decode,
    finalize_prefill_batch,
)
from hisim.simulation.pd_types import PDRequestState, RequestPhase

# Reuse the SAME analytic predictor that pd_demo ships so the two modes use
# identical per-token / per-step coefficients.
from pd_demo import _AnalyticPredictor, _hw_factory  # type: ignore


MODEL = ModelInfo(
    hidden_size=4096,
    num_attention_heads=32,
    num_hidden_layers=32,
    vocab_size=32000,
    num_key_value_heads=8,
    head_dim=128,
    num_full_attention=32,
    name="demo-llama-7b-ish",
)

BASE_SCHED = SchedulerConfig(
    model=MODEL,
    tp_size=1,
    pp_size=1,
    data_type=DataType.FP16,
    kv_cache_data_type=DataType.FP16,
    backend_name="sglang",
    backend_version="0.5.10",
)


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

def make_workload(n: int, burst: bool) -> list[PDRequestState]:
    inputs = [256, 512, 1024, 768, 2048, 384, 1536, 640, 320, 1280]
    outputs = [64, 128, 32, 96, 48, 80, 64, 128, 96, 48]
    reqs: list[PDRequestState] = []
    for i in range(n):
        reqs.append(PDRequestState(
            rid=f"r{i}",
            arrival_time=0.0 if burst else i * 0.002,
            phase=RequestPhase.WAITING_PREFILL,
            input_length=inputs[i % len(inputs)],
            output_length=outputs[i % len(outputs)],
        ))
    return reqs


# ---------------------------------------------------------------------------
# AGG simulator: ONE replica @ TP=8, prefill→decode serial per request,
# with FCFS over arriving requests. No KV transfer (co-located).
# ---------------------------------------------------------------------------

@dataclass
class AggResult:
    rid: str
    arrival: float
    start: float
    prefill_done: float
    finish: float


def run_agg(reqs: list[PDRequestState], tp: int) -> tuple[list[AggResult], float]:
    hw = _hw_factory("h100_sxm")
    cfg = SchedulerConfig(
        model=MODEL, tp_size=tp, pp_size=1,
        data_type=DataType.FP16, kv_cache_data_type=DataType.FP16,
        backend_name="sglang", backend_version="0.5.10",
    )
    pred = _AnalyticPredictor(MODEL, hw, cfg)

    clock = 0.0
    results: list[AggResult] = []
    for r in sorted(reqs, key=lambda x: x.arrival_time):
        start = max(clock, r.arrival_time)
        p_lat = pred.predict_prefill_seconds(r.input_length)
        prefill_done = start + p_lat
        # Decode: r.output_length steps, batch=1, past_kv grows
        t = prefill_done
        past = r.input_length
        for _ in range(r.output_length):
            t += pred.predict_decode_seconds(batch_size=1, past_kv_length=past)
            past += 1
        finish = t
        results.append(AggResult(r.rid, r.arrival_time, start, prefill_done, finish))
        clock = finish
    makespan = max(r.finish for r in results)
    return results, makespan


# ---------------------------------------------------------------------------
# PD simulator: reuse pd_demo's BackendA loop
# ---------------------------------------------------------------------------

def run_pd(reqs: list[PDRequestState], prefill_tp: int, decode_tp: int,
           bw_gbps: float, latency_us: float) -> tuple[list[PDRequestState], float, dict, dict]:
    import heapq
    cfg = DisaggConfig(
        enabled=True, backend="single_process",
        prefill=RolePredictorConfig(
            device_name="h100_sxm", tp_size=prefill_tp,
            replicas=1, max_running_per_replica=8,
        ),
        decode=RolePredictorConfig(
            device_name="h100_sxm", tp_size=decode_tp,
            replicas=1, max_running_per_replica=64,
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=bw_gbps, latency_us=latency_us),
    )
    bundle: DisaggPredictors = build_disagg(
        model=MODEL, base_sched_config=BASE_SCHED, disagg_config=cfg,
        predictor_factory=_AnalyticPredictor, hw_factory=_hw_factory,
    )
    backend = BackendA(bundle)
    decode_replicas = backend.decode_pool_size()

    events: list = []
    seq = 0
    def push(t, kind, payload):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, payload))

    for r in reqs:
        push(r.arrival_time, "arrive", r)

    ttft: dict[str, float] = {}
    e2e: dict[str, float] = {}
    decode_inflight: dict[int, list] = {}
    pd_states = {r.rid: r for r in reqs}

    def _tick(now):
        drain_kv_ready_and_admit_decode(backend, now=now, capacity=10**6)
        in_flight = {r.rid for batch in decode_inflight.values() for r in batch}
        pending_dec = [
            s for s in pd_states.values()
            if s.phase == RequestPhase.RUNNING_DECODE and s.rid not in in_flight
        ]
        if not pending_dec:
            return
        idle = [i for i in range(decode_replicas) if i not in decode_inflight]
        if not idle:
            return
        buckets: list[list] = [[] for _ in idle]
        for j, req in enumerate(pending_dec):
            buckets[j % len(idle)].append(req)
        for slot, replica_idx in enumerate(idle):
            batch = buckets[slot]
            if not batch:
                continue
            lat = decode_batch_latency(backend, batch, now)
            end_t = now + lat
            decode_inflight[replica_idx] = batch
            for r in batch:
                if r.rid not in ttft:
                    ttft[r.rid] = end_t - r.arrival_time
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
            _tick(now)
        elif kind == "decode_step_done":
            replica_idx = payload
            batch = decode_inflight.pop(replica_idx, [])
            backend.on_decode_step_done_batch(batch, now)
            for r in batch:
                if r.phase == RequestPhase.FINISHED:
                    e2e[r.rid] = now - r.arrival_time
            _tick(now)

    makespan = max(e2e.get(r.rid, 0.0) + r.arrival_time for r in reqs)
    return reqs, makespan, ttft, e2e


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    k = max(0, min(len(xs2) - 1, int(round((p / 100) * (len(xs2) - 1)))))
    return xs2[k]


def report_agg(rs: list[AggResult], makespan: float, total_out: int) -> dict:
    e2e = [r.finish - r.arrival for r in rs]
    ttft = [r.prefill_done - r.arrival for r in rs]
    return {
        "mode": "AGG (TP=8, 1 replica, P→D serial)",
        "requests": len(rs),
        "makespan_ms": makespan * 1000,
        "throughput_req_per_s": len(rs) / makespan if makespan > 0 else 0.0,
        "output_tok_per_s": total_out / makespan if makespan > 0 else 0.0,
        "ttft_mean_ms": statistics.mean(ttft) * 1000,
        "ttft_p99_ms": _pct(ttft, 99) * 1000,
        "e2e_mean_ms": statistics.mean(e2e) * 1000,
        "e2e_p99_ms": _pct(e2e, 99) * 1000,
    }


def report_pd(rs: list[PDRequestState], makespan: float, total_out: int,
              ttft: dict, e2e: dict) -> dict:
    finished = [r for r in rs if r.phase == RequestPhase.FINISHED]
    ttft_l = [ttft[r.rid] for r in finished if r.rid in ttft]
    e2e_l = [e2e[r.rid] for r in finished if r.rid in e2e]
    return {
        "mode": "PD (prefill_tp=4 + decode_tp=4, BackendA)",
        "requests": len(finished),
        "makespan_ms": makespan * 1000,
        "throughput_req_per_s": len(finished) / makespan if makespan > 0 else 0.0,
        "output_tok_per_s": total_out / makespan if makespan > 0 else 0.0,
        "ttft_mean_ms": statistics.mean(ttft_l) * 1000 if ttft_l else 0.0,
        "ttft_p99_ms": _pct(ttft_l, 99) * 1000 if ttft_l else 0.0,
        "e2e_mean_ms": statistics.mean(e2e_l) * 1000 if e2e_l else 0.0,
        "e2e_p99_ms": _pct(e2e_l, 99) * 1000 if e2e_l else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--burst", action="store_true", default=True)
    ap.add_argument("--bw-gbps", type=float, default=100.0)
    ap.add_argument("--latency-us", type=float, default=10.0)
    args = ap.parse_args()

    reqs_a = make_workload(args.requests, args.burst)
    reqs_b = make_workload(args.requests, args.burst)
    total_out = sum(r.output_length for r in reqs_a)

    print("=" * 78)
    print(f"Apples-to-apples: AGG TP=8 vs PD 4P+4D — {args.requests} requests, "
          f"{'burst' if args.burst else 'staggered'} arrival")
    print("=" * 78)

    rs_a, mk_a = run_agg(reqs_a, tp=8)
    rep_a = report_agg(rs_a, mk_a, total_out)

    rs_b, mk_b, ttft_b, e2e_b = run_pd(reqs_b, prefill_tp=4, decode_tp=4,
                        bw_gbps=args.bw_gbps, latency_us=args.latency_us)
    rep_b = report_pd(rs_b, mk_b, total_out, ttft_b, e2e_b)

    keys = ["mode", "requests", "makespan_ms", "throughput_req_per_s",
            "output_tok_per_s", "ttft_mean_ms", "ttft_p99_ms",
            "e2e_mean_ms", "e2e_p99_ms"]
    w = 24
    print(f"{'metric':<{w}} | {'AGG TP=8':>22} | {'PD 4P+4D':>22} | {'PD/AGG':>10}")
    print("-" * 88)
    for k in keys:
        a, b = rep_a[k], rep_b[k]
        if isinstance(a, str):
            print(f"{k:<{w}} | {a:>22} | {b:>22} | {'':>10}")
        else:
            ratio = (b / a) if (isinstance(a, (int, float)) and a) else 0.0
            print(f"{k:<{w}} | {a:>22.3f} | {b:>22.3f} | {ratio:>10.2f}x")
    print("-" * 88)
    print("PD wins iff: throughput ratio >1, e2e/ttft ratios <1.")


if __name__ == "__main__":
    main()
