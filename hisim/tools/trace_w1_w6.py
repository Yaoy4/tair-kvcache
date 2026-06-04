"""Trace BackendA on W1..W6 with explicit per-request timestamps + summary.

Run:
    PYTHONPATH=hisim/src:../aiconfigurator/src \
        venv/bin/python hisim/tools/trace_w1_w6.py
"""
from __future__ import annotations

from typing import List

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_ab_harness import (
    WorkloadRequest,
    run_workload,
)
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_runtime import build_pd_backend, start_pd_backend, shutdown_pd_backend
from hisim.simulation.types import SchedulerConfig


class _AnalyticPredictor:
    _PREFILL_US_PER_TOK = 0.5
    _DECODE_US_PER_STEP = 10.0

    def __init__(self, model=None, hw=None, config=None, **kwargs):
        pass

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self._PREFILL_US_PER_TOK * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        return self._DECODE_US_PER_STEP * 1e-6 * max(1, batch_size)


class _StubHW:
    def __init__(self, name):
        self.name = name


def _hw(name):
    return _StubHW(name)


def _role_factory(model, base_sched_config, role):
    class _F:
        def __call__(self):
            return _AnalyticPredictor()
    return _F()


def _model():
    return ModelInfo(
        hidden_size=4096, num_attention_heads=32, num_hidden_layers=32,
        vocab_size=32000, num_key_value_heads=8, head_dim=128,
        num_full_attention=32, name="trace",
    )


def _sched():
    return SchedulerConfig(
        model=_model(), tp_size=1, pp_size=1,
        data_type=DataType.FP16, kv_cache_data_type=DataType.FP16,
        backend_name="sglang", backend_version="0.4.8",
    )


def _serial(n, input_len, output_len, interarrival=0.0):
    return [WorkloadRequest(rid=f"r{i}", arrival_time=i * interarrival,
                            input_length=input_len, output_length=output_len)
            for i in range(n)]


WORKLOADS = [
    ("W1-single",       _serial(1,  64,  16),          1, 1, 100.0, 10.0),
    ("W2-serial",       _serial(8,  64,  16, 1e-4),    1, 1, 100.0, 10.0),
    ("W3-concurrent",   _serial(32, 128, 64),          2, 2, 100.0, 10.0),
    ("W4-long-decode",  _serial(8,  32,  256),         1, 1, 100.0, 10.0),
    ("W5-long-prefill", _serial(8,  1024, 16),         1, 1, 100.0, 10.0),
    ("W6-kv-bw-stress", _serial(16, 512, 32),          2, 2,  10.0, 10.0),
]


def _us(x):
    return "  -  " if x is None else f"{x*1e6:8.2f}"


def trace_one(name, reqs, pr, dr, bw, lat):
    cfg = DisaggConfig(
        enabled=True, backend="single_process",
        prefill=RolePredictorConfig(device_name="H20", tp_size=1, replicas=pr,
                                    max_running_per_replica=64),
        decode=RolePredictorConfig(device_name="H20", tp_size=1, replicas=dr,
                                   max_running_per_replica=64),
        kv_transfer=BandwidthTransferConfig(bw_gbps=bw, latency_us=lat),
    )
    backend = build_pd_backend(
        model=_model(), base_sched_config=_sched(), disagg_config=cfg,
        predictor_factory=_AnalyticPredictor, role_factory=_role_factory,
        hw_factory=_hw,
    )
    start_pd_backend(backend)
    try:
        result = run_workload(backend, reqs, backend_label="A")
    finally:
        shutdown_pd_backend(backend)

    print(f"\n=== {name}  (P={pr} D={dr}  bw={bw}Gbps  lat={lat}us  reqs={len(reqs)}) ===")
    rows = result.per_request
    show = rows if len(rows) <= 8 else rows[:4] + rows[-2:]
    print("rid    arrive  pf_start  pf_end  kv_ready  dc_start  dc_end  | "
          "pq_wait kv_xfer dq_wait  ttft     e2e   (all us)")
    for r in show:
        print(f"{r.rid:<6}"
              f"{_us(r.arrival_time)} {_us(r.prefill_start_time)} "
              f"{_us(r.prefill_end_time)} {_us(r.kv_ready_time)} "
              f"{_us(r.decode_start_time)} {_us(r.decode_end_time)} | "
              f"{_us(r.prefill_queue_wait)} {_us(r.kv_transfer_time)} "
              f"{_us(r.decode_queue_wait)} {_us(r.ttft)} {_us(r.e2e)}")
    if len(rows) > 8:
        print(f"   ... ({len(rows)-6} rows hidden)")

    # aggregate
    e2es = [r.e2e for r in rows if r.e2e is not None]
    pq   = [r.prefill_queue_wait for r in rows if r.prefill_queue_wait is not None]
    kv   = [r.kv_transfer_time for r in rows if r.kv_transfer_time is not None]
    dq   = [r.decode_queue_wait for r in rows if r.decode_queue_wait is not None]
    print(f"-- final_clock = {result.final_clock*1e6:.2f} us")
    print(f"-- mean   pq={sum(pq)/len(pq):8.2f}us  kv={sum(kv)/len(kv):8.2f}us  "
          f"dq={sum(dq)/len(dq):8.2f}us  e2e={sum(e2es)/len(e2es):8.2f}us")
    print(f"-- max    pq={max(pq):8.2f}us  kv={max(kv):8.2f}us  "
          f"dq={max(dq):8.2f}us  e2e={max(e2es):8.2f}us")


if __name__ == "__main__":
    for w in WORKLOADS:
        trace_one(*w)
