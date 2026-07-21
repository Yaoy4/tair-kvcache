"""HiSim-only PD topology + load sweep on RTX PRO 6000 (no real hardware).

ISL/OSL/concurrency conventions follow accuracy_validation_report.md Hisim
setup (closed-loop concurrency, request count = concurrency*10, ISL/OSL grid
subset of {512,1024,2048}x{512,1024}). Topology matrix isolates DP-only
(replicas), TP-only, and PP-only scaling at a fixed representative load point.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_ab_harness import WorkloadRequest, run_workload
from hisim.simulation.pd_aic_adapter import AICPredictorAdapter
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_runtime import build_pd_backend, shutdown_pd_backend, start_pd_backend
from hisim.simulation.pd_factory import _default_hw_factory
from hisim.simulation.types import SchedulerConfig
from hisim.time_predictor import AIConfiguratorTimePredictor

DEVICE = "rtx_pro_6000_server"
DATABASE_PATH = "/tmp/aic_custom_systems"
BACKEND_NAME = "sglang"
BACKEND_VERSION = "0.5.10"
PREFILL_OVERHEAD_MS = 12.0
DECODE_OVERHEAD_MS = 0.2
KV_BW_GBPS = 512.0
KV_LATENCY_US = 10.0
MAX_RUNNING_PER_REPLICA = 64


def _aic_predictor_factory(model, hw=None, config=None, **kwargs):
    return AICPredictorAdapter(AIConfiguratorTimePredictor(model, hw=hw, config=config, **kwargs))


def _aic_role_factory(model, base_sched_config, role):
    class _F:
        def __call__(self):
            return _aic_predictor_factory(model)
    return _F()


def _hw_factory(name):
    return _default_hw_factory(name)


def make_base_sched() -> SchedulerConfig:
    return SchedulerConfig(
        model=None, tp_size=1, pp_size=1,
        data_type=DataType.BF16, kv_cache_data_type=DataType.BF16,
        backend_name=BACKEND_NAME, backend_version=BACKEND_VERSION,
    )


def make_role(tp=1, dp=1, pp=1, replicas=1) -> RolePredictorConfig:
    return RolePredictorConfig(
        device_name=DEVICE, tp_size=tp, dp_size=dp, pp_size=pp, replicas=replicas,
        max_running_per_replica=MAX_RUNNING_PER_REPLICA,
        database_path=DATABASE_PATH,
        data_type="bf16", kv_cache_data_type="bf16",
        backend_version=BACKEND_VERSION,
        prefill_overhead_ms=PREFILL_OVERHEAD_MS, decode_overhead_ms=DECODE_OVERHEAD_MS,
    )


def make_workload(n_base, isl, osl, interarrival=1e-4):
    return [WorkloadRequest(rid="r" + str(i), arrival_time=i * interarrival,
                             input_length=isl, output_length=osl)
            for i in range(n_base)]


def run_cell(model, base_sched, tp=1, dp=1, pp=1, replicas=1, isl=1024, osl=512, concurrency=8):
    cfg = DisaggConfig(
        enabled=True, backend="single_process",
        prefill=make_role(tp=tp, dp=dp, pp=pp, replicas=replicas),
        decode=make_role(tp=tp, dp=dp, pp=pp, replicas=replicas),
        kv_transfer=BandwidthTransferConfig(bw_gbps=KV_BW_GBPS, latency_us=KV_LATENCY_US),
    )
    backend = build_pd_backend(
        model=model, base_sched_config=base_sched, disagg_config=cfg,
        predictor_factory=_aic_predictor_factory, role_factory=_aic_role_factory,
        hw_factory=_hw_factory,
    )
    start_pd_backend(backend)
    # Hisim predictions are deterministic (no measurement noise like real GPU
    # runs), so unlike accuracy_validation_report.md's concurrency*10 sampling
    # (needed to average out real-hardware jitter), one wave of `concurrency`
    # requests is enough for a stable mean here.
    n = max(concurrency, 1)
    reqs = make_workload(n, isl, osl)
    t0 = time.perf_counter()
    try:
        # aiconfigurator's operations.py does a bare print() per GEMM query;
        # at OSL>=512 x concurrency this floods millions of lines. Silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_workload(backend, reqs, backend_label="A-rtx6000")
    finally:
        shutdown_pd_backend(backend)
    wall_ms = (time.perf_counter() - t0) * 1000

    finished = [r for r in result.per_request if r.finished]
    ttfts = [r.ttft for r in finished if r.ttft is not None]
    e2es = [r.e2e for r in finished if r.e2e is not None]
    tpots = []
    for r in finished:
        if r.decode_start_time is not None and r.decode_end_time is not None and r.decode_step_count > 1:
            tpots.append((r.decode_end_time - r.decode_start_time) / (r.decode_step_count - 1))
    total_out_tokens = sum(r.output_length for r in finished)
    throughput = total_out_tokens / result.final_clock if result.final_clock > 0 else 0.0

    def pctl(xs, p):
        if not xs:
            return None
        xs2 = sorted(xs)
        idx = min(len(xs2) - 1, int(len(xs2) * p))
        return xs2[idx]

    return {
        "n_requests": n, "n_finished": len(finished),
        "isl": isl, "osl": osl, "concurrency": concurrency,
        "tp": tp, "dp": dp, "pp": pp, "replicas": replicas,
        "ttft_mean_ms": (sum(ttfts) / len(ttfts) * 1e3) if ttfts else None,
        "ttft_p50_ms": (pctl(ttfts, 0.5) * 1e3) if ttfts else None,
        "ttft_p90_ms": (pctl(ttfts, 0.9) * 1e3) if ttfts else None,
        "tpot_mean_ms": (sum(tpots) / len(tpots) * 1e3) if tpots else None,
        "e2e_mean_ms": (sum(e2es) / len(e2es) * 1e3) if e2es else None,
        "throughput_tok_s": throughput,
        "final_clock_ms": result.final_clock * 1e3,
        "sim_wall_ms": wall_ms,
    }


def main():
    print("loading model: Qwen/Qwen3-8B ...")
    model = ModelInfo.from_json("/mnt/nfs02/users/tjiang/Gitrepo/models/Qwen3-8B/config.json")
    if model is None:
        raise SystemExit("could not fetch HF config for Qwen/Qwen3-8B")
    base_sched = make_base_sched()

    results = {"load_sweep": [], "topology_sweep": []}

    ISL_VALUES = [512, 1024, 2048]
    OSL_VALUES = [512, 1024]
    CONC_VALUES = [1, 8, 16]
    print("")
    print("=== Load sweep (baseline topology 1P1D, tp=pp=1, replicas=1) ===")
    for isl in ISL_VALUES:
        for osl in OSL_VALUES:
            for conc in CONC_VALUES:
                label = "isl" + str(isl) + "_osl" + str(osl) + "_conc" + str(conc)
                print("  -> " + label)
                try:
                    cell = run_cell(model, base_sched, isl=isl, osl=osl, concurrency=conc)
                    cell["label"] = label
                    results["load_sweep"].append(cell)
                    print("     TTFT=%.2fms TPOT=%.2fms E2E=%.2fms Tput=%.1ftok/s" % (
                        cell["ttft_mean_ms"], cell["tpot_mean_ms"], cell["e2e_mean_ms"], cell["throughput_tok_s"]))
                except Exception as e:
                    print("     FAIL " + type(e).__name__ + ": " + str(e))
                    results["load_sweep"].append({"label": label, "isl": isl, "osl": osl,
                                                   "concurrency": conc, "error": str(e)})

    FIXED_ISL, FIXED_OSL, FIXED_CONC = 1024, 512, 8
    print("")
    print("=== Topology sweep (fixed load isl=%d osl=%d conc=%d) ===" % (FIXED_ISL, FIXED_OSL, FIXED_CONC))

    topo_configs = [
        ("baseline_1P1D", dict(tp=1, dp=1, pp=1, replicas=1)),
        ("dp_only_r2", dict(tp=1, dp=1, pp=1, replicas=2)),
        ("dp_only_r4", dict(tp=1, dp=1, pp=1, replicas=4)),
        ("tp_only_2", dict(tp=2, dp=1, pp=1, replicas=1)),
        ("tp_only_4", dict(tp=4, dp=1, pp=1, replicas=1)),
        ("pp_only_2", dict(tp=1, dp=1, pp=2, replicas=1)),
    ]
    for name, kwargs in topo_configs:
        print("  -> " + name + " " + str(kwargs))
        try:
            cell = run_cell(model, base_sched, isl=FIXED_ISL, osl=FIXED_OSL, concurrency=FIXED_CONC, **kwargs)
            cell["label"] = name
            results["topology_sweep"].append(cell)
            print("     TTFT=%.2fms TPOT=%.2fms E2E=%.2fms Tput=%.1ftok/s" % (
                cell["ttft_mean_ms"], cell["tpot_mean_ms"], cell["e2e_mean_ms"], cell["throughput_tok_s"]))
        except Exception as e:
            print("     FAIL " + type(e).__name__ + ": " + str(e))
            results["topology_sweep"].append({"label": name, "error": str(e), **kwargs})

    with open("/tmp/pd_rtx6000_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("")
    print("wrote /tmp/pd_rtx6000_sweep_results.json")


if __name__ == "__main__":
    main()
