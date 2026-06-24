"""PD topology robustness sweep on B60.

Runs a fixed workload (W3-concurrent by default) under multiple PD parallel
topologies to verify heterogeneous prefill/decode configurations work end-to-end.

Each topology specifies prefill-role TP/DP/PP/replicas and decode-role
TP/DP/PP/replicas independently. AIC perf-db is queried per-role with the
role's TP, so heterogeneous configs (e.g. prefill TP=4, decode TP=1) are honored.

Run:
    PYTHONPATH=hisim/src:/home/cjia/projects/aiconfigurator/src \
        SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        venv/bin/python hisim/tools/trace_pd_topology_b60.py \
        --device b60 --backend-name vllm --backend-version 0.12.0 \
        --model Qwen/Qwen3-8B \
        --out hisim/output/pd_topology_b60.html
"""
from __future__ import annotations

import argparse
import copy
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

from hisim.spec import AcceleratorInfo, DataType, ModelInfo
from hisim.simulation.pd_ab_harness import WorkloadRequest, run_workload
from hisim.simulation.pd_aic_adapter import AICPredictorAdapter
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_runtime import (
    build_pd_backend,
    shutdown_pd_backend,
    start_pd_backend,
)
from hisim.simulation.types import SchedulerConfig
from hisim.time_predictor import AIConfiguratorTimePredictor


def _aic_predictor_factory(model, hw=None, config=None, **kwargs):
    return AICPredictorAdapter(
        AIConfiguratorTimePredictor(model, hw=hw, config=config, **kwargs)
    )


def _aic_role_factory(model, base_sched_config, role):
    class _F:
        def __call__(self):
            return _aic_predictor_factory(model)
    return _F()


def _hw_factory(name):
    hw = AcceleratorInfo.find_by_hw_name(name)
    if hw is not None:
        return hw
    fallback = AcceleratorInfo.find_by_hw_name("H20")
    cloned = copy.deepcopy(fallback)
    cloned.name = name
    return cloned


# ---------------------------------------------------------------------------
# Workload (W3-concurrent: 32 reqs, isl=128, osl=64) by default
# ---------------------------------------------------------------------------


@dataclass
class WL:
    name: str
    reqs: list
    bw: float = 100.0
    lat: float = 10.0


def _workload(name: str, n: int, isl: int, osl: int, interarrival: float = 0.0) -> WL:
    reqs = [WorkloadRequest(rid=f"r{i}", arrival_time=i * interarrival,
                            input_length=isl, output_length=osl)
            for i in range(n)]
    return WL(name=name, reqs=reqs)


WORKLOADS = {
    "concurrent":  _workload("concurrent",   32, 128, 64),
    "long_prefill": _workload("long_prefill", 8, 1024, 16),
    "long_decode":  _workload("long_decode",  8, 32, 256),
}


# ---------------------------------------------------------------------------
# Topology matrix
# ---------------------------------------------------------------------------


@dataclass
class Topo:
    name: str
    purpose: str
    # prefill role
    p_tp: int = 1
    p_dp: int = 1
    p_pp: int = 1
    p_replicas: int = 1
    # decode role
    d_tp: int = 1
    d_dp: int = 1
    d_pp: int = 1
    d_replicas: int = 1

    @property
    def total_gpus(self) -> int:
        # heuristic count, not used by simulator
        p = self.p_tp * self.p_dp * self.p_pp * self.p_replicas
        d = self.d_tp * self.d_dp * self.d_pp * self.d_replicas
        return p + d

    def label(self) -> str:
        return (f"P[tp{self.p_tp}·dp{self.p_dp}·pp{self.p_pp}×{self.p_replicas}] "
                f"D[tp{self.d_tp}·dp{self.d_dp}·pp{self.d_pp}×{self.d_replicas}]")


TOPOLOGIES = [
    Topo("C1_baseline",       "Baseline: TP=1, single replica each",
         p_tp=1, p_replicas=1,  d_tp=1, d_replicas=1),
    Topo("C2_symmetric_tp2",  "Symmetric TP=2 on both roles",
         p_tp=2, p_replicas=1,  d_tp=2, d_replicas=1),
    Topo("C3_heavy_prefill",  "Heavy prefill: P-TP=4, D-TP=1",
         p_tp=4, p_replicas=1,  d_tp=1, d_replicas=1),
    Topo("C4_prefill_dp_decode_tp",
         "Prefill scales out (DP=4), Decode scales up (TP=4)",
         p_tp=1, p_replicas=4,  d_tp=4, d_replicas=1),
    Topo("C5_pp_smoke",       "PP=2 smoke test (path coverage)",
         p_tp=1, p_pp=2,        d_tp=1, d_pp=2),
    Topo("C6_imbalanced",     "Imbalanced: P[tp2×2] vs D[tp1×4]",
         p_tp=2, p_replicas=2,  d_tp=1, d_replicas=4),
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def make_sched(backend_name: str, backend_version: str) -> SchedulerConfig:
    return SchedulerConfig(
        model=None,
        tp_size=1, pp_size=1,
        data_type=DataType.FP16, kv_cache_data_type=DataType.FP16,
        backend_name=backend_name, backend_version=backend_version,
    )


def run_one(topo: Topo, wl: WL, model: ModelInfo, base_sched: SchedulerConfig,
            device: str) -> tuple:
    cfg = DisaggConfig(
        enabled=True, backend="single_process",
        prefill=RolePredictorConfig(
            device_name=device,
            tp_size=topo.p_tp, dp_size=topo.p_dp, pp_size=topo.p_pp,
            replicas=topo.p_replicas, max_running_per_replica=64,
        ),
        decode=RolePredictorConfig(
            device_name=device,
            tp_size=topo.d_tp, dp_size=topo.d_dp, pp_size=topo.d_pp,
            replicas=topo.d_replicas, max_running_per_replica=64,
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=wl.bw, latency_us=wl.lat),
    )
    backend = build_pd_backend(
        model=model, base_sched_config=base_sched, disagg_config=cfg,
        predictor_factory=_aic_predictor_factory, role_factory=_aic_role_factory,
        hw_factory=_hw_factory,
    )
    start_pd_backend(backend)
    t0 = time.perf_counter()
    try:
        result = run_workload(backend, wl.reqs, backend_label="A-real")
    finally:
        shutdown_pd_backend(backend)
    wall = time.perf_counter() - t0
    return result, wall


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


CSS = """
<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1400px;
     margin:30px auto;padding:0 20px;color:#1f2328;background:#fff;}
h1{border-bottom:2px solid #d0d7de;padding-bottom:8px;}
h2{margin-top:36px;padding:8px 12px;background:#f6f8fa;border-left:4px solid #0969da;}
table{border-collapse:collapse;width:100%;font-family:ui-monospace,Menlo,monospace;
  font-size:.85em;margin-top:10px;}
th,td{border:1px solid #d0d7de;padding:5px 9px;text-align:right;}
th{background:#f6f8fa;font-weight:600;}
th.l,td.l{text-align:left;}
tr:nth-child(even){background:#fafbfc;}
td.ok{background:#dafbe1;}
td.fail{background:#ffebe9;color:#82071e;font-weight:600;}
td.hi{background:#ffebe9;font-weight:600;}
td.med{background:#fff8c5;}
.config{background:#fff8c5;padding:10px 14px;border-radius:6px;margin:12px 0;
  font-family:ui-monospace,Menlo,monospace;font-size:.85em;}
.purpose{background:#ddf4ff;padding:6px 10px;border-radius:6px;margin:6px 0;font-size:.9em;}
.note{color:#57606a;font-size:.85em;margin-top:6px;}
pre.err{background:#ffebe9;color:#82071e;padding:8px;border-radius:6px;
  font-size:.78em;white-space:pre-wrap;}
</style>
"""


def fmt_ms(x):
    return "–" if x is None else f"{x*1e3:,.2f}"


def fmt_us(x):
    return "–" if x is None else f"{x*1e6:,.1f}"


def render_overview(rows, wl_name):
    head = ("<tr><th class='l'>Topology</th><th class='l'>Prefill</th>"
            "<th class='l'>Decode</th><th>GPUs</th><th>status</th>"
            "<th>max pf_run</th><th>max kv</th><th>max pq</th>"
            "<th>max e2e (ms)</th><th>final_clock (ms)</th>"
            "<th>sim wall (ms)</th></tr>")
    body = []
    for topo, res, wall, err in rows:
        if err is not None:
            body.append(
                f"<tr><td class='l'>{topo.name}</td>"
                f"<td class='l'>tp{topo.p_tp}·dp{topo.p_dp}·pp{topo.p_pp}×{topo.p_replicas}</td>"
                f"<td class='l'>tp{topo.d_tp}·dp{topo.d_dp}·pp{topo.d_pp}×{topo.d_replicas}</td>"
                f"<td>{topo.total_gpus}</td>"
                f"<td class='fail'>FAILED</td>"
                f"<td colspan='6' class='fail'>{type(err).__name__}: {str(err)[:120]}</td></tr>"
            )
            continue
        pf = max((r.prefill_end_time - r.prefill_start_time) for r in res.per_request
                 if r.prefill_end_time and r.prefill_start_time is not None)
        kv = max((r.kv_transfer_time or 0) for r in res.per_request)
        pq = max((r.prefill_queue_wait or 0) for r in res.per_request)
        e2e = max((r.e2e or 0) for r in res.per_request)
        body.append(
            f"<tr><td class='l'>{topo.name}</td>"
            f"<td class='l'>tp{topo.p_tp}·dp{topo.p_dp}·pp{topo.p_pp}×{topo.p_replicas}</td>"
            f"<td class='l'>tp{topo.d_tp}·dp{topo.d_dp}·pp{topo.d_pp}×{topo.d_replicas}</td>"
            f"<td>{topo.total_gpus}</td>"
            f"<td class='ok'>OK</td>"
            f"<td>{fmt_us(pf)}</td>"
            f"<td class='med'>{fmt_us(kv)}</td>"
            f"<td class='med'>{fmt_us(pq)}</td>"
            f"<td class='hi'>{fmt_ms(e2e)}</td>"
            f"<td>{fmt_ms(res.final_clock)}</td>"
            f"<td>{wall*1000:,.0f}</td></tr>"
        )
    return (f"<h2>Topology sweep on workload: <code>{wl_name}</code></h2>"
            f"<table>{head}<tbody>{''.join(body)}</tbody></table>")


def render_topo_purposes():
    rows = []
    for t in TOPOLOGIES:
        rows.append(
            f"<tr><td class='l'>{t.name}</td>"
            f"<td class='l'>{t.purpose}</td>"
            f"<td class='l'>{t.label()}</td></tr>"
        )
    return ("<h2>Topology matrix</h2><table>"
            "<tr><th class='l'>name</th><th class='l'>purpose</th>"
            "<th class='l'>config</th></tr>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="b60")
    ap.add_argument("--backend-name", default="vllm")
    ap.add_argument("--backend-version", default="0.12.0")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--workload", default="concurrent",
                    choices=list(WORKLOADS.keys()),
                    help="which workload to sweep (default: concurrent)")
    ap.add_argument("--all-workloads", action="store_true",
                    help="sweep all 3 workloads (concurrent/long_prefill/long_decode)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"loading model: {args.model} ...")
    model = ModelInfo.from_huggingface_id(args.model, timeout=10)
    if model is None:
        raise SystemExit(f"could not fetch HF config for {args.model}")

    base_sched = make_sched(args.backend_name, args.backend_version)

    wl_names = list(WORKLOADS.keys()) if args.all_workloads else [args.workload]

    all_results = {}  # wl_name -> [(topo, result, wall, err)]
    for wl_name in wl_names:
        wl = WORKLOADS[wl_name]
        print(f"\n=== workload: {wl_name} ({len(wl.reqs)} reqs) ===")
        rows = []
        for topo in TOPOLOGIES:
            print(f"  -> {topo.name:25s} {topo.label()}", flush=True)
            try:
                result, wall = run_one(topo, wl, model, base_sched, args.device)
                rows.append((topo, result, wall, None))
                e2e_max = max((r.e2e or 0) for r in result.per_request)
                print(f"     OK   e2e_max={e2e_max*1e3:.1f}ms  "
                      f"final_clock={result.final_clock*1e3:.1f}ms  "
                      f"sim_wall={wall*1000:.0f}ms")
            except Exception as e:
                rows.append((topo, None, 0.0, e))
                print(f"     FAIL {type(e).__name__}: {e}")
                traceback.print_exc()
        all_results[wl_name] = rows

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>HiSim PD Topology Sweep ({args.device})</title>",
             CSS, "</head><body>"]
    parts.append("<h1>HiSim PD Topology Robustness — heterogeneous parallel configs</h1>")
    parts.append(
        f"<div class='config'>"
        f"<b>device</b> = {args.device}  &nbsp; "
        f"<b>backend</b> = {args.backend_name} {args.backend_version}  &nbsp; "
        f"<b>model</b> = {args.model}  &nbsp; "
        f"<b>predictor</b> = AIConfiguratorTimePredictor (SILICON)"
        f"</div>"
    )
    parts.append(
        "<p class='note'>Each topology specifies prefill role TP/DP/PP/replicas and "
        "decode role TP/DP/PP/replicas independently. AIC perf-db is queried per-role "
        "with the role's TP, so heterogeneous configs are honored. "
        "<b>OK</b> means the simulator completed all requests; latencies reflect "
        "perf-db lookups (with possible nearest-neighbor interpolation for "
        "shapes outside the B60 vllm 0.12.0 coverage).</p>"
    )
    parts.append(render_topo_purposes())
    for wl_name in wl_names:
        parts.append(render_overview(all_results[wl_name], wl_name))
    parts.append("</body></html>")

    with open(args.out, "w") as f:
        f.write("".join(parts))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
