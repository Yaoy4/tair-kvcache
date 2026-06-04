"""Real-hardware W1..W6 trace: AIC perf-db on H100/H200 with a real model.

Uses AIConfiguratorTimePredictor (via AICPredictorAdapter) for both prefill
and decode roles. Numbers are then in real microseconds derived from
silicon-measured per-op latencies.

Run:
    PYTHONPATH=hisim/src:/home/cjia/projects/aiconfigurator/src \
        venv/bin/python hisim/tools/trace_w1_w6_real_hw.py \
        --device h100_sxm --backend-version 0.5.10 \
        --model Qwen/Qwen3-8B \
        --out hisim/output/w1_w6_real_h100.html
"""
from __future__ import annotations

import argparse
import html
import os
import time
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


# ---------------------------------------------------------------------------
# AIC role factory: builds adapter wrapping AIConfiguratorTimePredictor.
# ---------------------------------------------------------------------------


def _aic_predictor_factory(model, hw=None, config=None, **kwargs):
    base = AIConfiguratorTimePredictor(model, hw=hw, config=config, **kwargs)
    return AICPredictorAdapter(base)


def _aic_role_factory(model, base_sched_config, role):
    # picklable-ish — used only by BackendA in this script.
    class _F:
        def __call__(self):
            return _aic_predictor_factory(model)
    return _F()


def _hw_factory(name):
    """Resolve an AcceleratorInfo, falling back to H20 profile with renamed
    name if the device isn't registered (mirrors pd_factory behavior)."""
    import copy as _copy
    hw = AcceleratorInfo.find_by_hw_name(name)
    if hw is not None:
        return hw
    fallback = AcceleratorInfo.find_by_hw_name("H20")
    cloned = _copy.deepcopy(fallback)
    cloned.name = name
    return cloned


# ---------------------------------------------------------------------------
# Workload matrix (same shapes as the analytic W1..W6).
# ---------------------------------------------------------------------------


def _serial(n, input_len, output_len, interarrival=0.0):
    return [WorkloadRequest(rid=f"r{i}", arrival_time=i * interarrival,
                            input_length=input_len, output_length=output_len)
            for i in range(n)]


@dataclass
class WL:
    name: str
    reqs: list
    pr: int
    dr: int
    bw: float
    lat: float
    isl: int
    osl: int
    purpose: str
    bottleneck: str


WORKLOADS: List[WL] = [
    WL("W1-single",       _serial(1,  64,  16),       1, 1, 100.0, 10.0, 64,  16,
       "Single request — full lifecycle baseline", "—"),
    WL("W2-serial",       _serial(8,  64,  16, 1e-4), 1, 1, 100.0, 10.0, 64,  16,
       "Serial small arrivals (100µs inter-arrival)", "decode_run (serial)"),
    WL("W3-concurrent",   _serial(32, 128, 64),       2, 2, 100.0, 10.0, 128, 64,
       "32 concurrent — earliest-replica scheduling", "prefill_queue"),
    WL("W4-long-decode",  _serial(8,  32,  256),      1, 1, 100.0, 10.0, 32,  256,
       "Decode-bound (long output, short input)", "decode_run"),
    WL("W5-long-prefill", _serial(8,  1024, 16),      1, 1, 100.0, 10.0, 1024, 16,
       "Prefill-bound (long input, short output)", "prefill_queue"),
    WL("W6-kv-bw-stress", _serial(16, 512, 32),       2, 2,  10.0, 10.0, 512,  32,
       "KV bandwidth bottleneck (bw=10Gbps)", "kv_transfer"),
]


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def make_sched(backend_name: str, backend_version: str) -> SchedulerConfig:
    return SchedulerConfig(
        model=None,  # filled per-role
        tp_size=1, pp_size=1,
        data_type=DataType.FP16, kv_cache_data_type=DataType.FP16,
        backend_name=backend_name, backend_version=backend_version,
    )


def run_one(wl: WL, model: ModelInfo, base_sched: SchedulerConfig,
            device: str, tp: int) -> tuple:
    cfg = DisaggConfig(
        enabled=True, backend="single_process",
        prefill=RolePredictorConfig(device_name=device, tp_size=tp,
                                    replicas=wl.pr, max_running_per_replica=64),
        decode=RolePredictorConfig(device_name=device, tp_size=tp,
                                   replicas=wl.dr, max_running_per_replica=64),
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
# HTML rendering
# ---------------------------------------------------------------------------


CSS = """
<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1400px;
     margin:30px auto;padding:0 20px;color:#1f2328;background:#fff;}
h1{border-bottom:2px solid #d0d7de;padding-bottom:8px;}
h2{margin-top:36px;padding:8px 12px;background:#f6f8fa;border-left:4px solid #0969da;}
.meta{color:#57606a;font-size:.9em;margin-top:-8px;}
.purpose{background:#ddf4ff;padding:8px 12px;border-radius:6px;margin:12px 0;}
.bottleneck{display:inline-block;padding:2px 8px;border-radius:12px;
  background:#fff8c5;color:#633c01;font-weight:600;font-size:.85em;}
.config{background:#fff8c5;padding:10px 14px;border-radius:6px;margin:12px 0;
  font-family:ui-monospace,Menlo,monospace;font-size:.85em;}
table{border-collapse:collapse;width:100%;font-family:ui-monospace,Menlo,monospace;
  font-size:.82em;margin-top:10px;}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:right;}
th{background:#f6f8fa;font-weight:600;}
th.l,td.l{text-align:left;}
tr:nth-child(even){background:#fafbfc;}
td.hi{background:#ffebe9;font-weight:600;}
td.med{background:#fff8c5;}
.summary{background:#f6f8fa;padding:12px;border-radius:6px;margin-top:12px;font-size:.9em;}
.summary code{background:#eaeef2;padding:1px 6px;border-radius:3px;}
.overview td.hi{font-weight:700;}
.legend{font-size:.85em;color:#57606a;margin-top:8px;}
.compare{background:#dafbe1;padding:10px 14px;border-radius:6px;margin:12px 0;
  font-size:.9em;}
</style>
"""


def fmt(x):
    return "&nbsp;–" if x is None else f"{x*1e6:,.2f}"


def fmt_ms(x):
    return "&nbsp;–" if x is None else f"{x*1e3:,.3f}"


def render_per_request_table(result):
    rows = result.per_request
    headers = ["rid", "arrive", "pf_start", "pf_end", "kv_ready", "dc_start",
               "dc_end", "pq_wait", "kv_xfer", "dq_wait", "pf_run", "dc_run",
               "ttft", "e2e"]
    head = "".join(f"<th class='l'>{h}</th>" if h == "rid" else f"<th>{h}</th>"
                   for h in headers)
    pq_max = max((r.prefill_queue_wait or 0) for r in rows)
    kv_max = max((r.kv_transfer_time or 0) for r in rows)
    dq_max = max((r.decode_queue_wait or 0) for r in rows)
    e2e_max = max((r.e2e or 0) for r in rows)
    body = []
    for r in rows:
        pf_run = ((r.prefill_end_time - r.prefill_start_time)
                  if r.prefill_end_time and r.prefill_start_time is not None else None)
        dc_run = ((r.decode_end_time - r.decode_start_time)
                  if r.decode_end_time and r.decode_start_time is not None else None)
        body.append(
            "<tr>"
            f"<td class='l'>{r.rid}</td>"
            f"<td>{fmt(r.arrival_time)}</td><td>{fmt(r.prefill_start_time)}</td>"
            f"<td>{fmt(r.prefill_end_time)}</td><td>{fmt(r.kv_ready_time)}</td>"
            f"<td>{fmt(r.decode_start_time)}</td><td>{fmt(r.decode_end_time)}</td>"
            f"<td class='{'hi' if (r.prefill_queue_wait or 0) == pq_max and pq_max>0 else ''}'>{fmt(r.prefill_queue_wait)}</td>"
            f"<td class='{'hi' if (r.kv_transfer_time or 0) == kv_max and kv_max>0 else ''}'>{fmt(r.kv_transfer_time)}</td>"
            f"<td class='{'hi' if (r.decode_queue_wait or 0) == dq_max and dq_max>0 else ''}'>{fmt(r.decode_queue_wait)}</td>"
            f"<td>{fmt(pf_run)}</td><td>{fmt(dc_run)}</td>"
            f"<td>{fmt(r.ttft)}</td>"
            f"<td class='{'hi' if (r.e2e or 0) == e2e_max else ''}'>{fmt(r.e2e)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_summary(wl: WL, result, wall_s: float):
    rows = result.per_request
    pq = [r.prefill_queue_wait for r in rows if r.prefill_queue_wait is not None]
    kv = [r.kv_transfer_time for r in rows if r.kv_transfer_time is not None]
    dq = [r.decode_queue_wait for r in rows if r.decode_queue_wait is not None]
    e2e = [r.e2e for r in rows if r.e2e is not None]
    pf_run = [(r.prefill_end_time - r.prefill_start_time) for r in rows
              if r.prefill_end_time and r.prefill_start_time is not None]
    dc_run = [(r.decode_end_time - r.decode_start_time) for r in rows
              if r.decode_end_time and r.decode_start_time is not None]

    def stat(label, xs):
        if not xs:
            return f"<tr><td class='l'>{label}</td><td colspan='3'>–</td></tr>"
        return (f"<tr><td class='l'>{label}</td>"
                f"<td>{min(xs)*1e6:,.2f}</td>"
                f"<td>{sum(xs)/len(xs)*1e6:,.2f}</td>"
                f"<td>{max(xs)*1e6:,.2f}</td></tr>")

    return (
        f"<div class='summary'>"
        f"<b>final_clock</b> = <code>{result.final_clock*1e6:,.2f} µs</code> "
        f"&nbsp;|&nbsp; <b>n_reqs</b> = {len(rows)} "
        f"&nbsp;|&nbsp; <b>sim wall time</b> = <code>{wall_s*1000:.1f} ms</code> "
        f"&nbsp;|&nbsp; <b>bottleneck</b> "
        f"<span class='bottleneck'>{wl.bottleneck}</span>"
        f"<table style='margin-top:8px'>"
        f"<thead><tr><th class='l'>stage (µs)</th><th>min</th><th>mean</th><th>max</th></tr></thead>"
        f"<tbody>{stat('prefill_queue_wait',pq)}{stat('prefill_run',pf_run)}"
        f"{stat('kv_transfer',kv)}{stat('decode_queue_wait',dq)}"
        f"{stat('decode_run',dc_run)}{stat('e2e',e2e)}</tbody></table></div>"
    )


def render_overview(rows):
    head = ("<tr><th class='l'>Workload</th><th>reqs</th><th>P/D</th>"
            "<th>isl</th><th>osl</th><th>bw</th><th>max pf_run</th>"
            "<th>max kv</th><th>max pq</th><th>max e2e (ms)</th>"
            "<th>final_clock (ms)</th><th class='l'>bottleneck</th></tr>")
    body = []
    for wl, res, _wall in rows:
        pf = max((r.prefill_end_time - r.prefill_start_time) for r in res.per_request)
        kv = max(r.kv_transfer_time for r in res.per_request)
        pq = max(r.prefill_queue_wait for r in res.per_request)
        e2e = max(r.e2e for r in res.per_request)
        body.append(
            f"<tr><td class='l'>{wl.name}</td><td>{len(res.per_request)}</td>"
            f"<td>{wl.pr}/{wl.dr}</td><td>{wl.isl}</td><td>{wl.osl}</td>"
            f"<td>{wl.bw:g}</td>"
            f"<td>{pf*1e6:,.1f} µs</td>"
            f"<td class='med'>{kv*1e6:,.1f} µs</td>"
            f"<td class='med'>{pq*1e6:,.1f} µs</td>"
            f"<td class='hi'>{e2e*1e3:,.3f}</td>"
            f"<td>{res.final_clock*1e3:,.3f}</td>"
            f"<td class='l'><span class='bottleneck'>{wl.bottleneck}</span></td></tr>"
        )
    return "<table>" + head + "<tbody>" + "".join(body) + "</tbody></table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="h100_sxm",
                    help="AIC device key (e.g. h100_sxm, h200_sxm, b200_sxm)")
    ap.add_argument("--backend-name", default="sglang",
                    help="AIC backend name (sglang, vllm, trtllm)")
    ap.add_argument("--backend-version", default="0.5.10",
                    help="Backend version key under aiconfigurator perf-db")
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HuggingFace model id (used for ModelInfo)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"loading model info: {args.model} ...")
    model = ModelInfo.from_huggingface_id(args.model, timeout=10)
    if model is None:
        raise SystemExit(
            f"could not fetch HF config for {args.model}; "
            f"set HF_ENDPOINT or use a cached model.")

    base_sched = make_sched(args.backend_name, args.backend_version)

    print(f"running 6 workloads on {args.device} / {args.backend_name} {args.backend_version} "
          f"/ {args.model} / tp={args.tp} ...")
    runs = []
    for wl in WORKLOADS:
        print(f"  -> {wl.name}", flush=True)
        result, wall = run_one(wl, model, base_sched, args.device, args.tp)
        runs.append((wl, result, wall))

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>HiSim W1..W6 Real-HW ({args.device})</title>",
             CSS, "</head><body>"]
    parts.append(f"<h1>HiSim BackendA — W1..W6 on real AIC perf-db</h1>")
    parts.append(
        f"<div class='config'>"
        f"<b>device</b> = {args.device}  &nbsp; "
        f"<b>backend</b> = {args.backend_name} {args.backend_version}  &nbsp; "
        f"<b>model</b> = {args.model}  &nbsp; "
        f"<b>tp</b> = {args.tp}  &nbsp; "
        f"<b>predictor</b> = AIConfiguratorTimePredictor (SILICON mode)"
        f"</div>"
    )
    parts.append(
        "<p class='meta'>Latencies come from per-op silicon-measured AIC perf-db. "
        "KV transfer is analytic <code>latency_us + bytes/bw</code>. "
        "All times in microseconds unless noted as ms.</p>"
    )
    parts.append("<h2>Overview — max per stage across W1..W6</h2>")
    parts.append(render_overview(runs))
    parts.append(
        "<div class='compare'>"
        "<b>Compare to the analytic run</b>: the analytic predictor used "
        "<code>0.5 µs/tok</code> prefill + <code>10 µs/step</code> decode. "
        f"Real {args.device} numbers should be ~2-3 orders of magnitude larger and "
        "scale non-linearly with isl (attention is quadratic) and batch "
        "(GEMM regimes change). Use this report to verify the bottleneck "
        "story (W4=decode, W5=prefill, W6=KV) still holds on real silicon."
        "</div>"
    )

    for wl, result, wall in runs:
        parts.append(f"<h2>{html.escape(wl.name)} "
                     f"<span class='meta'>(P={wl.pr}, D={wl.dr}, isl={wl.isl}, "
                     f"osl={wl.osl}, bw={wl.bw}Gbps, lat={wl.lat}µs, "
                     f"reqs={len(wl.reqs)})</span></h2>")
        parts.append(f"<div class='purpose'><b>Purpose:</b> "
                     f"{html.escape(wl.purpose)}</div>")
        parts.append(render_summary(wl, result, wall))
        parts.append("<details><summary>Per-request timestamps "
                     "(red = max in column)</summary>")
        parts.append(render_per_request_table(result))
        parts.append("</details>")

    parts.append("</body></html>")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
