"""Generate an HTML report of W1..W6 BackendA traces.

Run:
    PYTHONPATH=hisim/src:../aiconfigurator/src \
        venv/bin/python hisim/tools/trace_w1_w6_html.py \
        --out hisim/output/w1_w6_report.html
"""
from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from typing import List

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_ab_harness import WorkloadRequest, run_workload
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


# ---- Predictor (matches Phase 6d analytic stub) ----

class _AnalyticPredictor:
    _PREFILL_US_PER_TOK = 0.5
    _DECODE_US_PER_STEP = 10.0

    def __init__(self, model=None, hw=None, config=None, **kwargs):
        pass

    def predict_prefill_seconds(self, batch_tokens):
        return batch_tokens * self._PREFILL_US_PER_TOK * 1e-6

    def predict_decode_seconds(self, batch_size, past_kv_length=0):
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
       "Single request — full lifecycle baseline",
       "—"),
    WL("W2-serial",       _serial(8,  64,  16, 1e-4), 1, 1, 100.0, 10.0, 64,  16,
       "Serial small arrivals (100µs inter-arrival)",
       "decode_run (serial)"),
    WL("W3-concurrent",   _serial(32, 128, 64),       2, 2, 100.0, 10.0, 128, 64,
       "32 concurrent arrivals — earliest-replica scheduling",
       "prefill_queue"),
    WL("W4-long-decode",  _serial(8,  32,  256),      1, 1, 100.0, 10.0, 32,  256,
       "Decode-bound (long output, short input)",
       "decode_run"),
    WL("W5-long-prefill", _serial(8,  1024, 16),      1, 1, 100.0, 10.0, 1024, 16,
       "Prefill-bound (long input, short output)",
       "prefill_queue"),
    WL("W6-kv-bw-stress", _serial(16, 512, 32),       2, 2,  10.0, 10.0, 512,  32,
       "KV bandwidth bottleneck (bw=10Gbps)",
       "kv_transfer"),
]


def run_one(wl: WL):
    cfg = DisaggConfig(
        enabled=True, backend="single_process",
        prefill=RolePredictorConfig(device_name="H20", tp_size=1,
                                    replicas=wl.pr, max_running_per_replica=64),
        decode=RolePredictorConfig(device_name="H20", tp_size=1,
                                   replicas=wl.dr, max_running_per_replica=64),
        kv_transfer=BandwidthTransferConfig(bw_gbps=wl.bw, latency_us=wl.lat),
    )
    backend = build_pd_backend(
        model=_model(), base_sched_config=_sched(), disagg_config=cfg,
        predictor_factory=_AnalyticPredictor, role_factory=_role_factory,
        hw_factory=_hw,
    )
    start_pd_backend(backend)
    try:
        return run_workload(backend, wl.reqs, backend_label="A")
    finally:
        shutdown_pd_backend(backend)


def fmt(x):
    return "&nbsp;–" if x is None else f"{x*1e6:,.2f}"


CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         max-width: 1300px; margin: 30px auto; padding: 0 20px;
         color: #1f2328; background: #fff; }
  h1 { border-bottom: 2px solid #d0d7de; padding-bottom: 8px; }
  h2 { margin-top: 36px; padding: 8px 12px; background: #f6f8fa;
       border-left: 4px solid #0969da; }
  .meta { color: #57606a; font-size: 0.9em; margin-top: -8px; }
  .purpose { background: #ddf4ff; padding: 8px 12px; border-radius: 6px;
             margin: 12px 0; }
  .bottleneck { display: inline-block; padding: 2px 8px; border-radius: 12px;
                background: #fff8c5; color: #633c01; font-weight: 600;
                font-size: 0.85em; }
  table { border-collapse: collapse; width: 100%; font-family:
          ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em;
          margin-top: 10px; }
  th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: right; }
  th { background: #f6f8fa; font-weight: 600; }
  th.l, td.l { text-align: left; }
  tr:nth-child(even) { background: #fafbfc; }
  td.hi { background: #ffebe9; font-weight: 600; }
  td.med { background: #fff8c5; }
  .summary { background: #f6f8fa; padding: 12px; border-radius: 6px;
             margin-top: 12px; font-size: 0.9em; }
  .summary code { background: #eaeef2; padding: 1px 6px; border-radius: 3px; }
  .overview table { font-size: 0.9em; }
  .overview td.hi { font-weight: 700; }
  .legend { font-size: 0.85em; color: #57606a; margin-top: 8px; }
  .legend code { background: #eaeef2; padding: 1px 6px; border-radius: 3px; }
</style>
"""


def render_per_request_table(wl: WL, result) -> str:
    rows = result.per_request
    headers = ["rid", "arrive", "pf_start", "pf_end", "kv_ready",
               "dc_start", "dc_end", "pq_wait", "kv_xfer", "dq_wait",
               "pf_run", "dc_run", "ttft", "e2e"]
    head_html = "".join(
        f"<th class='l'>{h}</th>" if h == "rid" else f"<th>{h}</th>"
        for h in headers
    )

    # find max per column for highlighting
    pq_max  = max((r.prefill_queue_wait or 0) for r in rows)
    kv_max  = max((r.kv_transfer_time or 0) for r in rows)
    dq_max  = max((r.decode_queue_wait or 0) for r in rows)
    e2e_max = max((r.e2e or 0) for r in rows)

    body = []
    for r in rows:
        pf_run = ((r.prefill_end_time - r.prefill_start_time)
                  if r.prefill_end_time and r.prefill_start_time is not None
                  else None)
        dc_run = ((r.decode_end_time - r.decode_start_time)
                  if r.decode_end_time and r.decode_start_time is not None
                  else None)

        def cell(val, is_max=False, med=False):
            cls = "hi" if is_max and val and val > 0 else ("med" if med else "")
            return f"<td class='{cls}'>{fmt(val)}</td>"

        body.append(
            "<tr>"
            f"<td class='l'>{r.rid}</td>"
            f"<td>{fmt(r.arrival_time)}</td>"
            f"<td>{fmt(r.prefill_start_time)}</td>"
            f"<td>{fmt(r.prefill_end_time)}</td>"
            f"<td>{fmt(r.kv_ready_time)}</td>"
            f"<td>{fmt(r.decode_start_time)}</td>"
            f"<td>{fmt(r.decode_end_time)}</td>"
            f"{cell(r.prefill_queue_wait, r.prefill_queue_wait == pq_max and pq_max>0)}"
            f"{cell(r.kv_transfer_time,  r.kv_transfer_time  == kv_max and kv_max>0)}"
            f"{cell(r.decode_queue_wait, r.decode_queue_wait == dq_max and dq_max>0)}"
            f"<td>{fmt(pf_run)}</td>"
            f"<td>{fmt(dc_run)}</td>"
            f"<td>{fmt(r.ttft)}</td>"
            f"{cell(r.e2e, r.e2e == e2e_max)}"
            "</tr>"
        )

    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_summary(wl: WL, result) -> str:
    rows = result.per_request
    pq = [r.prefill_queue_wait for r in rows if r.prefill_queue_wait is not None]
    kv = [r.kv_transfer_time for r in rows if r.kv_transfer_time is not None]
    dq = [r.decode_queue_wait for r in rows if r.decode_queue_wait is not None]
    e2e = [r.e2e for r in rows if r.e2e is not None]
    pf_run = [(r.prefill_end_time - r.prefill_start_time) for r in rows
              if r.prefill_end_time and r.prefill_start_time is not None]
    dc_run = [(r.decode_end_time - r.decode_start_time) for r in rows
              if r.decode_end_time and r.decode_start_time is not None]

    def stats(label, xs):
        if not xs:
            return f"<tr><td class='l'>{label}</td><td colspan='3'>–</td></tr>"
        mean = sum(xs) / len(xs) * 1e6
        mx = max(xs) * 1e6
        mn = min(xs) * 1e6
        return (f"<tr><td class='l'>{label}</td>"
                f"<td>{mn:,.2f}</td><td>{mean:,.2f}</td><td>{mx:,.2f}</td></tr>")

    return (
        f"<div class='summary'>"
        f"<b>final_clock</b> = <code>{result.final_clock*1e6:,.2f} µs</code> &nbsp;|&nbsp; "
        f"<b>n_reqs</b> = {len(rows)} &nbsp;|&nbsp; "
        f"<b>bottleneck</b> <span class='bottleneck'>{wl.bottleneck}</span>"
        f"<table style='margin-top:8px'>"
        f"<thead><tr><th class='l'>stage (µs)</th><th>min</th><th>mean</th><th>max</th></tr></thead>"
        f"<tbody>"
        f"{stats('prefill_queue_wait', pq)}"
        f"{stats('prefill_run',         pf_run)}"
        f"{stats('kv_transfer',         kv)}"
        f"{stats('decode_queue_wait',   dq)}"
        f"{stats('decode_run',          dc_run)}"
        f"{stats('e2e',                 e2e)}"
        f"</tbody></table></div>"
    )


def render_overview(rows) -> str:
    head = ("<tr><th class='l'>Workload</th><th>reqs</th><th>P/D</th>"
            "<th>isl</th><th>osl</th><th>bw (Gbps)</th>"
            "<th>max pf_run</th><th>max kv</th><th>max pq</th>"
            "<th>max e2e</th><th>final_clock</th>"
            "<th class='l'>bottleneck</th></tr>")
    body = []
    for wl, res in rows:
        pf_run = max((r.prefill_end_time - r.prefill_start_time) for r in res.per_request)
        kv     = max(r.kv_transfer_time for r in res.per_request)
        pq     = max(r.prefill_queue_wait for r in res.per_request)
        e2e    = max(r.e2e for r in res.per_request)
        body.append(
            f"<tr><td class='l'>{wl.name}</td><td>{len(res.per_request)}</td>"
            f"<td>{wl.pr}/{wl.dr}</td><td>{wl.isl}</td><td>{wl.osl}</td>"
            f"<td>{wl.bw:g}</td>"
            f"<td>{pf_run*1e6:,.1f}</td>"
            f"<td class='med'>{kv*1e6:,.1f}</td>"
            f"<td class='med'>{pq*1e6:,.1f}</td>"
            f"<td class='hi'>{e2e*1e6:,.1f}</td>"
            f"<td>{res.final_clock*1e6:,.1f}</td>"
            f"<td class='l'><span class='bottleneck'>{wl.bottleneck}</span></td></tr>"
        )
    return ("<div class='overview'><table><thead>" + head +
            "</thead><tbody>" + "".join(body) + "</tbody></table></div>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()

    runs = [(wl, run_one(wl)) for wl in WORKLOADS]

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>HiSim BackendA — W1..W6 Trace</title>", CSS, "</head><body>"]

    parts.append("<h1>HiSim BackendA — W1..W6 Per-Request Trace</h1>")
    parts.append(
        "<p class='meta'>Analytic predictor: prefill = 0.5 µs/tok &nbsp; | &nbsp; "
        "decode = 10 µs/step &nbsp; | &nbsp; KV transfer = "
        "<code>latency_us + bytes/bw_gbps</code>. "
        "All times are virtual-time microseconds.</p>"
    )

    parts.append("<h2>Overview — max per stage across W1..W6</h2>")
    parts.append(render_overview(runs))
    parts.append(
        "<p class='legend'>"
        "<span class='bottleneck'>yellow pill</span> = identified dominant stage. "
        "<code>max pq</code> = max prefill_queue_wait, "
        "<code>max kv</code> = max kv_transfer, "
        "<code>max e2e</code> highlights the run's slowest request.</p>"
    )

    for wl, result in runs:
        parts.append(f"<h2>{html.escape(wl.name)} "
                     f"<span class='meta'>(P={wl.pr}, D={wl.dr}, "
                     f"isl={wl.isl}, osl={wl.osl}, bw={wl.bw}Gbps, "
                     f"lat={wl.lat}µs, reqs={len(wl.reqs)})</span></h2>")
        parts.append(f"<div class='purpose'><b>Purpose:</b> "
                     f"{html.escape(wl.purpose)}</div>")
        parts.append(render_summary(wl, result))
        parts.append("<details open><summary>Per-request timestamps "
                     "(red = max in column)</summary>")
        parts.append(render_per_request_table(wl, result))
        parts.append("</details>")

    parts.append("""
<h2>Reading the numbers</h2>
<div class='summary'>
<b>e2e decomposition for one request:</b><br>
<code>e2e = prefill_queue_wait + prefill_run + kv_transfer + decode_queue_wait + decode_run</code><br><br>
<b>Bottleneck identification — read the &quot;max&quot; column of the per-stage table:</b>
<ul>
  <li><b>W4</b> max e2e ≈ 20,500 µs ≫ everything else → <code>decode_run</code> dominates → scale decode replicas / decode TP.</li>
  <li><b>W5</b> max prefill_queue_wait ≈ 3,584 µs grows linearly with rid → prefill pool is saturated → scale prefill replicas / prefill TP.</li>
  <li><b>W6</b> max kv_transfer ≈ 6,721 µs (vs 94 µs in W1) → KV link bandwidth is the bottleneck → upgrade NIC / RDMA / NVLink.</li>
  <li><b>W3</b> max prefill_queue_wait ≈ 960 µs forms a perfect 64µs-step staircase → earliest-replica scheduling on 2 prefill workers.</li>
</ul>

<b>Why this is enough for bottleneck localization:</b><br>
The three queue/transfer percentiles (<code>prefill_queue_p95</code>, <code>kv_transfer_p95</code>,
<code>decode_queue_p95</code>) plus the two compute durations (<code>prefill_run</code>, <code>decode_run</code>)
cover all 5 stages of the PD pipeline. Whichever has the largest absolute value is the dominant cost.
</div>
""")
    parts.append("</body></html>")

    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
