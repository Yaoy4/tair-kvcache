"""PD disaggregation demo (post Phase 1c).

Runs a small in-process simulation that exercises the real PD building blocks
shipped so far:

  * pd_config.DisaggConfig / RolePredictorConfig / BandwidthTransferConfig
  * pd_factory.build_disagg (with a stub predictor — no AIC DB needed)
  * pd_transfer.BandwidthTransferModel
  * pd_controller.PDController (full state machine)

What is NOT yet built (and therefore inlined in this demo as a placeholder):
  * Backend A's actual scheduler-hook integration (Phase 2)
  * Stage-aware metrics aggregation (Phase 3)
  * AIC bridge JSON emission (Phase 1d)

The intent is so you can SEE:
  - heterogeneous prefill vs decode device names flowing through to per-role
    predictors;
  - kv_bytes_per_token derived from utils (not hand-picked);
  - prefill and decode pools advancing on INDEPENDENT virtual-time clocks;
  - KV transfer time inserted between phases;
  - per-request TTFT / KV-transit / E2E numbers that change as you adjust the
    --bw, --prefill-replicas, or --decode-replicas knobs.

Usage:
  PYTHONPATH=hisim/src:../aiconfigurator/src \\
    python hisim/tools/pd_demo.py --requests 6 --bw-gbps 100 --latency-us 10
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from typing import Optional

from hisim.spec import ModelInfo
from hisim.simulation.types import SchedulerConfig
from hisim.spec import DataType
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_factory import DisaggPredictors, build_disagg
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.sim_args import SimulationArgs


# ---------------------------------------------------------------------------
# Stubs that stand in for parts not yet built.
# ---------------------------------------------------------------------------


@dataclass
class _StubHW:
    name: str


def _hw_factory(name: str) -> _StubHW:
    return _StubHW(name)


class _AnalyticPredictor:
    """Predicts per-step latency without needing AIC perf DBs.

    Prefill: linear in total prefill tokens (microseconds per token).
    Decode:  fixed per-step latency, modulated by batch size.

    Sized so heterogeneous devices behave plausibly: faster device → smaller
    per-token coefficient.
    """

    # microseconds-per-token for prefill, microseconds-per-step for decode
    _PREFILL_US_PER_TOK = {
        "h100_sxm": 0.4,
        "h200_sxm": 0.3,
        "h20": 1.2,
        "a100_sxm": 0.8,
    }
    _DECODE_US_PER_STEP = {
        "h100_sxm": 12.0,
        "h200_sxm": 10.0,
        "h20": 25.0,
        "a100_sxm": 18.0,
    }

    def __init__(self, model, hw, config, **kwargs):
        self.model = model
        self.hw = hw
        self.config = config
        # Per-device scaling, with a TP discount (rough).
        device = hw.name
        tp = max(1, config.tp_size)
        self.prefill_us_per_tok = self._PREFILL_US_PER_TOK.get(device, 0.5) / tp
        self.decode_us_per_step = self._DECODE_US_PER_STEP.get(device, 15.0) / tp

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self.prefill_us_per_tok * 1e-6

    def predict_decode_seconds(self, batch_size: int) -> float:
        # weak super-linearity; demo only
        return self.decode_us_per_step * (1 + 0.05 * max(0, batch_size - 1)) * 1e-6


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_row(req: PDRequestState, ttft, kv_ms, e2e) -> str:
    return (
        f"  rid={req.rid:<6} in={req.input_length:<5} out={req.output_length:<4} "
        f"TTFT={ttft*1000:7.2f} ms   KV={kv_ms:6.2f} ms   "
        f"E2E={e2e*1000:8.2f} ms   phase={req.phase.name}"
    )


def run_demo(args) -> None:
    # ---- build inputs ------------------------------------------------------
    model = ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="demo-llama-7b-ish",
    )
    base_sched = SchedulerConfig(
        model=model,
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )
    if args.config is not None:
        # Load disagg config from a Hisim JSON (e.g., produced by
        # tools/aic_to_hisim_bridge.py --emit-disagg). This exercises the
        # full Phase 1b + 1d pipeline.
        sim_args = SimulationArgs.from_json(args.config)
        disagg_cfg = sim_args.disagg
        if not disagg_cfg.enabled:
            raise SystemExit(
                f"Config {args.config} has no enabled disagg block. "
                "Pass --emit-disagg to the AIC bridge when generating it."
            )
        print(f"[demo] loaded disagg config from {args.config}")
    else:
        disagg_cfg = DisaggConfig(
            enabled=True,
            backend="single_process",
            prefill=RolePredictorConfig(
                device_name=args.prefill_device,
                tp_size=args.prefill_tp,
                replicas=args.prefill_replicas,
                max_running_per_replica=8,
            ),
            decode=RolePredictorConfig(
                device_name=args.decode_device,
                tp_size=args.decode_tp,
                replicas=args.decode_replicas,
                max_running_per_replica=64,
            ),
            kv_transfer=BandwidthTransferConfig(
                bw_gbps=args.bw_gbps, latency_us=args.latency_us
            ),
        )

    # ---- build factory output (real code path) -----------------------------
    bundle: DisaggPredictors = build_disagg(
        model=model,
        base_sched_config=base_sched,
        disagg_config=disagg_cfg,
        predictor_factory=_AnalyticPredictor,
        hw_factory=_hw_factory,
    )

    print("=" * 78)
    print("PD demo  —  showing real Phase 0/1 plumbing in action")
    print("=" * 78)
    print(
        f"model            : {model.name}  "
        f"(layers={model.num_hidden_layers}, kv_heads={model.num_key_value_heads})"
    )
    print(
        f"prefill role     : device={disagg_cfg.prefill.device_name:<10} "
        f"tp={disagg_cfg.prefill.tp_size}  "
        f"replicas={disagg_cfg.prefill.replicas}"
    )
    print(
        f"decode  role     : device={disagg_cfg.decode.device_name:<10} "
        f"tp={disagg_cfg.decode.tp_size}  "
        f"replicas={disagg_cfg.decode.replicas}"
    )
    print(
        f"KV transfer      : bw={disagg_cfg.kv_transfer.bw_gbps} GB/s, "
        f"latency={disagg_cfg.kv_transfer.latency_us} us"
    )
    print(
        f"kv_bytes_per_tok : {bundle.kv_bytes_per_token}  "
        f"(derived via utils.calc_kv_cache_cell_elems — never hand-picked)"
    )
    print("-" * 78)

    # ---- build a workload --------------------------------------------------
    rng_inputs = [256, 512, 1024, 768, 2048, 384, 1536, 640, 320, 1280]
    rng_outputs = [64, 128, 32, 96, 48, 80, 64, 128, 96, 48]
    reqs = []
    for i in range(args.requests):
        reqs.append(
            PDRequestState(
                rid=f"r{i}",
                arrival_time=i * 0.002,  # 2 ms apart
                phase=RequestPhase.WAITING_PREFILL,
                input_length=rng_inputs[i % len(rng_inputs)],
                output_length=rng_outputs[i % len(rng_outputs)],
            )
        )

    # ---- run a simple two-clock simulation via BackendA --------------------
    backend = BackendA(bundle)

    # event queue of (time, seq, kind, payload)
    events = []
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, payload))

    for r in reqs:
        push(r.arrival_time, "arrive", r)

    # bookkeeping
    ttft = {}
    kv_seconds = {}
    e2e = {}

    while events:
        now, _, kind, payload = heapq.heappop(events)

        if kind == "arrive":
            req = payload
            _, end = backend.try_admit_prefill(req, now)
            push(end, "prefill_done", req)

        elif kind == "prefill_done":
            req = payload
            kv_ready = backend.compute_kv_ready_time(req, now)
            kv_seconds[req.rid] = kv_ready - now
            backend.on_prefill_done(req, now, kv_ready)
            push(kv_ready, "kv_ready", req)

        elif kind == "kv_ready":
            req = payload
            backend.advance_to_kv_ready(req, now)
            # admit one decode slot via the controller, then schedule one step
            admitted = backend.controller().admit_decode(capacity=1, now=now)
            if not admitted:
                continue
            req = admitted[0]
            idx, end = backend.try_admit_decode_step(req, now)
            ttft[req.rid] = (end - bundle.decode.predict_decode_seconds(1)) - req.arrival_time
            push(end, "decode_step", (req, idx))

        elif kind == "decode_step":
            req, idx = payload
            backend.on_decode_step_done(req, now)
            if req.phase == RequestPhase.FINISHED:
                e2e[req.rid] = now - req.arrival_time
                continue
            _, end = backend.try_admit_decode_step(req, now)
            push(end, "decode_step", (req, idx))

    # ---- print -------------------------------------------------------------
    print("Per-request results:")
    for r in reqs:
        print(
            _format_row(
                r,
                ttft.get(r.rid, float("nan")),
                kv_seconds.get(r.rid, 0.0) * 1000,
                e2e.get(r.rid, float("nan")),
            )
        )

    print("-" * 78)
    total_ttft = sum(ttft.values()) / max(1, len(ttft))
    total_e2e = sum(e2e.values()) / max(1, len(e2e))
    total_kv = sum(kv_seconds.values()) / max(1, len(kv_seconds))
    print(
        f"averages         : TTFT={total_ttft*1000:.2f} ms   "
        f"KV-transit={total_kv*1000:.2f} ms   E2E={total_e2e*1000:.2f} ms"
    )
    print(
        f"final prefill clock per replica : "
        f"{[round(x*1000,2) for x in backend._prefill_pool.busy_until]} ms"
    )
    print(
        f"final decode  clock per replica : "
        f"{[round(x*1000,2) for x in backend._decode_pool.busy_until]} ms"
    )
    print(
        f"controller queues  : prefill_waiting={backend.controller().prefill_waiting_count()}  "
        f"kv_transit={backend.controller().kv_transit_count()}  "
        f"decode_waiting={backend.controller().decode_waiting_count()}"
    )
    print("=" * 78)
    print(
        "Try:   --bw-gbps 25      (slower KV link → KV-transit grows)\n"
        "       --prefill-replicas 1 --decode-replicas 4 (queueing shifts)\n"
        "       --prefill-device h20 --decode-device h100_sxm (heterogeneous)\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=6)
    p.add_argument("--config", default=None,
                   help="Optional Hisim config JSON (e.g. produced by aic_to_hisim_bridge.py --emit-disagg)")
    p.add_argument("--prefill-device", default="h100_sxm")
    p.add_argument("--decode-device", default="h100_sxm")
    p.add_argument("--prefill-tp", type=int, default=1)
    p.add_argument("--decode-tp", type=int, default=1)
    p.add_argument("--prefill-replicas", type=int, default=2)
    p.add_argument("--decode-replicas", type=int, default=2)
    p.add_argument("--bw-gbps", type=float, default=100.0)
    p.add_argument("--latency-us", type=float, default=10.0)
    args = p.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
