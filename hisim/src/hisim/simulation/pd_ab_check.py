"""Phase 6e — CLI entry point for the A/B equivalence check.

Wraps :mod:`hisim.simulation.pd_ab_harness` + :mod:`hisim.simulation.pd_ab_compare`
into a single ``python -m hisim.simulation.pd_ab_check`` invocation suitable
for CI.

Exit codes:
* ``0`` — all workloads classified OK (or WARN).
* ``1`` — at least one workload returned a *blocking* classification
  (STRUCT, PER_REQ, or AGGREGATE per the 6a spec).
* ``2`` — usage / config error before any backend ran.

The picklable predictor / role factory used here is the same analytic
closed-form pair already used by the W1–W6 unit tests, so this script
can run in any environment without an AIC perf DB. A future flag can
swap in a real-AIC factory.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_ab_compare import ComparisonReport, FailureClass, classify
from hisim.simulation.pd_ab_harness import RunResult, WorkloadRequest, run_ab
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.types import SchedulerConfig


# ---------------------------------------------------------------------------
# Picklable defaults (top-level so spawned BackendB workers can import them).
# ---------------------------------------------------------------------------


class _StubHW:
    def __init__(self, name: str):
        self.name = name


def _stub_hw_factory(name: str) -> _StubHW:
    return _StubHW(name)


class _AnalyticPredictor:
    """Closed-form deterministic predictor — A/B-identical by construction."""

    _PREFILL_US_PER_TOK = 0.5
    _DECODE_US_PER_STEP = 10.0

    def __init__(self, model=None, hw=None, config=None, **kwargs):
        pass

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self._PREFILL_US_PER_TOK * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        return self._DECODE_US_PER_STEP * 1e-6 * max(1, batch_size)


class _AnalyticAdapterFactory:
    def __call__(self):
        return _AnalyticPredictor()


def _analytic_role_factory(model, base_sched_config, role):
    return _AnalyticAdapterFactory()


# ---------------------------------------------------------------------------
# Workload matrix (mirrors test_pd_ab_workload_matrix.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    name: str
    requests: Sequence[WorkloadRequest]
    prefill_replicas: int
    decode_replicas: int
    bw_gbps: float = 100.0
    latency_us: float = 10.0


def _serial(n: int, *, input_len: int, output_len: int,
            interarrival: float = 0.0) -> List[WorkloadRequest]:
    return [
        WorkloadRequest(
            rid=f"r{i}",
            arrival_time=i * interarrival,
            input_length=input_len,
            output_length=output_len,
        )
        for i in range(n)
    ]


def default_workloads() -> List[Workload]:
    return [
        Workload("W1-single",
                 _serial(1, input_len=64, output_len=16),
                 prefill_replicas=1, decode_replicas=1),
        Workload("W2-serial",
                 _serial(8, input_len=64, output_len=16, interarrival=1e-4),
                 prefill_replicas=1, decode_replicas=1),
        Workload("W3-concurrent",
                 _serial(32, input_len=128, output_len=64),
                 prefill_replicas=2, decode_replicas=2),
        Workload("W4-long-decode",
                 _serial(8, input_len=32, output_len=256),
                 prefill_replicas=1, decode_replicas=1),
        Workload("W5-long-prefill",
                 _serial(8, input_len=1024, output_len=16),
                 prefill_replicas=1, decode_replicas=1),
        Workload("W6-kv-bw-stress",
                 _serial(16, input_len=512, output_len=32),
                 prefill_replicas=2, decode_replicas=2, bw_gbps=10.0),
    ]


# ---------------------------------------------------------------------------
# Config builders.
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
        name="pd-ab-check",
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


def _make_disagg_cfg(w: Workload) -> DisaggConfig:
    return DisaggConfig(
        enabled=True,
        backend="single_process",  # overridden per side by run_ab
        prefill=RolePredictorConfig(
            device_name="H20", tp_size=1,
            replicas=w.prefill_replicas, max_running_per_replica=64,
        ),
        decode=RolePredictorConfig(
            device_name="H20", tp_size=1,
            replicas=w.decode_replicas, max_running_per_replica=64,
        ),
        kv_transfer=BandwidthTransferConfig(
            bw_gbps=w.bw_gbps, latency_us=w.latency_us,
        ),
    )


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


@dataclass
class WorkloadOutcome:
    name: str
    report: ComparisonReport
    runtime_seconds: float


def run_check(
    workloads: Optional[Sequence[Workload]] = None,
    *,
    predictor_factory: Optional[Callable] = None,
    role_factory: Optional[Callable] = None,
    hw_factory: Optional[Callable] = None,
) -> Tuple[List[WorkloadOutcome], bool]:
    """Run ``run_ab`` + ``classify`` across ``workloads``.

    Returns ``(outcomes, blocking)`` where ``blocking`` is True iff any
    workload's report ``is_blocking``. Pure function — no I/O.
    """
    import time

    workloads = list(workloads) if workloads is not None else default_workloads()
    predictor_factory = predictor_factory or _AnalyticPredictor
    role_factory = role_factory or _analytic_role_factory
    hw_factory = hw_factory or _stub_hw_factory

    outcomes: List[WorkloadOutcome] = []
    blocking = False
    for w in workloads:
        cfg = _make_disagg_cfg(w)
        t0 = time.perf_counter()
        result_a, result_b = run_ab(
            model=_make_model(),
            base_sched_config=_make_base_sched(),
            disagg_config=cfg,
            workload=w.requests,
            predictor_factory=predictor_factory,
            role_factory=role_factory,
            hw_factory=hw_factory,
        )
        report = classify(result_a, result_b)
        elapsed = time.perf_counter() - t0
        outcomes.append(WorkloadOutcome(w.name, report, elapsed))
        if report.is_blocking:
            blocking = True
    return outcomes, blocking


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _format_outcome(o: WorkloadOutcome, *, verbose: bool) -> str:
    head = (
        f"  {o.name:<20s} {o.report.classification.value:<10s} "
        f"({o.runtime_seconds:5.2f}s, "
        f"n={int(o.report.aggregate_a.get('n_finished', 0))})"
    )
    if not verbose or not o.report.mismatches:
        return head
    lines = [head, "    first mismatches:"]
    for m in o.report.mismatches[:5]:
        lines.append(f"      {m}")
    if len(o.report.mismatches) > 5:
        lines.append(f"      ... +{len(o.report.mismatches) - 5} more")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pd_ab_check",
        description="Run A/B equivalence across the W1-W6 workload matrix.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print first 5 mismatches for any non-OK workload.",
    )
    parser.add_argument(
        "--workload", action="append", default=None,
        help="Restrict to one or more workloads by name (W1-single, etc.). "
             "Repeatable. Default: run all.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    all_workloads = default_workloads()
    if args.workload:
        selected = [w for w in all_workloads if w.name in set(args.workload)]
        if not selected:
            print(
                f"error: no workloads matched {args.workload!r}; "
                f"available={[w.name for w in all_workloads]}",
                file=sys.stderr,
            )
            return 2
    else:
        selected = all_workloads

    print(f"pd_ab_check: running {len(selected)} workload(s)")
    outcomes, blocking = run_check(selected)
    for o in outcomes:
        print(_format_outcome(o, verbose=args.verbose or o.report.is_blocking))

    n_ok = sum(1 for o in outcomes if o.report.classification is FailureClass.OK)
    n_block = sum(1 for o in outcomes if o.report.is_blocking)
    print(
        f"\nsummary: {n_ok}/{len(outcomes)} OK, "
        f"{n_block} blocking failure(s)"
    )
    return 1 if blocking else 0


if __name__ == "__main__":  # pragma: no cover — exercised via tests + CI
    sys.exit(main())
