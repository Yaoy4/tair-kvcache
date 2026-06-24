"""Phase 6d — A/B equivalence across the W1–W6 workload matrix.

These are end-to-end equivalence tests: build BackendA and BackendB from the
same config, drive both through each workload via ``run_ab``, then assert
``classify`` returns ``FailureClass.OK``.

Per the 6a spec, *any* non-OK classification is either a backend bug or a
spec violation — never a tolerance miss. Tests therefore assert
``classification is OK`` and dump the report on failure for diagnosis.

Cost note: W3 (32 reqs × 64 decode steps) and W4 (8 reqs × 256 decode steps)
both incur ~2k IPC roundtrips per backend through BackendB workers. The
suite runs in <30s on a developer machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.pd_ab_compare import FailureClass, classify
from hisim.simulation.pd_ab_harness import WorkloadRequest, run_ab
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.types import SchedulerConfig


# ---------------------------------------------------------------------------
# Picklable helpers (top-level so spawned BackendB workers can import them).
# ---------------------------------------------------------------------------


class _StubHW:
    def __init__(self, name: str):
        self.name = name


def _stub_hw_factory(name: str) -> _StubHW:
    return _StubHW(name)


class _AnalyticPredictor:
    """Deterministic closed-form predictor — identical on A and B."""

    _PREFILL_US_PER_TOK = 0.5
    _DECODE_US_PER_STEP = 10.0

    def __init__(self, model=None, hw=None, config=None, **kwargs):
        pass

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self._PREFILL_US_PER_TOK * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        return self._DECODE_US_PER_STEP * 1e-6 * max(1, batch_size)


class _AnalyticAdapterFactory:
    """Picklable factory for BackendB workers."""

    def __call__(self):
        return _AnalyticPredictor()


def _analytic_role_factory(model, base_sched_config, role):
    return _AnalyticAdapterFactory()


def _make_model() -> ModelInfo:
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="phase6d-stub",
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


def _make_disagg_cfg(
    *,
    prefill_replicas: int,
    decode_replicas: int,
    bw_gbps: float = 100.0,
    latency_us: float = 10.0,
) -> DisaggConfig:
    return DisaggConfig(
        enabled=True,
        backend="single_process",  # overridden per side by run_ab
        prefill=RolePredictorConfig(
            device_name="H20",
            tp_size=1,
            replicas=prefill_replicas,
            max_running_per_replica=64,
        ),
        decode=RolePredictorConfig(
            device_name="H20",
            tp_size=1,
            replicas=decode_replicas,
            max_running_per_replica=64,
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=bw_gbps, latency_us=latency_us),
    )


# ---------------------------------------------------------------------------
# Workload matrix (mirrors 6a spec §"Workload matrix the harness MUST cover").
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    name: str
    requests: List[WorkloadRequest]
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


def _workload_w2() -> Workload:
    # 8 reqs, serial arrivals (small inter-arrival), small input/output.
    return Workload(
        name="W2-serial",
        requests=_serial(8, input_len=64, output_len=16, interarrival=1e-4),
        prefill_replicas=1,
        decode_replicas=1,
    )


def _workload_w3() -> Workload:
    # 32 concurrent (all arrive at t=0), 2/2 replicas.
    return Workload(
        name="W3-concurrent",
        requests=_serial(32, input_len=128, output_len=64, interarrival=0.0),
        prefill_replicas=2,
        decode_replicas=2,
    )


def _workload_w4() -> Workload:
    # decode-bound: long output, short input, 1/1.
    return Workload(
        name="W4-long-decode",
        requests=_serial(8, input_len=32, output_len=256, interarrival=0.0),
        prefill_replicas=1,
        decode_replicas=1,
    )


def _workload_w5() -> Workload:
    # prefill-bound: long input, short output, 1/1.
    return Workload(
        name="W5-long-prefill",
        requests=_serial(8, input_len=1024, output_len=16, interarrival=0.0),
        prefill_replicas=1,
        decode_replicas=1,
    )


def _workload_w6() -> Workload:
    # KV-bandwidth stress: deliberately low bw_gbps so kv_transfer_time
    # is the dominant per-request term.
    return Workload(
        name="W6-kv-bw-stress",
        requests=_serial(16, input_len=512, output_len=32, interarrival=0.0),
        prefill_replicas=2,
        decode_replicas=2,
        bw_gbps=10.0,
    )


ALL_WORKLOADS: Tuple[Workload, ...] = (
    _workload_w2(),
    _workload_w3(),
    _workload_w4(),
    _workload_w5(),
    _workload_w6(),
)


# ---------------------------------------------------------------------------
# Parameterized equivalence test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workload", ALL_WORKLOADS, ids=lambda w: w.name)
def test_run_ab_equivalence_across_workload_matrix(workload):
    cfg = _make_disagg_cfg(
        prefill_replicas=workload.prefill_replicas,
        decode_replicas=workload.decode_replicas,
        bw_gbps=workload.bw_gbps,
        latency_us=workload.latency_us,
    )
    result_a, result_b = run_ab(
        model=_make_model(),
        base_sched_config=_make_base_sched(),
        disagg_config=cfg,
        workload=workload.requests,
        predictor_factory=_AnalyticPredictor,
        role_factory=_analytic_role_factory,
        hw_factory=_stub_hw_factory,
    )

    # Sanity: every request finished on both sides.
    assert len(result_a.per_request) == len(workload.requests)
    assert len(result_b.per_request) == len(workload.requests)
    assert all(r.finished for r in result_a.per_request), (
        f"BackendA did not finish all requests on {workload.name}"
    )
    assert all(r.finished for r in result_b.per_request), (
        f"BackendB did not finish all requests on {workload.name}"
    )

    rep = classify(result_a, result_b)
    if rep.classification is not FailureClass.OK:
        # Build a compact diagnostic: classification + first 5 mismatches.
        head = "\n  ".join(str(m) for m in rep.mismatches[:5])
        pytest.fail(
            f"{workload.name}: classification={rep.classification.value}\n"
            f"  first mismatches:\n  {head}\n"
            f"  agg_a={rep.aggregate_a}\n  agg_b={rep.aggregate_b}"
        )

    # Spec invariants: aggregate metrics are non-empty and well-formed.
    assert rep.aggregate_a["n_finished"] == len(workload.requests)
    assert rep.aggregate_a["mean_e2e"] > 0.0
    assert rep.aggregate_a["throughput_rps"] > 0.0


# ---------------------------------------------------------------------------
# Extra W3-focused stress test: with concurrent arrivals, the replica-pool
# tie-breaking must produce identical ordering on A and B. If it ever
# diverges, this test is the canary.
# ---------------------------------------------------------------------------


def test_run_ab_w3_concurrent_replica_ordering_is_identical():
    workload = _workload_w3()
    cfg = _make_disagg_cfg(
        prefill_replicas=workload.prefill_replicas,
        decode_replicas=workload.decode_replicas,
    )
    result_a, result_b = run_ab(
        model=_make_model(),
        base_sched_config=_make_base_sched(),
        disagg_config=cfg,
        workload=workload.requests,
        predictor_factory=_AnalyticPredictor,
        role_factory=_analytic_role_factory,
        hw_factory=_stub_hw_factory,
    )
    # Pin: every request's prefill_start_time must match exactly between
    # backends. If tie-breaking diverged, multiple rids would shift by one
    # prefill-slot's duration.
    a_by = result_a.by_rid()
    b_by = result_b.by_rid()
    for rid in sorted(a_by):
        a_t = a_by[rid].prefill_start_time
        b_t = b_by[rid].prefill_start_time
        assert a_t is not None and b_t is not None
        assert abs(a_t - b_t) <= 1e-6, (
            f"prefill_start_time diverged for {rid}: A={a_t} B={b_t}"
        )
