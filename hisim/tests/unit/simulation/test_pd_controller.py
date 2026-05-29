from abc import ABC
from typing import List

import pytest

from hisim.simulation.pd_controller import ExecutionBackend, PDController
from hisim.simulation.pd_transfer import BandwidthTransferModel, KVModelConfig
from hisim.simulation.pd_types import PDRequestState, RequestPhase


class RecordingBackend(ExecutionBackend):
    def __init__(self):
        self.prefill_submissions: List[List[PDRequestState]] = []
        self.decode_steps: List[List[PDRequestState]] = []
        self.handoffs: List[PDRequestState] = []
        self.clock_advances: List[float] = []

    def submit_prefill_batch(self, reqs, now):
        self.prefill_submissions.append(list(reqs))

    def progress_decode(self, reqs, now):
        self.decode_steps.append(list(reqs))

    def advance_clock(self, now):
        self.clock_advances.append(now)

    def handoff_kv(self, req, now):
        self.handoffs.append(req)


def make_controller(bw_gbps=100.0, latency_us=10.0, kv_bytes_per_token=1024):
    transfer = BandwidthTransferModel(bw_gbps=bw_gbps, latency_us=latency_us)
    kv_cfg = KVModelConfig(kv_bytes_per_token=kv_bytes_per_token)
    return PDController(transfer_model=transfer, kv_model_cfg=kv_cfg)


def test_execution_backend_is_abstract():
    assert issubclass(ExecutionBackend, ABC)


def test_on_request_arrival_enqueues_prefill_waiting():
    ctrl = make_controller()
    req = PDRequestState(rid="r1", arrival_time=0.0)
    ctrl.on_request_arrival(req, now=0.0)
    assert req.phase == RequestPhase.WAITING_PREFILL
    assert ctrl.prefill_waiting_count() == 1
    assert ctrl.decode_waiting_count() == 0
    assert ctrl.kv_transit_count() == 0


def test_admit_prefill_respects_capacity_and_transitions_phase():
    ctrl = make_controller()
    for i in range(3):
        ctrl.on_request_arrival(
            PDRequestState(rid=f"r{i}", arrival_time=0.0), now=0.0
        )

    admitted = ctrl.admit_prefill(capacity=2, now=0.0)
    assert len(admitted) == 2
    for req in admitted:
        assert req.phase == RequestPhase.RUNNING_PREFILL
        assert req.prefill_start_time == 0.0
    assert ctrl.prefill_waiting_count() == 1


def test_admit_prefill_with_zero_capacity_admits_nothing():
    ctrl = make_controller()
    ctrl.on_request_arrival(PDRequestState(rid="r1", arrival_time=0.0), now=0.0)
    assert ctrl.admit_prefill(capacity=0, now=0.0) == []
    assert ctrl.prefill_waiting_count() == 1


def test_compute_kv_ready_time_uses_bandwidth_model():
    ctrl = make_controller(bw_gbps=100.0, latency_us=10.0, kv_bytes_per_token=1024)
    req = PDRequestState(rid="r1", arrival_time=0.0, input_length=2048)
    expected = 1.0 + 10e-6 + (2048 * 1024) / (100.0 * 1e9)
    assert ctrl.compute_kv_ready_time(req, now=1.0) == pytest.approx(expected, rel=1e-12)


def test_on_prefill_done_uses_caller_supplied_kv_ready_time():
    ctrl = make_controller(bw_gbps=100.0, latency_us=10.0, kv_bytes_per_token=1024)
    req = PDRequestState(rid="r1", arrival_time=0.0, input_length=2048)
    ctrl.on_request_arrival(req, now=0.0)
    ctrl.admit_prefill(capacity=1, now=0.0)

    # Caller (backend) supplies the absolute ready time. The controller does NOT compute it.
    ctrl.on_prefill_done(req, now=1.0, kv_ready_time=3.25)

    assert req.phase == RequestPhase.KV_TRANSIT
    assert req.prefill_end_time == 1.0
    assert req.kv_ready_time == 3.25
    assert ctrl.kv_transit_count() == 1


def test_on_prefill_done_rejects_kv_ready_time_before_now():
    ctrl = make_controller()
    req = PDRequestState(rid="r1", arrival_time=0.0, input_length=8)
    ctrl.on_request_arrival(req, now=0.0)
    ctrl.admit_prefill(capacity=1, now=0.0)
    with pytest.raises(ValueError):
        ctrl.on_prefill_done(req, now=1.0, kv_ready_time=0.5)


def test_poll_kv_ready_returns_only_ready_requests_and_moves_to_waiting_decode():
    ctrl = make_controller(bw_gbps=100.0, latency_us=10.0, kv_bytes_per_token=1024)
    req = PDRequestState(rid="r1", arrival_time=0.0, input_length=2048)
    ctrl.on_request_arrival(req, now=0.0)
    ctrl.admit_prefill(capacity=1, now=0.0)
    ready_at = ctrl.compute_kv_ready_time(req, now=1.0)
    ctrl.on_prefill_done(req, now=1.0, kv_ready_time=ready_at)

    # Not yet ready.
    assert ctrl.poll_kv_ready(now=1.0) == []
    assert req.phase == RequestPhase.KV_TRANSIT

    ready = ctrl.poll_kv_ready(now=ready_at)
    assert ready == [req]
    assert req.phase == RequestPhase.WAITING_DECODE
    assert ctrl.kv_transit_count() == 0
    assert ctrl.decode_waiting_count() == 1


def test_admit_decode_respects_capacity_and_sets_decode_start_time():
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    reqs = []
    for i in range(3):
        r = PDRequestState(rid=f"r{i}", arrival_time=0.0, input_length=8)
        ctrl.on_request_arrival(r, now=0.0)
        ctrl.admit_prefill(capacity=3, now=0.0)
        ctrl.on_prefill_done(r, now=1.0, kv_ready_time=ctrl.compute_kv_ready_time(r, 1.0))
        reqs.append(r)
    ctrl.poll_kv_ready(now=10.0)

    admitted = ctrl.admit_decode(capacity=2, now=10.0)
    assert len(admitted) == 2
    for r in admitted:
        assert r.phase == RequestPhase.RUNNING_DECODE
        assert r.decode_start_time == 10.0
    assert ctrl.decode_waiting_count() == 1


def test_on_decode_step_done_increments_state_and_finishes_when_done():
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    req = PDRequestState(rid="r1", arrival_time=0.0, input_length=4, output_length=2)
    ctrl.on_request_arrival(req, now=0.0)
    ctrl.admit_prefill(capacity=1, now=0.0)
    ctrl.on_prefill_done(req, now=1.0, kv_ready_time=ctrl.compute_kv_ready_time(req, 1.0))
    ctrl.poll_kv_ready(now=10.0)
    ctrl.admit_decode(capacity=1, now=10.0)

    # past_kv_length should be initialized to input_length on decode admission.
    assert req.current_past_kv_length == 4

    ctrl.on_decode_step_done([req], now=10.1)
    assert req.decode_step_count == 1
    assert req.current_past_kv_length == 5
    assert req.phase == RequestPhase.RUNNING_DECODE

    ctrl.on_decode_step_done([req], now=10.2)
    assert req.decode_step_count == 2
    assert req.current_past_kv_length == 6
    assert req.phase == RequestPhase.FINISHED


def test_full_state_flow_end_to_end():
    ctrl = make_controller(bw_gbps=100.0, latency_us=10.0, kv_bytes_per_token=1024)
    req = PDRequestState(rid="r1", arrival_time=0.0, input_length=512, output_length=1)

    ctrl.on_request_arrival(req, now=0.0)
    assert req.phase == RequestPhase.WAITING_PREFILL

    ctrl.admit_prefill(capacity=1, now=0.0)
    assert req.phase == RequestPhase.RUNNING_PREFILL

    kv_ready = ctrl.compute_kv_ready_time(req, now=0.5)
    ctrl.on_prefill_done(req, now=0.5, kv_ready_time=kv_ready)
    assert req.phase == RequestPhase.KV_TRANSIT

    ctrl.poll_kv_ready(now=req.kv_ready_time)
    assert req.phase == RequestPhase.WAITING_DECODE

    ctrl.admit_decode(capacity=1, now=req.kv_ready_time)
    assert req.phase == RequestPhase.RUNNING_DECODE

    ctrl.on_decode_step_done([req], now=req.kv_ready_time + 0.01)
    assert req.phase == RequestPhase.FINISHED


def test_pd_controller_has_no_sglang_dependency():
    import ast
    import inspect

    module = inspect.getmodule(PDController)
    tree = ast.parse(inspect.getsource(module))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name.startswith("sglang") for name in imported)
    assert not any(name.startswith("hisim.simulation.sglang") for name in imported)
