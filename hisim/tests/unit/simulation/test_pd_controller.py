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


def test_invalid_phase_transition_is_rejected():
    ctrl = make_controller()
    req = PDRequestState(rid="r1", arrival_time=0.0)
    ctrl.on_request_arrival(req, now=0.0)
    ctrl.admit_prefill(capacity=1, now=0.0)

    with pytest.raises(ValueError, match="request arrival"):
        ctrl.on_request_arrival(req, now=0.1)


def test_compute_kv_ready_time_uses_bandwidth_model():
    ctrl = make_controller(bw_gbps=100.0, latency_us=10.0, kv_bytes_per_token=1024)
    req = PDRequestState(rid="r1", arrival_time=0.0, input_length=2048)
    expected = 1.0 + 10e-6 + (2048 * 1024) / (100.0 * 1e9)
    assert ctrl.compute_kv_ready_time(req, now=1.0) == pytest.approx(expected, rel=1e-12)


def test_kv_transfers_share_one_capacity_accounted_link():
    ctrl = make_controller(bw_gbps=1.0, latency_us=10.0, kv_bytes_per_token=1024)
    first = ctrl.compute_batch_kv_ready_time(total_tokens=100, now=1.0)
    second = ctrl.compute_batch_kv_ready_time(total_tokens=100, now=1.0)
    one_transfer = first - 1.0

    assert second == pytest.approx(first + one_transfer)


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


def test_admit_decode_targeted_only_starts_requested_rids():
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    reqs = []
    for i in range(3):
        r = PDRequestState(rid=f"r{i}", arrival_time=0.0, input_length=8)
        ctrl.on_request_arrival(r, now=0.0)
        ctrl.admit_prefill(capacity=3, now=0.0)
        ctrl.on_prefill_done(
            r,
            now=1.0,
            kv_ready_time=ctrl.compute_kv_ready_time(r, 1.0),
        )
        reqs.append(r)
    ctrl.poll_kv_ready(now=10.0)

    admitted = ctrl.admit_decode_targeted({"r1"}, now=10.0)

    assert [r.rid for r in admitted] == ["r1"]
    assert reqs[1].phase == RequestPhase.RUNNING_DECODE
    assert reqs[1].decode_start_time == 10.0
    assert reqs[0].phase == RequestPhase.WAITING_DECODE
    assert reqs[2].phase == RequestPhase.WAITING_DECODE
    assert ctrl.decode_waiting_count() == 2
    later = ctrl.admit_decode(capacity=2, now=11.0)
    assert [r.rid for r in later] == ["r0", "r2"]


def test_admit_decode_targeted_respects_token_budget_over_count():
    """A token_budget cap can defer admission even when max_count still has
    room -- this is what lets Backend A stop a decode replica from
    accepting a batch that would exceed its real per-accelerator KV-token
    capacity, closing the gap that let the AIConfigurator predictor's
    OOM-negation sentinel silently corrupt a replica's simulated clock.
    """
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    reqs = []
    for i in range(3):
        # input_length=8, output_length=2 -> token cost 10 each.
        r = PDRequestState(
            rid=f"r{i}", arrival_time=0.0, input_length=8, output_length=2
        )
        ctrl.on_request_arrival(r, now=0.0)
        ctrl.admit_prefill(capacity=3, now=0.0)
        ctrl.on_prefill_done(
            r,
            now=1.0,
            kv_ready_time=ctrl.compute_kv_ready_time(r, 1.0),
        )
        reqs.append(r)
    ctrl.poll_kv_ready(now=10.0)

    def token_cost(req: PDRequestState) -> int:
        return req.input_length + req.output_length

    # max_count=3 (no count constraint) but token_budget=15 only fits ONE
    # request (cost 10) before the running sum (20) would exceed budget.
    admitted = ctrl.admit_decode_targeted(
        {"r0", "r1", "r2"},
        now=10.0,
        max_count=3,
        token_budget=15,
        token_cost=token_cost,
    )

    assert [r.rid for r in admitted] == ["r0"]
    assert reqs[0].phase == RequestPhase.RUNNING_DECODE
    assert reqs[1].phase == RequestPhase.WAITING_DECODE
    assert reqs[2].phase == RequestPhase.WAITING_DECODE
    assert ctrl.decode_waiting_count() == 2

    # Later round: budget freed up (e.g. r0 finished) -- both remaining
    # requests now fit under a fresh, larger budget.
    admitted_2 = ctrl.admit_decode_targeted(
        {"r1", "r2"},
        now=11.0,
        max_count=3,
        token_budget=20,
        token_cost=token_cost,
    )
    assert {r.rid for r in admitted_2} == {"r1", "r2"}
    assert ctrl.decode_waiting_count() == 0


def test_admit_decode_targeted_token_budget_none_is_count_only():
    """token_budget=None (the default) must behave exactly like the
    pre-existing count-only admission -- backward compatible for every
    caller that doesn't opt into KV-aware admission."""
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    reqs = []
    for i in range(2):
        r = PDRequestState(
            rid=f"r{i}", arrival_time=0.0, input_length=100000, output_length=100000
        )
        ctrl.on_request_arrival(r, now=0.0)
        ctrl.admit_prefill(capacity=2, now=0.0)
        ctrl.on_prefill_done(
            r, now=1.0, kv_ready_time=ctrl.compute_kv_ready_time(r, 1.0)
        )
        reqs.append(r)
    ctrl.poll_kv_ready(now=10.0)

    admitted = ctrl.admit_decode_targeted({"r0", "r1"}, now=10.0, max_count=2)

    assert {r.rid for r in admitted} == {"r0", "r1"}


def test_decode_forward_advances_one_output_token_until_osl():
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
    assert req.decode_end_time == 10.2


@pytest.mark.parametrize(
    "phase",
    [
        RequestPhase.WAITING_PREFILL,
        RequestPhase.RUNNING_PREFILL,
        RequestPhase.KV_TRANSIT,
        RequestPhase.WAITING_DECODE,
        RequestPhase.RUNNING_DECODE,
    ],
)
def test_terminate_request_is_idempotent_from_every_live_phase(phase):
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    req = PDRequestState(
        rid="terminated", arrival_time=0.0, input_length=4, output_length=8
    )
    ctrl.on_request_arrival(req, now=0.0)
    if phase != RequestPhase.WAITING_PREFILL:
        ctrl.admit_prefill(capacity=1, now=0.0)
    if phase in (
        RequestPhase.KV_TRANSIT,
        RequestPhase.WAITING_DECODE,
        RequestPhase.RUNNING_DECODE,
    ):
        ctrl.on_prefill_done(req, now=1.0, kv_ready_time=2.0)
    if phase in (RequestPhase.WAITING_DECODE, RequestPhase.RUNNING_DECODE):
        ctrl.poll_kv_ready(now=2.0)
    if phase == RequestPhase.RUNNING_DECODE:
        ctrl.admit_decode(capacity=1, now=2.0)

    ctrl.terminate_request(req, now=3.0)
    ctrl.terminate_request(req, now=4.0)

    assert req.phase == RequestPhase.FINISHED
    assert ctrl.prefill_waiting_count() == 0
    assert ctrl.kv_transit_count() == 0
    assert ctrl.decode_waiting_count() == 0
    if phase == RequestPhase.RUNNING_DECODE:
        assert req.decode_end_time == pytest.approx(3.0)


@pytest.mark.parametrize(
    "phase",
    [
        RequestPhase.KV_TRANSIT,
        RequestPhase.WAITING_DECODE,
        RequestPhase.RUNNING_DECODE,
    ],
)
def test_reset_for_retract_reverts_to_waiting_prefill_and_preserves_progress(phase):
    # Reproduces the "0721-Retract" bug scenario: real SGLang evicts an
    # in-flight decode request's KV cache under memory pressure and re-queues
    # the same rid for a fresh prefill. HiSim's PD phase for that rid can be
    # in any of KV_TRANSIT / WAITING_DECODE / RUNNING_DECODE at that instant
    # (its virtual PD clock is decoupled from real SGLang's), and must be
    # reset to WAITING_PREFILL -- but decode progress already produced
    # (decode_step_count / current_past_kv_length) must NOT be rewound, since
    # real SGLang keeps already-generated output tokens across retraction and
    # only rebuilds their KV cache.
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    req = PDRequestState(
        rid="retracted", arrival_time=0.0, input_length=4, output_length=8
    )
    ctrl.on_request_arrival(req, now=0.0)
    ctrl.admit_prefill(capacity=1, now=0.0)
    ctrl.on_prefill_done(req, now=1.0, kv_ready_time=2.0)
    if phase in (RequestPhase.WAITING_DECODE, RequestPhase.RUNNING_DECODE):
        ctrl.poll_kv_ready(now=2.0)
    if phase == RequestPhase.RUNNING_DECODE:
        ctrl.admit_decode(capacity=1, now=2.0)
        ctrl.on_decode_step_done([req], now=2.1)
        ctrl.on_decode_step_done([req], now=2.2)

    progress_before = req.decode_step_count
    kv_before = req.current_past_kv_length

    ctrl.reset_for_retract(req, now=3.0)

    assert req.phase == RequestPhase.WAITING_PREFILL
    assert ctrl.prefill_waiting_count() == 0
    assert ctrl.kv_transit_count() == 0
    assert ctrl.decode_waiting_count() == 0
    # Progress counters must survive the reset unchanged.
    assert req.decode_step_count == progress_before
    assert req.current_past_kv_length == kv_before
    assert req.output_length == 8

    # The request must be re-admittable exactly like a fresh arrival.
    ctrl.on_request_arrival(req, now=3.0)
    admitted = ctrl.admit_prefill(capacity=1, now=3.0)
    assert admitted == [req]
    assert req.phase == RequestPhase.RUNNING_PREFILL


def test_reset_for_retract_rejects_already_finished_request():
    ctrl = make_controller(bw_gbps=1e9, latency_us=0.0, kv_bytes_per_token=1)
    req = PDRequestState(rid="done", arrival_time=0.0, output_length=1)
    ctrl.on_request_arrival(req, now=0.0)
    ctrl.admit_prefill(capacity=1, now=0.0)
    ctrl.on_prefill_done(req, now=1.0, kv_ready_time=1.0)
    ctrl.poll_kv_ready(now=1.0)
    ctrl.admit_decode(capacity=1, now=1.0)
    ctrl.on_decode_step_done([req], now=1.1)
    assert req.phase == RequestPhase.FINISHED

    with pytest.raises(ValueError, match="already-finished"):
        ctrl.reset_for_retract(req, now=2.0)



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
