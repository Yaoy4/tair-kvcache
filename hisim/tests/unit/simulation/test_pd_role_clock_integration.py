"""Integration tests for the P1 dual-clock decoupling, driving a *real*
BackendA through the exact role-clock orchestration the SGLang hook performs.

These tests are SGLang-free: they replicate the hook's prefill/decode driver
loop (the same pd_runtime + pd_timeline calls in the same order) against a real
BackendA built via build_disagg, and assert the end-to-end timeline properties
the hook closure is responsible for:

- p1-02: a prefill (extend) batch advances the prefill role clock but NOT the
         decode role clock (the core decoupling invariant).
- p1-03: decode start syncs to KV-ready at the prefill->decode handoff.
- p1-04: a request's KV transfer does not freeze another request's in-flight
         decode (ITL stays == decode step latency, vs the old single-clock path
         that would add prefill+KV to it).
- p1-07: makespan (max last_event_time) == max decode_end_time, and is strictly
         smaller than the old single-global-clock makespan for overlapping work.
"""
import pytest

from hisim.spec import DataType, ModelInfo
from hisim.simulation.types import SchedulerConfig
from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.pd_runtime import (
    admit_prefill_batch_latency,
    decode_batch_latency,
    finalize_prefill_batch,
)
from hisim.simulation.pd_timeline import prefill_batch_start, sync_decode_start


# ---------------------------------------------------------------------------
# Real-backend fixtures (mirrors test_pd_backend_a.py)
# ---------------------------------------------------------------------------
def _model():
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="t",
    )


def _base():
    return SchedulerConfig(
        model=_model(),
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )


class _StubPredictor:
    _PREFILL_US_PER_TOK = {"fast": 0.1, "slow": 1.0}
    _DECODE_US_PER_STEP = {"fast": 5.0, "slow": 50.0}

    def __init__(self, model, hw, config, **kwargs):
        self._prefill_us = self._PREFILL_US_PER_TOK[hw.name]
        self._decode_us = self._DECODE_US_PER_STEP[hw.name]

    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        return batch_tokens * self._prefill_us * 1e-6

    def predict_decode_seconds(self, batch_size: int, past_kv_length=0) -> float:
        if isinstance(past_kv_length, int):
            pkv_total = past_kv_length * batch_size
        else:
            pkv_total = sum(int(x) for x in past_kv_length)
        return (self._decode_us * batch_size + 0.001 * pkv_total) * 1e-6


class _StubHW:
    def __init__(self, name):
        self.name = name


def _backend(bw_gbps=100.0, latency_us=10.0):
    cfg = DisaggConfig(
        enabled=True,
        backend="single_process",
        prefill=RolePredictorConfig(
            device_name="fast", tp_size=1, replicas=2, max_running_per_replica=8
        ),
        decode=RolePredictorConfig(
            device_name="fast", tp_size=1, replicas=2, max_running_per_replica=64
        ),
        kv_transfer=BandwidthTransferConfig(bw_gbps=bw_gbps, latency_us=latency_us),
        decode_queue_mode="single_replica",
    )
    bundle = build_disagg(
        model=_model(),
        base_sched_config=_base(),
        disagg_config=cfg,
        predictor_factory=_StubPredictor,
        hw_factory=lambda name: _StubHW(name),
    )
    return BackendA(bundle)


# ---------------------------------------------------------------------------
# A faithful, SGLang-free replica of the hook's role-clock driver.
# ---------------------------------------------------------------------------
class HookDriver:
    """Replays exactly what wrapped_run_batch / wrapped_process_batch_result do
    for the PD path, but against a plain BackendA and dict bookkeeping."""

    def __init__(self, backend):
        self.backend = backend
        self.states = {}  # rid -> PDRequestState
        self.prefill_clock = 0.0
        self.decode_clock = 0.0
        self.last_decode_step_end = 0.0
        # per-rid token-recording mirror of process_batch_result
        self.last_event = {}  # rid -> float
        self.arrival = {}  # rid -> float
        self.gen = {}  # rid -> list[float]
        # old single-global-clock makespan, for the regression comparison
        self.old_global_clock = 0.0

    def extend(self, reqs):
        """reqs: list of (rid, input_len, output_len, created_time)."""
        arrivals = [ct for (_, _, _, ct) in reqs]
        now_clock = prefill_batch_start(self.prefill_clock, arrivals)
        states = []
        for (rid, ilen, olen, ct) in reqs:
            s = self.states.get(rid)
            if s is None:
                s = PDRequestState(
                    rid=rid,
                    arrival_time=ct,
                    phase=RequestPhase.WAITING_PREFILL,
                    input_length=ilen,
                    output_length=olen,
                )
                self.states[rid] = s
                self.arrival[rid] = ct
                self.last_event[rid] = ct  # mirrors RequestStats.last_event_time
                self.gen[rid] = []
            states.append(s)
        pd_latency = admit_prefill_batch_latency(self.backend, states, now_clock)
        finalize_prefill_batch(self.backend, states, now_clock + pd_latency)
        self.prefill_clock = now_clock + pd_latency
        # old path: global clock absorbed prefill + the slowest KV transfer.
        kv_extra = max(
            (s.kv_ready_time - (now_clock + pd_latency) for s in states),
            default=0.0,
        )
        self.old_global_clock += pd_latency + max(kv_extra, 0.0)

    def decode(self, batch_rids):
        step_start = self.decode_clock
        for rid in batch_rids:
            s = self.states.get(rid)
            if s is not None and s.phase in (
                RequestPhase.KV_TRANSIT,
                RequestPhase.WAITING_DECODE,
            ):
                step_start = sync_decode_start(step_start, s.kv_ready_time)
        ctrl = self.backend.controller()
        ctrl.poll_kv_ready(step_start)
        ctrl.admit_decode_targeted(set(batch_rids), step_start)
        states = [
            self.states[rid]
            for rid in batch_rids
            if self.states[rid].phase == RequestPhase.RUNNING_DECODE
        ]
        if not states:
            return
        pd_latency = decode_batch_latency(self.backend, states, step_start)
        step_end = step_start + pd_latency
        self.backend.on_decode_step_done_batch(states, step_end)
        self.decode_clock = step_end
        self.last_decode_step_end = step_end
        self.old_global_clock += pd_latency
        # process_batch_result mirror: record one token per decoding rid.
        for s in states:
            self.gen[s.rid].append(self.last_decode_step_end - self.last_event[s.rid])
            self.last_event[s.rid] = self.last_decode_step_end


# ---------------------------------------------------------------------------
# p1-02: prefill advance does not move the decode clock
# ---------------------------------------------------------------------------
def test_prefill_batch_does_not_advance_decode_clock():
    d = HookDriver(_backend())
    assert d.prefill_clock == 0.0 and d.decode_clock == 0.0
    d.extend([("a", 1000, 2, 0.0)])
    assert d.prefill_clock > 0.0  # prefill clock advanced by the extend batch
    assert d.decode_clock == 0.0  # decode clock untouched (decoupling invariant)


# ---------------------------------------------------------------------------
# p1-03: decode start syncs to KV-ready at the handoff
# ---------------------------------------------------------------------------
def test_decode_start_syncs_to_kv_ready():
    d = HookDriver(_backend(bw_gbps=1.0))  # slow KV -> non-trivial kv_ready
    d.extend([("a", 4000, 2, 0.0)])
    kv_ready = d.states["a"].kv_ready_time
    assert kv_ready > 0.0
    # decode clock is still 0 (< kv_ready); the first decode step must wait.
    d.decode(["a"])
    assert d.states["a"].decode_start_time == pytest.approx(kv_ready)
    assert d.decode_clock == pytest.approx(kv_ready + (d.gen["a"][0] - kv_ready))


# ---------------------------------------------------------------------------
# p1-04: one request's KV transfer must not freeze another's decode
# ---------------------------------------------------------------------------
def test_kv_transfer_of_a_does_not_inflate_b_itl():
    d = HookDriver(_backend(bw_gbps=1.0))  # deliberately slow KV transfer
    # B is already decoding; A then does a big prefill + slow KV transfer.
    d.extend([("b", 100, 5, 0.0)])
    d.decode(["b"])  # B first token (TTFT)
    d.decode(["b"])  # B second token -> establishes a clean ITL baseline
    itl_baseline = d.gen["b"][-1]

    # A arrives: large prefill and a slow KV transfer happen "concurrently".
    d.extend([("a", 8000, 2, 0.0)])
    assert d.states["a"].kv_ready_time > d.decode_clock  # KV genuinely in flight

    # B keeps decoding; its ITL must equal the decode step latency, NOT include
    # A's prefill + KV transfer. (The stub predictor adds a legitimate ~1e-9s
    # per-step past-KV cost, far below A's prefill ~8e-4s / KV transfer; an abs
    # 1e-6 tolerance proves no prefill/KV leaked into B's ITL.)
    d.decode(["b"])
    itl_after = d.gen["b"][-1]
    assert itl_after == pytest.approx(itl_baseline, abs=1e-6)


# ---------------------------------------------------------------------------
# p1-05/06: single-request ITL/TTFT/E2E on the decode role clock
# ---------------------------------------------------------------------------
def test_single_request_gen_latencies_telescope_to_e2e():
    d = HookDriver(_backend())
    d.extend([("a", 512, 3, 0.0)])
    d.decode(["a"])
    d.decode(["a"])
    d.decode(["a"])
    gen = d.gen["a"]
    assert len(gen) == 3
    # TTFT (gen[0]) includes prefill + KV + first decode step; ITLs are equal
    # up to the stub predictor's tiny (~1e-9s) per-step past-KV growth.
    assert gen[0] > gen[1]
    assert gen[1] == pytest.approx(gen[2], abs=1e-6)
    # E2E telescopes to decode_end - arrival.
    e2e = sum(gen)
    assert e2e == pytest.approx(
        d.states["a"].decode_end_time - d.states["a"].arrival_time
    )


# ---------------------------------------------------------------------------
# p1-07: makespan reflects overlap and beats the old single-clock makespan
# ---------------------------------------------------------------------------
def test_overlapping_makespan_is_decode_end_and_below_old_clock():
    d = HookDriver(_backend())
    # Two requests with interleaved prefill/decode (overlapping roles).
    d.extend([("a", 1024, 3, 0.0)])
    d.decode(["a"])               # A token 1
    d.extend([("b", 1024, 3, 0.0)])  # B prefill while A is decoding
    d.decode(["a", "b"])          # A token 2 + B token 1
    d.decode(["a", "b"])          # A token 3 + B token 2
    d.decode(["b"])               # B token 3

    makespan = max(d.last_event.values())
    decode_end = max(
        s.decode_end_time for s in d.states.values() if s.decode_end_time is not None
    )
    # p1-07: throughput makespan == true decode makespan.
    assert makespan == pytest.approx(decode_end)
    # ... and strictly below the old serialised global-clock makespan.
    assert makespan < d.old_global_clock
