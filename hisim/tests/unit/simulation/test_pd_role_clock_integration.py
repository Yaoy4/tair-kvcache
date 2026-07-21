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


def _backend(bw_gbps=100.0, latency_us=10.0, decode_queue_mode="single_replica"):
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
        decode_queue_mode=decode_queue_mode,
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
        # mirrors production's `predicted_latency`: the round's reported
        # real-time cost (what time.sleep() would be given in sglang_hook.py).
        self.last_round_latency = 0.0

    def extend(self, reqs):
        """reqs: list of (rid, input_len, output_len, created_time)."""
        arrivals = [ct for (_, _, _, ct) in reqs]
        # Prefill's "engine free" floor is read off the backend's own replica
        # pool (min busy_until across prefill replicas), not a hand-tracked
        # scalar -- see the sglang_hook.py fix this mirrors. self.prefill_clock
        # is still updated below for the p1-02 decoupling assertion, but it is
        # no longer what gates `now_clock`.
        now_clock = prefill_batch_start(
            self.backend.earliest_pool_time("prefill"), arrivals
        )
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
        self.last_round_latency = pd_latency
        # old path: global clock absorbed prefill + the slowest KV transfer.
        kv_extra = max(
            (s.kv_ready_time - (now_clock + pd_latency) for s in states),
            default=0.0,
        )
        self.old_global_clock += pd_latency + max(kv_extra, 0.0)

    def decode(self, batch_rids):
        ctrl = self.backend.controller()
        if self.backend.decode_queue_mode() == "per_replica_queue":
            batch_states = [self.states[rid] for rid in batch_rids if rid in self.states]
            replica_by_rid = self.backend.bind_decode_replicas(batch_states)
            bucket_rids = {}
            for rid in batch_rids:
                state = self.states.get(rid)
                if state is None:
                    continue
                bucket_rids.setdefault(replica_by_rid[rid], []).append(rid)

            # Mirrors the sglang_hook.py fix: compute every bucket's own
            # synced step_start BEFORE polling, then poll once with the MAX
            # across buckets (not the min -- see sglang_hook.py for why the
            # min under-promotes requests on a bucket whose own floor is
            # later, leaving them stuck in KV_TRANSIT and un-admittable).
            bucket_step_start = {}
            for replica_idx, rids in bucket_rids.items():
                step_start = self.backend.decode_replica_time(replica_idx)
                for rid in rids:
                    s = self.states[rid]
                    if s.phase in (
                        RequestPhase.KV_TRANSIT,
                        RequestPhase.WAITING_DECODE,
                    ):
                        step_start = sync_decode_start(step_start, s.kv_ready_time)
                bucket_step_start[replica_idx] = step_start
            ctrl.poll_kv_ready(max(bucket_step_start.values()))

            token_times = {}
            bucket_step_starts = []
            bucket_step_ends = []
            for replica_idx in sorted(bucket_rids, key=self.backend.decode_replica_time):
                step_start = bucket_step_start[replica_idx]
                # Mirrors the sglang_hook.py fix: call the backend's own
                # admit_decode_for_replica (not the raw controller method) so
                # this harness exercises the same capacity accounting AND
                # sticky-binding side effects production relies on. Only
                # offer rids that actually need (re-)admission this round --
                # a continuing RUNNING_DECODE rid was never a candidate for
                # admission, and admit_decode_for_replica un-binds anything
                # it doesn't admit, so including it would spuriously strip
                # its sticky replica assignment every round.
                pending_rids = {
                    rid
                    for rid in bucket_rids[replica_idx]
                    if self.states[rid].phase != RequestPhase.RUNNING_DECODE
                }
                if pending_rids:
                    self.backend.admit_decode_for_replica(
                        replica_idx, pending_rids, step_start
                    )
                states = [
                    self.states[rid]
                    for rid in bucket_rids[replica_idx]
                    if self.states[rid].phase == RequestPhase.RUNNING_DECODE
                ]
                if not states:
                    continue
                pd_latency = decode_batch_latency(self.backend, states, step_start)
                step_end = step_start + pd_latency
                self.backend.on_decode_step_done_batch(states, step_end)
                bucket_step_starts.append(step_start)
                bucket_step_ends.append(step_end)
                for s in states:
                    token_times[s.rid] = step_end
            if not token_times:
                if batch_rids:
                    raise RuntimeError(
                        "PD decode batch contained no admissible request "
                        "state; native and PD capacity/state tracking "
                        "diverged"
                    )
                return
            self.decode_clock = max(bucket_step_ends)
            self.last_decode_step_end = self.decode_clock
            self.old_global_clock += max(bucket_step_ends) - min(bucket_step_starts)
            # Mirrors the sglang_hook.py fix: the round's reported real-time
            # cost is the SLOWEST bucket's OWN step latency (its own
            # step_end - step_start), never a span mixing one bucket's start
            # with a different bucket's end -- buckets are independent
            # replica clocks and must never be forced to "catch up" to one
            # another just because they happen to share one native-scheduler
            # round.
            self.last_round_latency = max(
                end - start
                for start, end in zip(bucket_step_starts, bucket_step_ends)
            )
            for rid, token_time in token_times.items():
                self.gen[rid].append(token_time - self.last_event[rid])
                self.last_event[rid] = token_time
            return

        # Decode's "engine free" floor is likewise read off the backend
        # (mode-aware: pool min under per_replica_queue, pool max otherwise --
        # see BackendA.earliest_pool_time), not a hand-tracked scalar. This
        # mirrors the sglang_hook.py fix. self.decode_clock is still updated
        # below for the p1-02 decoupling assertion, but no longer gates
        # `step_start`.
        step_start = self.backend.earliest_pool_time("decode")
        for rid in batch_rids:
            s = self.states.get(rid)
            if s is not None and s.phase in (
                RequestPhase.KV_TRANSIT,
                RequestPhase.WAITING_DECODE,
            ):
                step_start = sync_decode_start(step_start, s.kv_ready_time)
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
        self.last_round_latency = pd_latency
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


# ---------------------------------------------------------------------------
# Regression: with prefill_replicas > 1, an idle replica must not wait for a
# different, still-busy replica. Before this fix, the prefill floor was a
# hand-tracked scalar that remembered only the specific replica the previous
# extend batch landed on, so a second concurrently-arriving request was
# forced to wait for THAT replica even when a different one had been idle
# the whole time.
# ---------------------------------------------------------------------------
def test_concurrent_extend_batches_use_idle_replica_not_busy_one():
    d = HookDriver(_backend())  # 2 prefill replicas
    # A: a big request that occupies one prefill replica for a while.
    d.extend([("a", 20_000, 1, 0.0)])
    a_end = d.states["a"].prefill_end_time
    assert a_end > 0.0

    # B: a tiny, independent request that also arrived at t=0. SGLang's
    # serialized loop happens to hand it to the hook as the *next* extend
    # iteration, but the second prefill replica has been idle since t=0 and
    # should serve it immediately, concurrently with A still prefilling on
    # the first replica.
    d.extend([("b", 10, 1, 0.0)])
    b_start = d.states["b"].prefill_start_time

    # B must start at/near t=0 on the idle replica, NOT be pushed back to
    # wait for A's (unrelated) replica to free up.
    assert b_start == pytest.approx(0.0, abs=1e-9)
    assert b_start < a_end


# ---------------------------------------------------------------------------
# Regression: under decode_queue_mode="per_replica_queue", decode replicas
# hold disjoint, independently-progressing requests (sticky rid->replica
# assignment). A heavily-loaded replica must not hold back a different,
# lightly-loaded replica's own continuing requests -- even though every
# SGLang decode iteration bundles all running requests into one shared call.
# ---------------------------------------------------------------------------
def test_concurrent_decode_replicas_use_own_busy_until_not_slowest():
    d = HookDriver(_backend(decode_queue_mode="per_replica_queue"))
    # "a": heavy request (large KV) and "b": light request (tiny KV) become
    # decode-ready together and get sticky round-robin assigned to DIFFERENT
    # decode replicas.
    d.extend([("a", 100_000, 10, 0.0)])
    d.extend([("b", 10, 10, 0.0)])
    d.decode(["a", "b"])  # tick 1: bundled together, as SGLang really does it

    a_idx = d.backend._decode_replica_by_rid["a"]
    b_idx = d.backend._decode_replica_by_rid["b"]
    assert a_idx != b_idx
    a_busy_after_tick1 = d.backend._decode_pool.busy_until[a_idx]
    b_busy_after_tick1 = d.backend._decode_pool.busy_until[b_idx]
    # "a" is genuinely the much slower replica after tick 1.
    assert a_busy_after_tick1 > b_busy_after_tick1

    d.decode(["a", "b"])  # tick 2: still bundled together (both continuing)
    b_busy_after_tick2 = d.backend._decode_pool.busy_until[b_idx]

    # "b"'s own second step must be bound by ITS OWN replica's prior busy_until,
    # not forced to wait for "a"'s much slower replica to finish tick 1.
    assert b_busy_after_tick2 < a_busy_after_tick1


def test_new_decode_waiter_on_other_replica_does_not_block_running_request():
    d = HookDriver(
        _backend(bw_gbps=1.0, decode_queue_mode="per_replica_queue")
    )
    d.extend([("a", 100, 5, 0.0)])
    d.decode(["a"])
    d.decode(["a"])
    a_itl_baseline = d.gen["a"][-1]

    d.extend([("b", 8_000, 2, 0.0)])
    assert d.states["b"].kv_ready_time > d.last_event["a"]

    d.decode(["a", "b"])

    a_idx = d.backend._decode_replica_by_rid["a"]
    b_idx = d.backend._decode_replica_by_rid["b"]
    assert a_idx != b_idx
    assert d.gen["a"][-1] == pytest.approx(a_itl_baseline, abs=1e-6)
    # "b" must actually be admitted in this SAME decode() call (not silently
    # dropped) -- an explicit check, rather than relying on the TypeError a
    # still-None decode_start_time would otherwise raise below.
    assert d.states["b"].phase == RequestPhase.RUNNING_DECODE
    assert d.states["b"].decode_start_time is not None
    assert d.last_event["a"] < d.states["b"].decode_start_time


# ---------------------------------------------------------------------------
# Regression: reproduces the exact live-1P2D crash --
# "RuntimeError: PD decode batch contained no admissible request state;
# native and PD capacity/state tracking diverged" -- raised spuriously by
# sglang_hook.py even though the native SGLang batch was non-empty.
#
# Root cause: with decode_queue_mode="per_replica_queue" and 2+ decode
# replicas whose busy_until has diverged (one replica previously did real
# work, the other is still pristine), the hook polled KV-readiness ONCE per
# scheduler iteration using the MINIMUM raw replica busy_until across all
# active buckets, then computed each bucket's own (correctly synced, later)
# admission floor separately -- but never re-polled at that later floor.
# Any request whose kv_ready_time fell strictly between the minimum and its
# own bucket's floor was left stuck in KV_TRANSIT and could not be admitted
# by admit_decode_for_replica, which only promotes requests already marked
# WAITING_DECODE by a prior poll. When every bucket hits this, token_times
# stays empty for a round where the native scheduler batch is non-empty,
# raising the divergence error. This reproduces it with exactly 2 fresh,
# never-before-decoded requests landing on 2 replicas with different
# busy_until, and asserts BOTH are admitted in the same round instead.
# ---------------------------------------------------------------------------
def test_decode_admits_both_buckets_when_busy_until_diverges_from_kv_ready():
    d = HookDriver(_backend(bw_gbps=1.0, decode_queue_mode="per_replica_queue"))

    # Warm up replica 0 (first-ever bind ties go to the lower index) with a
    # short-lived request so it ends with a positive busy_until, then
    # finishes and frees its slot. Replica 1 stays pristine at busy_until ==
    # 0.0, so the two replicas now have genuinely different raw floors.
    d.extend([("warm", 50, 1, 0.0)])
    d.decode(["warm"])
    warm_idx = d.backend._decode_replica_by_rid.get("warm")
    assert warm_idx is None  # popped on FINISHED
    assert d.states["warm"].phase == RequestPhase.FINISHED
    busy_until = list(d.backend._decode_pool.busy_until)
    assert max(busy_until) > 0.0 and min(busy_until) == 0.0

    # Two brand-new requests, never decoded before: sticky assignment prefers
    # the least-loaded replica first (the pristine one, busy_until == 0.0),
    # then the other (the warmed-up one, busy_until > 0.0). Both are still
    # KV_TRANSIT once bound -- neither has a pre-existing RUNNING_DECODE
    # state to fall back on, so if either bucket fails to admit this round,
    # its states list is empty.
    d.extend([("p", 50, 5, 0.0), ("q", 4_000, 5, 0.0)])
    assert d.states["p"].kv_ready_time > 0.0
    assert d.states["q"].kv_ready_time > 0.0

    # Must not raise "PD decode batch contained no admissible request state".
    d.decode(["p", "q"])

    p_idx = d.backend._decode_replica_by_rid["p"]
    q_idx = d.backend._decode_replica_by_rid["q"]
    assert p_idx != q_idx
    assert d.states["p"].phase == RequestPhase.RUNNING_DECODE
    assert d.states["q"].phase == RequestPhase.RUNNING_DECODE
    assert d.states["p"].decode_start_time is not None
    assert d.states["q"].decode_start_time is not None
    assert d.gen["p"] and d.gen["q"]


# ---------------------------------------------------------------------------
# Regression: admit_decode_for_replica un-binds any rid it does not admit
# this round -- correct for a genuine capacity rejection, but a request
# already RUNNING_DECODE (continuing from a prior round, simply bundled into
# this same native decode batch alongside other replicas' requests) is never
# a candidate for (re-)admission in the first place. If the caller offers
# such a rid to admit_decode_for_replica anyway, its sticky replica binding
# gets silently popped even though the request is still very much running --
# and on the very next round, bind_decode_replicas treats it as unbound and
# may reassign it to a DIFFERENT replica, which has no physical meaning (a
# request's KV cache lives on one specific replica; it cannot silently hop
# to another mid-generation) and would corrupt that replica's busy_until /
# running-count bookkeeping. This asserts a continuing request's replica
# assignment is stable across many rounds sharing a batch with unrelated,
# newly-admitted requests on other replicas.
# ---------------------------------------------------------------------------
def test_running_decode_request_keeps_sticky_replica_across_rounds():
    d = HookDriver(_backend(decode_queue_mode="per_replica_queue"))

    d.extend([("a", 100_000, 20, 0.0)])
    d.decode(["a"])  # tick 1: "a" admitted, bound to whichever replica is idle.
    a_idx_after_tick1 = d.backend._decode_replica_by_rid["a"]
    assert d.states["a"].phase == RequestPhase.RUNNING_DECODE

    d.decode(["a"])  # tick 2: "a" alone, continuing -- no new admission at all.
    assert d.backend._decode_replica_by_rid.get("a") == a_idx_after_tick1

    # tick 3: "a" (still running, from replica a_idx_after_tick1) is bundled
    # together with a brand-new request "b" that lands on the OTHER replica.
    # Before the fix, offering "a" to admit_decode_for_replica here (it is
    # never actually admitted, since it is not WAITING_DECODE) would pop
    # "a"'s binding as a side effect.
    d.extend([("b", 50, 5, 0.0)])
    d.decode(["a", "b"])
    assert d.backend._decode_replica_by_rid["a"] == a_idx_after_tick1
    assert d.states["a"].phase == RequestPhase.RUNNING_DECODE

    # tick 4: repeat once more to confirm the binding survives a second
    # shared round, not just a single grace round.
    d.decode(["a", "b"])
    assert d.backend._decode_replica_by_rid["a"] == a_idx_after_tick1
    assert d.states["a"].phase == RequestPhase.RUNNING_DECODE


# ---------------------------------------------------------------------------
# Regression: reproduces the live-server latency-inflation bug found when
# re-validating the two crash fixes above at scale -- under
# decode_queue_mode="per_replica_queue", E2E/TTFT/TPOT got *worse* the more
# decode replicas were added (e.g. a live 1P16D run showed +7434% TTFT vs the
# 1P1D baseline), the opposite of what adding decode capacity should ever do.
#
# Root cause: sglang_hook.py reported each decode round's real-time cost
# (predicted_latency, the value handed to time.sleep()) as
# max(bucket_step_ends) - min(bucket_step_starts) -- a SPAN across every
# active bucket's own local clock. Buckets are independent replica clocks by
# design (a lightly-loaded bucket legitimately runs ahead of a heavily-loaded
# one -- see test_concurrent_decode_replicas_use_own_busy_until_not_slowest
# above), so min(bucket_step_starts) can be a bucket that has been genuinely
# idle (not "behind") for a long time. The old formula mistook that idle gap
# for extra round latency and re-charged it via time.sleep() on EVERY
# subsequent round; since one request can span up to output_len decode
# rounds, even a small per-round overcount compounds into multi-second
# E2E/TTFT inflation -- worse the more decode replicas exist (more chances
# for an idle/busy split).
# ---------------------------------------------------------------------------
def test_decode_round_latency_ignores_idle_bucket_clock_skew():
    d = HookDriver(_backend(decode_queue_mode="per_replica_queue"))
    # "a": heavy request, "b": light request -- sticky-bound to different
    # replicas (mirrors test_concurrent_decode_replicas_use_own_busy_until_
    # not_slowest's setup, which already proves these land on different
    # buckets). "a" carries a much larger past_kv_length than "b" every
    # round, so its own per-round latency is consistently bigger and its
    # bucket clock keeps pulling further ahead of "b"'s round after round --
    # mirroring how the real bug compounds over many decode rounds instead
    # of just one.
    d.extend([("a", 100_000, 20, 0.0)])
    d.extend([("b", 10, 20, 0.0)])
    for _ in range(5):
        d.decode(["a", "b"])  # ticks 1-5: let the bucket clocks diverge.

    a_idx = d.backend._decode_replica_by_rid["a"]
    b_idx = d.backend._decode_replica_by_rid["b"]
    a_step_start = d.backend._decode_pool.busy_until[a_idx]
    b_step_start = d.backend._decode_pool.busy_until[b_idx]
    # Sanity check (also asserted by the sibling test above): "a"'s replica
    # clock is genuinely far ahead of "b"'s after several shared rounds.
    assert a_step_start > b_step_start

    d.decode(["a", "b"])  # tick 6: both continuing, no new admission at all.
    a_step_end = d.backend._decode_pool.busy_until[a_idx]
    b_step_end = d.backend._decode_pool.busy_until[b_idx]
    a_pd_latency = a_step_end - a_step_start
    b_pd_latency = b_step_end - b_step_start

    # What the OLD buggy formula would have reported for this same round:
    # a cross-bucket SPAN that bakes in the entire accumulated clock gap
    # between "a" and "b" on top of the round's own latency.
    old_buggy_latency = max(a_step_end, b_step_end) - min(a_step_start, b_step_start)

    # The fixed formula must equal the SLOWEST bucket's OWN tick-6 step
    # latency (never mixing one bucket's start with the other's end)...
    assert d.last_round_latency == pytest.approx(max(a_pd_latency, b_pd_latency))
    # ...which must be strictly less than what the old buggy formula would
    # have reported, since the gap accumulated over ticks 1-5 is nonzero.
    assert d.last_round_latency < old_buggy_latency
