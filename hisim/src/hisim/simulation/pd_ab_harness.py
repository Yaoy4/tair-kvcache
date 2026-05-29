"""Phase 6b — A/B equivalence harness skeleton.

Drives an identical workload through BackendA (single-process) and BackendB
(two-process) via the shared ``PDBackendProtocol`` and returns per-request
+ aggregate results suitable for the equivalence checker (Phase 6c).

Design rules:

* Pure ``PDBackendProtocol`` calls — no SGLang imports, no hook-specific
  paths. Both backends see the *same* event-driven driver, so any
  divergence in results is a backend bug, not a harness-ordering artifact
  (see `hisim/docs/pd_ab_equivalence_spec.md` invariants).
* The driver is purely sequential: it schedules each request through
  ``arrive → prefill_done → kv_ready → decode_step_done`` via a min-heap
  of virtual-time events. Concurrency comes from the backend's replica
  pool, not from harness-level threading.
* BackendB lifecycle (start + shutdown) is owned here so callers don't
  leak spawned workers if a test raises.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from hisim.spec.model import ModelInfo
from hisim.simulation.pd_backend_protocol import PDBackendProtocol
from hisim.simulation.pd_config import DisaggConfig
from hisim.simulation.pd_runtime import (
    build_pd_backend,
    shutdown_pd_backend,
    start_pd_backend,
)
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.types import SchedulerConfig


# ---------------------------------------------------------------------------
# Input / output dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadRequest:
    """Backend-agnostic description of a single PD request."""

    rid: str
    arrival_time: float
    input_length: int
    output_length: int

    def to_state(self) -> PDRequestState:
        return PDRequestState(
            rid=self.rid,
            arrival_time=self.arrival_time,
            phase=RequestPhase.WAITING_PREFILL,
            input_length=int(self.input_length),
            output_length=int(self.output_length),
        )


@dataclass
class RequestResult:
    """All per-request timestamps the equivalence checker needs.

    Every time field is a virtual-time second. ``None`` means the request
    never reached that phase (used by the checker to flag a STRUCT
    invariant violation).
    """

    rid: str
    arrival_time: float
    input_length: int
    output_length: int
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    kv_ready_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    decode_end_time: Optional[float] = None
    decode_step_count: int = 0
    finished: bool = False

    @property
    def ttft(self) -> Optional[float]:
        """Time-to-first-token = prefill end - arrival (HiSim convention)."""
        if self.prefill_end_time is None:
            return None
        return self.prefill_end_time - self.arrival_time

    @property
    def e2e(self) -> Optional[float]:
        if self.decode_end_time is None:
            return None
        return self.decode_end_time - self.arrival_time

    @property
    def kv_transfer_time(self) -> Optional[float]:
        if self.kv_ready_time is None or self.prefill_end_time is None:
            return None
        return self.kv_ready_time - self.prefill_end_time

    @property
    def prefill_queue_wait(self) -> Optional[float]:
        if self.prefill_start_time is None:
            return None
        return self.prefill_start_time - self.arrival_time

    @property
    def decode_queue_wait(self) -> Optional[float]:
        if self.decode_start_time is None or self.kv_ready_time is None:
            return None
        return self.decode_start_time - self.kv_ready_time


@dataclass
class RunResult:
    """Output of one full backend run over a workload."""

    backend_label: str
    per_request: List[RequestResult] = field(default_factory=list)
    final_clock: float = 0.0

    def by_rid(self) -> dict:
        return {r.rid: r for r in self.per_request}


# ---------------------------------------------------------------------------
# Driver: event-loop walk through PDBackendProtocol.
# ---------------------------------------------------------------------------


def run_workload(
    backend: PDBackendProtocol,
    workload: Sequence[WorkloadRequest],
    *,
    backend_label: str = "",
) -> RunResult:
    """Drive ``workload`` through ``backend`` and capture per-request results.

    Event kinds in the min-heap:

    * ``"arrive"``        — admit prefill for one request.
    * ``"prefill_done"``  — finalize prefill, schedule the KV-ready event.
    * ``"kv_ready"``      — promote to WAITING_DECODE, admit first decode step.
    * ``"decode_step_done"`` — on_decode_step_done; if more output tokens
      remain, admit the next decode step; otherwise mark finished and
      record ``decode_end_time``.

    Order rules: events are popped by ``(time, seq)`` so ties break in
    submission order. ``seq`` is per-call monotonic so both backends see
    identical orderings for identical inputs.
    """
    # Build mutable PDRequestStates (the harness owns them; backends mutate).
    states = {w.rid: w.to_state() for w in workload}
    results = {
        w.rid: RequestResult(
            rid=w.rid,
            arrival_time=float(w.arrival_time),
            input_length=int(w.input_length),
            output_length=int(w.output_length),
        )
        for w in workload
    }

    heap: list = []
    seq = 0

    def push(t: float, kind: str, rid: str) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(heap, (float(t), seq, kind, rid))

    # Seed: every request arrives at its declared arrival_time.
    for w in workload:
        push(w.arrival_time, "arrive", w.rid)

    final_clock = 0.0

    while heap:
        now, _, kind, rid = heapq.heappop(heap)
        if now > final_clock:
            final_clock = now
        state = states[rid]
        result = results[rid]

        if kind == "arrive":
            _, end_t = backend.try_admit_prefill(state, now)
            # The controller updates prefill_start_time on admit_prefill.
            result.prefill_start_time = state.prefill_start_time
            push(end_t, "prefill_done", rid)

        elif kind == "prefill_done":
            kv_ready = backend.compute_kv_ready_time(state, now)
            backend.on_prefill_done(state, now, kv_ready)
            result.prefill_end_time = state.prefill_end_time
            result.kv_ready_time = state.kv_ready_time
            push(kv_ready, "kv_ready", rid)

        elif kind == "kv_ready":
            backend.advance_to_kv_ready(state, now)
            # advance_to_kv_ready moves req from KV_TRANSIT to WAITING_DECODE
            # via the controller. We then explicitly admit a decode slot so
            # decode_start_time is set, mirroring sglang_hook's flow.
            ctrl = backend.controller()
            admitted = ctrl.admit_decode(capacity=1, now=now)
            if not admitted or admitted[0].rid != rid:
                # If the controller didn't admit this rid (e.g., capacity
                # already saturated by an earlier request), defer one step.
                # This shouldn't happen for the W1-style workloads but is a
                # safety valve; reschedule the kv_ready event slightly later.
                push(now, "kv_ready", rid)
                continue
            result.decode_start_time = state.decode_start_time
            # Run the first decode step.
            _, end_t = backend.try_admit_decode_step(state, now)
            push(end_t, "decode_step_done", rid)

        elif kind == "decode_step_done":
            backend.on_decode_step_done(state, now)
            if state.phase == RequestPhase.FINISHED:
                result.decode_end_time = now
                result.decode_step_count = state.decode_step_count
                result.finished = True
            else:
                _, end_t = backend.try_admit_decode_step(state, now)
                push(end_t, "decode_step_done", rid)

        else:  # pragma: no cover — defensive, kinds are produced internally
            raise AssertionError(f"unknown event kind: {kind!r}")

    return RunResult(
        backend_label=backend_label,
        per_request=[results[w.rid] for w in workload],
        final_clock=final_clock,
    )


# ---------------------------------------------------------------------------
# Top-level A/B runner.
# ---------------------------------------------------------------------------


def run_ab(
    *,
    model: ModelInfo,
    base_sched_config: SchedulerConfig,
    disagg_config: DisaggConfig,
    workload: Sequence[WorkloadRequest],
    predictor_factory: Optional[Callable] = None,
    role_factory: Optional[Callable] = None,
    hw_factory: Optional[Callable] = None,
    mp_context=None,
) -> "tuple[RunResult, RunResult]":
    """Build BackendA + BackendB from the same config and drive both through
    ``workload``. Returns ``(result_a, result_b)``.

    BackendB lifecycle is owned here: it is started before use and
    shutdown in a ``finally`` so spawned workers can't leak if the
    BackendA run raises. The ``disagg_config.backend`` field is overridden
    per side; the caller's original value is irrelevant.
    """
    # ---- BackendA side ----
    cfg_a = _with_backend(disagg_config, "single_process")
    backend_a = build_pd_backend(
        model=model,
        base_sched_config=base_sched_config,
        disagg_config=cfg_a,
        predictor_factory=predictor_factory,
        hw_factory=hw_factory,
    )
    result_a = run_workload(backend_a, workload, backend_label="A")

    # ---- BackendB side ----
    cfg_b = _with_backend(disagg_config, "two_process")
    backend_b = build_pd_backend(
        model=model,
        base_sched_config=base_sched_config,
        disagg_config=cfg_b,
        role_factory=role_factory,
        hw_factory=hw_factory,
        mp_context=mp_context,
    )
    start_pd_backend(backend_b)
    try:
        result_b = run_workload(backend_b, workload, backend_label="B")
    finally:
        shutdown_pd_backend(backend_b)

    return result_a, result_b


def _with_backend(cfg: DisaggConfig, backend: str) -> DisaggConfig:
    """Return a shallow copy of ``cfg`` with ``backend`` overridden.

    Uses ``object.__setattr__`` so the frozen guard in DisaggConfig
    (if any) doesn't reject the rebind. Validation happens at use time.
    """
    from dataclasses import replace

    try:
        return replace(cfg, backend=backend)
    except Exception:
        # Defensive: if DisaggConfig isn't a dataclass for some reason,
        # mutate-via-setattr on a shallow copy.
        import copy

        clone = copy.copy(cfg)
        object.__setattr__(clone, "backend", backend)
        return clone
