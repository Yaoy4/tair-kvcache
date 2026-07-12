"""Pure, SGLang-free timeline helpers for PD dual-clock decoupling (P1).

Pure semantic layer — must NOT import SGLang or backend-specific code
(mirrors ``pd_metrics``). These functions encode the *role-clock* arithmetic
so the (hard-to-unit-test) SGLang hook closure stays a thin orchestration
layer over logic that is fully covered by fast, deterministic unit tests.

Background
----------
The single SGLang loop drives two logical pools. Their virtual availability
is held by backend-owned per-replica ``busy_until`` values rather than scalar
hook clocks. Prefill starts are anchored to request arrival; decode starts are
additionally gated by each request's ``kv_ready_time``.
"""
from __future__ import annotations

from typing import Iterable, Optional


def sync_decode_start(decode_clock: float, kv_ready_time: Optional[float]) -> float:
    """Earliest time a request's decode may begin.

    A decode step can start no earlier than (a) the decode engine being free
    (``decode_clock``) and (b) the request's KV cache having arrived on the
    decode node (``kv_ready_time``). This is the prefill->decode handoff
    synchronisation point: when KV is still in flight the decode clock *jumps*
    forward to ``kv_ready_time``; otherwise the engine-free time dominates.

    ``kv_ready_time is None`` means "no KV gate known", so the engine-free time
    is returned unchanged.
    """
    if kv_ready_time is None:
        return decode_clock
    return max(decode_clock, kv_ready_time)


def prefill_batch_start(
    prefill_clock: float, arrivals: Iterable[Optional[float]]
) -> float:
    """Earliest time a prefill (extend) batch may begin.

    The prefill engine starts a batch no earlier than it is free
    (``prefill_clock``) and all of the batch's requests have arrived. Arrivals
    that are ``None`` or negative (unknown / unset ``created_time``) are
    ignored so the prefill clock self-anchors to whatever real arrival times
    are available, falling back to its own value when none are.
    """
    start = prefill_clock
    for arrival in arrivals:
        if arrival is not None and arrival >= 0.0:
            start = max(start, arrival)
    return start


def prefill_queue_baseline(
    arrival_time: float, queue_start_time: Optional[float]
) -> float:
    """Return the server-side baseline for prefill queue wait.

    ``arrival_time`` remains the end-to-end request baseline used by TTFT and
    E2E.  Queue wait starts when the server enqueues the request; missing or
    invalid queue timestamps fall back to arrival for older traces.
    """
    if queue_start_time is None or queue_start_time < 0.0:
        return arrival_time
    return queue_start_time


def prefill_admission_baseline(
    arrival_time: float, queue_end_time: Optional[float]
) -> float:
    """Earliest legal prefill start after native scheduler admission."""
    if queue_end_time is None or queue_end_time < 0.0:
        return arrival_time
    return queue_end_time


def closed_loop_first_token_latency(
    token_time: float,
    *,
    prefill_start_time: Optional[float],
    kv_ready_time: Optional[float],
    decode_start_time: Optional[float],
) -> Optional[float]:
    """Service-only TTFT for the first decode token in closed-loop emulation.

    Under ``request_rate=inf`` + scheduler cap ``max_running_requests=C``, the
    simulator can hold requests in a synthetic t=0 queue. That queue is a
    client-side pacing artifact for closed-loop benchmarking and must not be
    included in TTFT.

    For the first emitted token we therefore stitch the request's service spans:

    For a token produced by an explicit decode forward:

    ``(kv_ready_time - prefill_start_time) + (token_time - decode_start_time)``

    For the first token sampled directly from final-prefill logits,
    ``decode_start_time`` is ``None`` and the result is simply
    ``token_time - prefill_start_time`` (prefill + logits/KV handoff +
    decode-side sampling).

    Both forms exclude the synthetic pre-admission/cap wait. Returns ``None``
    when a timestamp required by the selected form is missing so callers can
    fall back to open-loop accounting.
    """
    if prefill_start_time is None:
        return None
    if decode_start_time is None:
        if kv_ready_time is None:
            return None
        return max(token_time - prefill_start_time, 0.0)
    if kv_ready_time is None:
        return None
    prefill_and_kv = max(kv_ready_time - prefill_start_time, 0.0)
    first_decode_step = max(token_time - decode_start_time, 0.0)
    return prefill_and_kv + first_decode_step


def first_token_latency(state, first_step_lat: float) -> float:
    """TTFT contribution recorded at a PD request's first decode token.

    Equals ``decode_start_time + first_step_lat - arrival_time``. Because
    ``decode_start_time`` already folds in prefill-queue + prefill +
    KV-transfer + decode-queue (it is set to ``sync_decode_start`` at
    admission), the first ``gen_token_latencies`` entry captures the full TTFT,
    and ``E2E = sum(gen_token_latencies)`` telescopes to
    ``decode_end_time - arrival_time``.
    """
    return state.decode_start_time + first_step_lat - state.arrival_time


def decode_step_token_latency(step_lat: float) -> float:
    """ITL/TPOT contribution of a single decode step.

    Currently the identity on the decode step latency (one forward pass == one
    inter-token gap). Kept as a named seam so future per-step refinements
    (e.g. variable step cost) have a single tested entry point.
    """
    return step_lat


def advance_after_decode_step(step_start: float, step_lat: float) -> float:
    """Replica-local decode completion after a step at ``step_start``."""
    return step_start + step_lat


def has_unsupported_chunked(reqs) -> bool:
    """Return whether an extend batch contains a non-final prefill chunk.

    Retained as a compatibility helper for callers that want to report chunked
    traffic. PD now supports chunk affinity, per-chunk prediction and full-
    prompt KV sizing; a true result no longer means the workload is unsupported.
    """
    return any(getattr(req, "is_chunked", 0) != 0 for req in reqs)
