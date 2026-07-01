"""Pure, SGLang-free timeline helpers for PD dual-clock decoupling (P1).

Pure semantic layer — must NOT import SGLang or backend-specific code
(mirrors ``pd_metrics``). These functions encode the *role-clock* arithmetic
so the (hard-to-unit-test) SGLang hook closure stays a thin orchestration
layer over logic that is fully covered by fast, deterministic unit tests.

Background
----------
The single-process SGLang hook drives **two** logical engines (prefill and
decode) from **one** scheduler loop. Historically it advanced a single global
clock by every batch's latency, so a prefill batch (and its KV transfer) froze
all in-flight decode requests in virtual time. P1 decouples the two roles into
two independent "role clocks":

- ``PD_PREFILL_CLOCK``: advanced only by prefill (extend) batches.
- ``PD_DECODE_CLOCK``:  advanced only by decode batches; jumps forward to a
  request's ``kv_ready_time`` at the prefill->decode handoff.

Both clocks are *self-anchoring*: they start at 0 and are reconciled against
already-anchored quantities (each request's real ``arrival_time`` for prefill,
and ``kv_ready_time`` for decode) via ``max(...)``. This keeps the absolute
time base aligned with ``RequestStats.created_time`` in both OFFLINE and
BLOCKING simulation modes without any explicit epoch bookkeeping.
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

    ``(kv_ready_time - prefill_start_time) + (token_time - decode_start_time)``

    which equals ``prefill + KV transfer + first decode step`` and excludes the
    synthetic pre-admission/cap wait. Returns ``None`` when any required
    timestamp is missing so callers can fall back to open-loop accounting.
    """
    if (
        prefill_start_time is None
        or kv_ready_time is None
        or decode_start_time is None
    ):
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
    """New ``PD_DECODE_CLOCK`` value after a decode step that began at
    ``step_start`` and took ``step_lat`` seconds."""
    return step_start + step_lat


def has_unsupported_chunked(reqs) -> bool:
    """True if any request in an extend batch is mid chunked-prefill.

    SGLang flags non-final chunks with ``is_chunked != 0``. The PD glue assumes
    one extend batch per request (input_length / kv_ready / decode-base captured
    once), so a chunked request would corrupt those quantities. The hook uses
    this to emit a single clear warning instead of silently producing bad PD
    numbers. Requests without an ``is_chunked`` attribute are treated as
    unchunked (0).
    """
    return any(getattr(req, "is_chunked", 0) != 0 for req in reqs)
