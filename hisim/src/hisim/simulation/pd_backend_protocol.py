"""Phase 5c.0 — Structural type for PD execution backends.

Both ``BackendA`` (single-process virtual-time) and ``BackendB`` (two-process
spawn) implement the same public surface. This module pins that contract as
a :class:`typing.Protocol` so:

* ``pd_runtime`` helpers can be typed against either backend without an
  import cycle (the protocol lives in a leaf module),
* the SGLang hook in 5c.1 can have a single ``PDBackendProtocol`` slot it
  fills with either backend based on ``disagg.backend``,
* future backends only need to match the protocol — no inheritance required.

This file is intentionally behavior-free: it adds types only.
"""
from __future__ import annotations

from typing import Iterable, Protocol, Sequence, Tuple, runtime_checkable

from hisim.simulation.pd_controller import PDController
from hisim.simulation.pd_types import PDRequestState


@runtime_checkable
class PDBackendProtocol(Protocol):
    """Structural contract shared by BackendA and BackendB.

    Methods are grouped by call site:

    Introspection (used by sglang_hook + tests):
        controller, prefill_pool_size, decode_pool_size, earliest_pool_time
    Prefill scheduling:
        try_admit_prefill, compute_kv_ready_time, on_prefill_done,
        advance_to_kv_ready
    Decode scheduling:
        try_admit_decode_step, try_admit_decode_batch,
        on_decode_step_done, on_decode_step_done_batch
    """

    # ---- introspection ----
    def controller(self) -> PDController: ...
    def prefill_pool_size(self) -> int: ...
    def decode_pool_size(self) -> int: ...
    def earliest_pool_time(self, pool: str) -> float: ...

    # ---- prefill ----
    def try_admit_prefill(
        self, req: PDRequestState, now: float
    ) -> Tuple[int, float]: ...

    def compute_kv_ready_time(
        self, req: PDRequestState, now: float
    ) -> float: ...

    def on_prefill_done(
        self, req: PDRequestState, now: float, kv_ready_time: float
    ) -> None: ...

    def advance_to_kv_ready(
        self, req: PDRequestState, now: float
    ) -> None: ...

    # ---- decode ----
    def try_admit_decode_step(
        self, req: PDRequestState, now: float
    ) -> Tuple[int, float]: ...

    def try_admit_decode_batch(
        self, reqs: Sequence[PDRequestState], now: float
    ) -> Tuple[int, float]: ...

    def on_decode_step_done(
        self, req: PDRequestState, now: float
    ) -> None: ...

    def on_decode_step_done_batch(
        self, reqs: Iterable[PDRequestState], now: float
    ) -> None: ...
