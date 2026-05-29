"""Phase 2a — Backend A: single-process virtual-time PD adapter.

Encapsulates the dual-pool, dual-clock scheduling that is needed to simulate
PD disaggregation in one process. Stays free of sglang imports so it is unit-
testable with a stub predictor and reusable by both the demo and the Phase 2b
SGLang hook.

Design:
- Holds a DisaggPredictors bundle (per-role predictor instances + KV transfer
  model + kv_bytes_per_token derived via utils).
- Owns two replica pools, each tracked as a busy_until vector of size
  `RolePredictorConfig.replicas`.
- Delegates request lifecycle bookkeeping to a PDController so that Backend B
  can reuse the same controller transitions later.

The per-role predictor is expected to expose two methods:
    predict_prefill_seconds(batch_tokens: int) -> float
    predict_decode_seconds(batch_size: int) -> float

For tests we inject a stub that satisfies that protocol; in production the
real AIConfiguratorTimePredictor will need a thin adapter (added in 2b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from hisim.simulation.pd_controller import PDController
from hisim.simulation.pd_factory import DisaggPredictors
from hisim.simulation.pd_types import PDRequestState


@dataclass
class _ReplicaPool:
    name: str
    busy_until: List[float]

    @classmethod
    def make(cls, name: str, replicas: int) -> "_ReplicaPool":
        if replicas < 1:
            raise ValueError(f"{name} pool requires replicas >= 1, got {replicas}")
        return cls(name=name, busy_until=[0.0] * replicas)

    def earliest_replica(self) -> Tuple[int, float]:
        idx = min(range(len(self.busy_until)), key=lambda i: self.busy_until[i])
        return idx, self.busy_until[idx]

    def earliest_time(self) -> float:
        return min(self.busy_until)


class BackendA:
    """Single-process virtual-time backend for PD disaggregation."""

    def __init__(self, bundle: DisaggPredictors,
                 controller: Optional[PDController] = None):
        self._bundle = bundle
        self._controller = controller or PDController(
            transfer_model=bundle.transfer_model,
            kv_model_cfg=bundle.kv_model_config,
        )
        self._prefill_pool = _ReplicaPool.make(
            "prefill", bundle.prefill_replicas
        )
        self._decode_pool = _ReplicaPool.make(
            "decode", bundle.decode_replicas
        )

    # ---- introspection ----
    def controller(self) -> PDController:
        return self._controller

    def prefill_pool_size(self) -> int:
        return len(self._prefill_pool.busy_until)

    def decode_pool_size(self) -> int:
        return len(self._decode_pool.busy_until)

    def earliest_pool_time(self, pool: str) -> float:
        if pool == "prefill":
            return self._prefill_pool.earliest_time()
        if pool == "decode":
            return self._decode_pool.earliest_time()
        raise ValueError(f"unknown pool {pool!r}; expected 'prefill' or 'decode'")

    # ---- scheduling primitives ----
    def try_admit_prefill(
        self, req: PDRequestState, now: float
    ) -> Tuple[int, float]:
        """Schedule one request onto the earliest-free prefill replica.

        Advances that replica's busy_until clock. Updates req state to
        RUNNING_PREFILL via the controller. Returns (replica_idx, end_time).
        """
        idx, free_at = self._prefill_pool.earliest_replica()
        start = max(now, free_at)
        dur = self._bundle.prefill.predict_prefill_seconds(req.input_length)
        end = start + dur
        self._prefill_pool.busy_until[idx] = end
        # Drive the controller's state machine.
        self._controller.on_request_arrival(req, now)
        admitted = self._controller.admit_prefill(capacity=1, now=start)
        # admit_prefill takes from its own queue; we just appended `req` so it
        # is guaranteed to be the one admitted.
        if not admitted or admitted[0] is not req:
            raise AssertionError("controller did not admit just-enqueued request")
        return idx, end

    def compute_kv_ready_time(self, req: PDRequestState, now: float) -> float:
        return self._controller.compute_kv_ready_time(req, now)

    def on_prefill_done(
        self, req: PDRequestState, now: float, kv_ready_time: float
    ) -> None:
        self._controller.on_prefill_done(req, now, kv_ready_time)

    def advance_to_kv_ready(self, req: PDRequestState, now: float) -> None:
        """Convenience: move req from KV_TRANSIT → WAITING_DECODE at `now`."""
        # poll_kv_ready scans all in-flight transfers; for one request that is
        # already known-ready it is the right primitive.
        self._controller.poll_kv_ready(now)

    def try_admit_decode_step(
        self, req: PDRequestState, now: float
    ) -> Tuple[int, float]:
        """Schedule one decode step for `req` on the earliest-free decode
        replica. Returns (replica_idx, end_time).
        """
        idx, free_at = self._decode_pool.earliest_replica()
        start = max(now, free_at)
        dur = self._bundle.decode.predict_decode_seconds(batch_size=1)
        end = start + dur
        self._decode_pool.busy_until[idx] = end
        return idx, end

    def on_decode_step_done(self, req: PDRequestState, now: float) -> None:
        self._controller.on_decode_step_done([req], now)
