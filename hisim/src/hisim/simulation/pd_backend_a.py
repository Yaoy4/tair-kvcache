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

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from hisim.simulation.pd_controller import PDController
from hisim.simulation.pd_factory import DisaggPredictors
from hisim.simulation.pd_types import PDRequestState, RequestPhase


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
        self._decode_queue_mode = getattr(bundle, "decode_queue_mode", "single_replica")
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
        self._decode_replica_by_rid: dict[str, int] = {}

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
        if self._decode_queue_mode == "per_replica_queue":
            idx = self._decode_replica_by_rid.get(req.rid)
            if idx is None:
                idx, _ = self._decode_pool.earliest_replica()
                self._decode_replica_by_rid[req.rid] = idx
            free_at = self._decode_pool.busy_until[idx]
        else:
            idx, free_at = self._decode_pool.earliest_replica()
        start = max(now, free_at)
        dur = self._bundle.decode.predict_decode_seconds(batch_size=1)
        end = start + dur
        self._decode_pool.busy_until[idx] = end
        return idx, end

    def try_admit_decode_batch(
        self, reqs: Sequence[PDRequestState], now: float
    ) -> Tuple[int, float]:
        """Schedule one decode step for an entire batch on the earliest-free
        decode replica.

        Passes each request's ``current_past_kv_length`` to the predictor so
        latency reflects real batch composition. Returns (replica_idx,
        end_time). Caller is responsible for calling
        :meth:`on_decode_step_done_batch` once the step completes.
        """
        if not reqs:
            raise ValueError("try_admit_decode_batch requires at least one request")
        if self._decode_queue_mode == "per_replica_queue":
            return self._try_admit_decode_batch_per_replica(reqs, now)
        idx, free_at = self._decode_pool.earliest_replica()
        start = max(now, free_at)
        past_kv = [int(r.current_past_kv_length) for r in reqs]
        dur = self._bundle.decode.predict_decode_seconds(
            batch_size=len(reqs), past_kv_length=past_kv
        )
        end = start + dur
        self._decode_pool.busy_until[idx] = end
        return idx, end

    def _try_admit_decode_batch_per_replica(
        self, reqs: Sequence[PDRequestState], now: float
    ) -> Tuple[int, float]:
        replica_order = sorted(
            range(len(self._decode_pool.busy_until)),
            key=lambda i: self._decode_pool.busy_until[i],
        )
        if not replica_order:
            raise ValueError("decode pool requires at least one replica")

        next_replica_slot = 0
        buckets: dict[int, list[PDRequestState]] = defaultdict(list)
        for req in reqs:
            replica_idx = self._decode_replica_by_rid.get(req.rid)
            if replica_idx is None:
                replica_idx = replica_order[next_replica_slot % len(replica_order)]
                self._decode_replica_by_rid[req.rid] = replica_idx
                next_replica_slot += 1
            buckets[replica_idx].append(req)

        first_replica_idx: Optional[int] = None
        max_end = now
        for replica_idx in replica_order:
            bucket = buckets.get(replica_idx)
            if not bucket:
                continue
            start = max(now, self._decode_pool.busy_until[replica_idx])
            past_kv = [int(r.current_past_kv_length) for r in bucket]
            dur = self._bundle.decode.predict_decode_seconds(
                batch_size=len(bucket), past_kv_length=past_kv
            )
            end = start + dur
            self._decode_pool.busy_until[replica_idx] = end
            if end > max_end:
                max_end = end
            if first_replica_idx is None:
                first_replica_idx = replica_idx
        if first_replica_idx is None:
            first_replica_idx = replica_order[0]
        return first_replica_idx, max_end

    def on_decode_step_done(self, req: PDRequestState, now: float) -> None:
        self._controller.on_decode_step_done([req], now)
        if req.phase == RequestPhase.FINISHED:
            self._decode_replica_by_rid.pop(req.rid, None)

    def on_decode_step_done_batch(
        self, reqs: Iterable[PDRequestState], now: float
    ) -> None:
        self._controller.on_decode_step_done(reqs, now)
        for req in reqs:
            if req.phase == RequestPhase.FINISHED:
                self._decode_replica_by_rid.pop(req.rid, None)
