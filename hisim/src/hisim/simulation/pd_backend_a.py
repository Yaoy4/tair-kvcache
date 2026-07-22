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
from typing import AbstractSet, Iterable, List, Optional, Sequence, Tuple

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

    def latest_time(self) -> float:
        return max(self.busy_until)


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
        self._prefill_replica_by_rid: dict[str, int] = {}
        self._prefill_max_running_per_replica: int = (
            getattr(bundle, "prefill_max_running_per_replica", None)
            or (1 << 31) - 1
        )
        self._prefill_running_count: dict[int, int] = defaultdict(int)
        self._prefill_slot_reserved: set[str] = set()
        self._next_prefill_batch_id = 0
        self._decode_replica_by_rid: dict[str, int] = {}
        self._decode_running_count: dict[int, int] = defaultdict(int)
        self._single_decode_running_rids: set[str] = set()
        self._max_running_per_replica: int = (
            getattr(bundle, "decode_max_running_per_replica", None)
            or getattr(bundle, "max_running_per_replica", (1 << 31) - 1)
        )

    # ---- introspection ----
    def controller(self) -> PDController:
        return self._controller

    def prefill_pool_size(self) -> int:
        return len(self._prefill_pool.busy_until)

    def decode_pool_size(self) -> int:
        return len(self._decode_pool.busy_until)

    def decode_queue_mode(self) -> str:
        return self._decode_queue_mode

    def bind_prefill_replicas(
        self, reqs: Sequence[PDRequestState]
    ) -> dict[str, int]:
        """Bind requests to prefill replicas, preserving chunk affinity.

        New requests are spread round-robin over replicas ordered by virtual
        availability.  A chunked request keeps its original replica until its
        final chunk calls :meth:`on_prefill_done`.
        """
        if not reqs:
            return {}
        mapping: dict[str, int] = {}
        for req in reqs:
            self._validate_prefill_phase(req)
            replica_idx = self._reserve_prefill_slot(req)
            if replica_idx is None:
                raise RuntimeError(
                    "prefill capacity exhausted while binding request "
                    f"rid={req.rid!r}"
                )
            mapping[req.rid] = replica_idx
        return mapping

    def prefill_replica_for(self, rid: str) -> Optional[int]:
        return self._prefill_replica_by_rid.get(rid)

    def _validate_prefill_phase(self, req: PDRequestState) -> None:
        if req.phase not in (
            RequestPhase.WAITING_PREFILL,
            RequestPhase.RUNNING_PREFILL,
        ):
            raise ValueError(
                f"cannot schedule prefill for rid={req.rid!r} "
                f"in phase={req.phase.value}"
            )
        if (
            req.phase == RequestPhase.RUNNING_PREFILL
            and req.rid not in self._prefill_slot_reserved
        ):
            raise ValueError(
                f"running prefill request rid={req.rid!r} has no reserved slot"
            )

    def _reserve_prefill_slot(self, req: PDRequestState) -> Optional[int]:
        replica_idx = self._prefill_replica_by_rid.get(req.rid)
        if replica_idx is not None:
            return replica_idx
        candidates = [
            idx
            for idx in range(len(self._prefill_pool.busy_until))
            if self._prefill_running_count[idx]
            < self._prefill_max_running_per_replica
        ]
        if not candidates:
            return None
        replica_idx = min(
            candidates,
            key=lambda idx: (
                self._prefill_running_count[idx],
                self._prefill_pool.busy_until[idx],
                idx,
            ),
        )
        self._prefill_replica_by_rid[req.rid] = replica_idx
        self._prefill_slot_reserved.add(req.rid)
        self._prefill_running_count[replica_idx] += 1
        req.prefill_replica_idx = replica_idx
        return replica_idx

    def _release_prefill_slot(self, req: PDRequestState) -> None:
        if req.rid not in self._prefill_slot_reserved:
            return
        replica_idx = self._prefill_replica_by_rid.get(req.rid)
        self._prefill_slot_reserved.remove(req.rid)
        if replica_idx is not None:
            self._prefill_running_count[replica_idx] = max(
                0, self._prefill_running_count[replica_idx] - 1
            )

    def _predict_prefill_batch(self, wave: Sequence[PDRequestState]) -> float:
        input_lengths = [int(req.input_length) for req in wave]
        batch_predict = getattr(
            self._bundle.prefill, "predict_prefill_batch_seconds", None
        )
        if batch_predict is not None:
            return float(batch_predict(input_lengths))
        return float(
            self._bundle.prefill.predict_prefill_seconds(sum(input_lengths))
        )

    def _schedule_prefill_wave(
        self, replica_idx: int, wave: Sequence[PDRequestState], now: float
    ) -> float:
        start = max(now, self._prefill_pool.busy_until[replica_idx])
        end = start + self._predict_prefill_batch(wave)
        self._prefill_pool.busy_until[replica_idx] = end
        self._next_prefill_batch_id += 1
        batch_id = self._next_prefill_batch_id

        fresh = [req for req in wave if req.phase == RequestPhase.WAITING_PREFILL]
        for fresh_req in fresh:
            self._controller.on_request_arrival(fresh_req, now)
        if fresh:
            admitted = self._controller.admit_prefill(
                capacity=len(fresh), now=start
            )
            if len(admitted) != len(fresh) or any(
                actual is not expected
                for actual, expected in zip(admitted, fresh)
            ):
                raise AssertionError(
                    "controller did not preserve replica-local prefill batch"
                )
        for req in wave:
            req.prefill_replica_idx = replica_idx
            req.prefill_batch_id = batch_id
            req.prefill_end_time = end
            if req.prefill_is_final_chunk:
                self._release_prefill_slot(req)
        return end

    def earliest_pool_time(self, pool: str) -> float:
        """The correct floor for this pool's next admission call.

        For "prefill" (and "decode" under `per_replica_queue`), each request
        or bucket is genuinely independent, so the floor is the earliest ANY
        replica is free (min busy_until) -- a slower replica must never hold
        back a faster, unrelated one; the per-replica `max(now, busy_until)`
        inside try_admit_* self-corrects for whichever replica actually gets
        picked.

        For "decode" under the default `single_replica` mode, every call
        bundles the WHOLE current cohort onto a single, freshly-chosen
        replica -- there is no sticky per-request replica affinity, so the
        only thing that correctly orders consecutive calls is the LAST call's
        actual completion time regardless of which replica it landed on, i.e.
        the latest (max) busy_until across the pool. Using min here would let
        a request's own next step start before its own previous step ended.
        """
        if pool == "prefill":
            return self._prefill_pool.earliest_time()
        if pool == "decode":
            if self._decode_queue_mode == "per_replica_queue":
                return self._decode_pool.earliest_time()
            return self._decode_pool.latest_time()
        raise ValueError(f"unknown pool {pool!r}; expected 'prefill' or 'decode'")

    def bind_decode_replicas(
        self, reqs: Sequence[PDRequestState]
    ) -> dict[str, int]:
        if not reqs:
            return {}
        allowed = {
            RequestPhase.KV_TRANSIT,
            RequestPhase.WAITING_DECODE,
            RequestPhase.RUNNING_DECODE,
        }
        for req in reqs:
            if req.phase not in allowed:
                raise ValueError(
                    f"cannot bind decode replica for rid={req.rid!r} "
                    f"in phase={req.phase.value}"
                )
        if self._decode_queue_mode != "per_replica_queue":
            idx, _ = self._decode_pool.earliest_replica()
            return {req.rid: idx for req in reqs}

        replica_indices = list(range(len(self._decode_pool.busy_until)))
        if not replica_indices:
            raise ValueError("decode pool requires at least one replica")

        planned_count = dict(self._decode_running_count)
        mapping: dict[str, int] = {}
        for req in reqs:
            replica_idx = self._decode_replica_by_rid.get(req.rid)
            if replica_idx is None:
                candidates = [
                    idx
                    for idx in replica_indices
                    if planned_count.get(idx, 0)
                    < self._max_running_per_replica
                ]
                if not candidates:
                    raise RuntimeError("decode replica capacity exhausted")
                replica_idx = min(
                    candidates,
                    key=lambda idx: (
                        planned_count.get(idx, 0),
                        self._decode_pool.busy_until[idx],
                        idx,
                    ),
                )
                self._decode_replica_by_rid[req.rid] = replica_idx
                planned_count[replica_idx] = planned_count.get(replica_idx, 0) + 1
            mapping[req.rid] = replica_idx
        return mapping

    def decode_replica_time(self, replica_idx: int) -> float:
        return self._decode_pool.busy_until[replica_idx]

    def decode_batch_capacity(self) -> int:
        if self._decode_queue_mode == "per_replica_queue":
            return self._max_running_per_replica * self.decode_pool_size()
        return self._max_running_per_replica

    def admit_decode_single_replica(
        self, rids: AbstractSet[str], now: float
    ) -> List[PDRequestState]:
        available = max(
            0,
            self._max_running_per_replica
            - len(self._single_decode_running_rids),
        )
        admitted = self._controller.admit_decode_targeted(
            rids, now, max_count=available
        )
        self._single_decode_running_rids.update(req.rid for req in admitted)
        return admitted

    # ---- scheduling primitives ----
    def try_admit_prefill(
        self, req: PDRequestState, now: float
    ) -> Tuple[int, float]:
        """Schedule one request onto the earliest-free prefill replica.

        Advances that replica's busy_until clock. Updates req state to
        RUNNING_PREFILL via the controller. Returns (replica_idx, end_time).
        """
        return self.try_admit_prefill_batch([req], now)

    def try_admit_prefill_batch(
        self, reqs: Sequence[PDRequestState], now: float
    ) -> Tuple[int, float]:
        """Partition a scheduler batch into replica-local prefill batches.

        Each replica-local wave uses one predictor call with the sum of that
        wave's input lengths. Waves are capacity-bounded and requests retain
        replica affinity across chunked-prefill iterations.

        Phase guard: only WAITING_PREFILL requests go through the controller
        lifecycle (on_request_arrival + admit_prefill). Mid-chunk requests
        (already RUNNING_PREFILL) contribute to latency estimation but bypass
        state transitions so their phase is not reset.
        """
        if not reqs:
            raise ValueError("try_admit_prefill_batch requires at least one request")
        if len({req.rid for req in reqs}) != len(reqs):
            raise ValueError("prefill batch contains duplicate request ids")
        for req in reqs:
            self._validate_prefill_phase(req)
        pending = list(reqs)
        first_replica_idx: Optional[int] = None
        max_end = now
        while pending:
            buckets: dict[int, list[PDRequestState]] = defaultdict(list)
            remaining: list[PDRequestState] = []
            for req in pending:
                replica_idx = self._prefill_replica_by_rid.get(req.rid)
                if replica_idx is None:
                    replica_idx = self._reserve_prefill_slot(req)
                if replica_idx is None:
                    remaining.append(req)
                    continue
                buckets[replica_idx].append(req)

            if not buckets:
                blocked = ", ".join(req.rid for req in remaining)
                raise RuntimeError(
                    "prefill capacity exhausted by unfinished chunked requests; "
                    f"cannot schedule: {blocked}"
                )

            for replica_idx in sorted(
                buckets, key=lambda idx: self._prefill_pool.busy_until[idx]
            ):
                if first_replica_idx is None:
                    first_replica_idx = replica_idx
                end = self._schedule_prefill_wave(
                    replica_idx, buckets[replica_idx], now
                )
                max_end = max(max_end, end)
            pending = remaining

        if first_replica_idx is None:
            raise AssertionError("non-empty prefill batch produced no replica bucket")
        return first_replica_idx, max_end

    def compute_kv_ready_time(self, req: PDRequestState, now: float) -> float:
        return self._controller.compute_kv_ready_time(req, now)

    def compute_batch_kv_ready_time(self, total_tokens: int, now: float) -> float:
        return self._controller.compute_batch_kv_ready_time(total_tokens, now)

    def on_prefill_done(
        self, req: PDRequestState, now: float, kv_ready_time: float
    ) -> None:
        self._controller.on_prefill_done(req, now, kv_ready_time)
        self._release_prefill_slot(req)
        self._prefill_replica_by_rid.pop(req.rid, None)

    def on_prefill_token_sampled(
        self, req: PDRequestState, now: float
    ) -> None:
        self._controller.on_prefill_token_sampled(req, now)
        if req.phase == RequestPhase.FINISHED:
            self._release_prefill_slot(req)
            self._prefill_replica_by_rid.pop(req.rid, None)
            self._decode_replica_by_rid.pop(req.rid, None)

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
        if req.phase == RequestPhase.WAITING_DECODE:
            if self._decode_queue_mode == "per_replica_queue":
                idx = self.bind_decode_replicas([req])[req.rid]
                admitted = self.admit_decode_for_replica(
                    idx, {req.rid}, now
                )
            else:
                admitted = self.admit_decode_single_replica({req.rid}, now)
            if admitted != [req]:
                raise AssertionError("decode step could not admit waiting request")
        if req.phase != RequestPhase.RUNNING_DECODE:
            raise ValueError(
                f"cannot schedule decode step for rid={req.rid!r} "
                f"in phase={req.phase.value}"
            )
        if self._decode_queue_mode == "per_replica_queue":
            idx = self.bind_decode_replicas([req])[req.rid]
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
        for req in reqs:
            if req.phase != RequestPhase.RUNNING_DECODE:
                raise ValueError(
                    f"cannot schedule decode batch for rid={req.rid!r} "
                    f"in phase={req.phase.value}"
                )
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
        mapping = self.bind_decode_replicas(reqs)
        replica_order = sorted(
            range(len(self._decode_pool.busy_until)),
            key=lambda i: self._decode_pool.busy_until[i],
        )
        if not replica_order:
            raise ValueError("decode pool requires at least one replica")

        buckets: dict[int, list[PDRequestState]] = defaultdict(list)
        for req in reqs:
            replica_idx = mapping[req.rid]
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
            self._single_decode_running_rids.discard(req.rid)
            replica_idx = self._decode_replica_by_rid.pop(req.rid, None)
            if replica_idx is not None:
                self._decode_running_count[replica_idx] = max(
                    0, self._decode_running_count[replica_idx] - 1
                )

    def on_decode_step_done_batch(
        self, reqs: Iterable[PDRequestState], now: float
    ) -> None:
        self._controller.on_decode_step_done(reqs, now)
        for req in reqs:
            if req.phase == RequestPhase.FINISHED:
                self._single_decode_running_rids.discard(req.rid)
                replica_idx = self._decode_replica_by_rid.pop(req.rid, None)
                if replica_idx is not None:
                    self._decode_running_count[replica_idx] = max(
                        0, self._decode_running_count[replica_idx] - 1
                    )

    def terminate_request(self, req: PDRequestState, now: float) -> None:
        """Release every reservation for an external abort/cancellation."""
        self._controller.terminate_request(req, now)
        self._release_prefill_slot(req)
        self._prefill_replica_by_rid.pop(req.rid, None)
        self._single_decode_running_rids.discard(req.rid)
        replica_idx = self._decode_replica_by_rid.pop(req.rid, None)
        if replica_idx is not None:
            self._decode_running_count[replica_idx] = max(
                0, self._decode_running_count[replica_idx] - 1
            )

    def reset_for_retract(self, req: PDRequestState, now: float) -> None:
        """Re-sync PD state when SGLang retracts an in-flight decode request.

        Releases every prefill- and decode-side reservation so the request's
        upcoming re-prefill goes through normal capacity admission instead of
        (a) illegally reusing its stale sticky prefill-replica binding for
        free (bypassing ``max_running_per_replica`` -- sticky lookups skip
        the capacity check by design, which is correct for chunk
        continuations but wrong for a request starting over), or (b)
        permanently leaking a decode-replica capacity slot that nothing else
        will ever release.

        Mirrors :meth:`terminate_request`'s reservation cleanup but delegates
        the phase transition to :meth:`PDController.reset_for_retract`
        (WAITING_PREFILL, not FINISHED) and leaves decode-progress fields
        (``decode_step_count`` / ``current_past_kv_length`` /
        ``output_length``) untouched -- see that method's docstring for why.
        """
        self._controller.reset_for_retract(req, now)
        self._release_prefill_slot(req)
        self._prefill_replica_by_rid.pop(req.rid, None)
        self._single_decode_running_rids.discard(req.rid)
        replica_idx = self._decode_replica_by_rid.pop(req.rid, None)
        if replica_idx is not None:
            self._decode_running_count[replica_idx] = max(
                0, self._decode_running_count[replica_idx] - 1
            )
        req.prefill_replica_idx = None
        req.prefill_batch_id = None
        req.prefill_start_time = None
        req.prefill_end_time = None
        req.kv_ready_time = None
        req.decode_start_time = None
        req.decode_end_time = None
        req.prefill_is_final_chunk = True

    def admit_decode_for_replica(
        self, replica_idx: int, rids: "AbstractSet[str]", now: float
    ) -> List[PDRequestState]:
        """Admit requests for a specific decode replica, respecting max_running_per_replica."""
        capacity = max(
            0, self._max_running_per_replica - self._decode_running_count[replica_idx]
        )
        admitted = self._controller.admit_decode_targeted(rids, now, max_count=capacity)
        self._decode_running_count[replica_idx] += len(admitted)
        admitted_rids = {req.rid for req in admitted}
        for rid in set(rids) - admitted_rids:
            self._decode_replica_by_rid.pop(rid, None)
        return admitted
