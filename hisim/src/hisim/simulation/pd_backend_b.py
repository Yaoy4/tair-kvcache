"""Phase 5a — Backend B skeleton: two-process PD adapter.

Mirrors :class:`hisim.simulation.pd_backend_a.BackendA`'s public surface but
offloads each predictor call to a dedicated worker process (one per replica,
per role) over ``multiprocessing.Queue`` IPC.

Phase 5a scope (skeleton):
    * process spawn + clean shutdown lifecycle,
    * synchronous request/reply per call (keeps API drop-in with Backend A),
    * same :class:`PDController` integration; PD semantic layer unchanged.

Out of scope for 5a (left for 5b/5c):
    * true async fire-and-forget admission,
    * hosting real SGLang schedulers inside the workers,
    * stats merge across workers (predictor calls return scalar durations only).

Workers receive a *predictor factory* (a top-level picklable callable) and
construct the predictor lazily inside the child process. This avoids forcing
the predictor object itself to be picklable — important because the production
AIConfigurator predictor holds DB handles that do not survive ``spawn``.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from typing import AbstractSet, Callable, Iterable, List, Optional, Sequence, Tuple

from hisim.simulation.pd_controller import PDController
from hisim.simulation.pd_factory import DisaggPredictors
from hisim.simulation.pd_types import PDRequestState, RequestPhase


# ---------------------------------------------------------------------------
# IPC messages — all picklable, all defined at module top level.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PrefillJob:
    job_id: int
    input_lengths: Tuple[int, ...]


@dataclass(frozen=True)
class _DecodeJob:
    job_id: int
    batch_size: int
    past_kv: Tuple[int, ...]


@dataclass(frozen=True)
class _Result:
    job_id: int
    duration: float = 0.0
    error: Optional[str] = None  # repr of exception type + str + traceback


@dataclass(frozen=True)
class _Shutdown:
    pass


_SHUTDOWN = _Shutdown()


class BackendBWorkerError(RuntimeError):
    """Raised in the parent when a BackendB worker reports an exception."""


# ---------------------------------------------------------------------------
# Worker entry points — must be top-level so multiprocessing 'spawn' can
# import them in the child interpreter.
# ---------------------------------------------------------------------------


def _prefill_worker_main(predictor_factory, in_q, out_q):
    import traceback
    try:
        predictor = predictor_factory()
    except BaseException as e:
        # Boot failed. Forward error on the NEXT job request so the parent
        # surfaces it instead of blocking forever.
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        while True:
            msg = in_q.get()
            if isinstance(msg, _Shutdown):
                return
            out_q.put(_Result(job_id=msg.job_id, error=err))
    while True:
        msg = in_q.get()
        if isinstance(msg, _Shutdown):
            return
        try:
            batch_predict = getattr(
                predictor, "predict_prefill_batch_seconds", None
            )
            if batch_predict is not None:
                dur = batch_predict(list(msg.input_lengths))
            else:
                dur = predictor.predict_prefill_seconds(sum(msg.input_lengths))
            out_q.put(_Result(job_id=msg.job_id, duration=float(dur)))
        except BaseException as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            out_q.put(_Result(job_id=msg.job_id, error=err))


def _decode_worker_main(predictor_factory, in_q, out_q):
    import traceback
    try:
        predictor = predictor_factory()
    except BaseException as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        while True:
            msg = in_q.get()
            if isinstance(msg, _Shutdown):
                return
            out_q.put(_Result(job_id=msg.job_id, error=err))
    while True:
        msg = in_q.get()
        if isinstance(msg, _Shutdown):
            return
        try:
            dur = predictor.predict_decode_seconds(
                batch_size=msg.batch_size,
                past_kv_length=list(msg.past_kv),
            )
            out_q.put(_Result(job_id=msg.job_id, duration=float(dur)))
        except BaseException as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            out_q.put(_Result(job_id=msg.job_id, error=err))


# ---------------------------------------------------------------------------
# Worker handle (one per replica).
# ---------------------------------------------------------------------------


@dataclass
class _WorkerHandle:
    process: "mp.Process"
    in_q: "mp.Queue"
    out_q: "mp.Queue"
    busy_until: float = 0.0


# ---------------------------------------------------------------------------
# Backend B
# ---------------------------------------------------------------------------


class BackendB:
    """Two-process PD backend skeleton.

    Public API mirrors :class:`BackendA` so callers (``pd_runtime``,
    ``sglang_hook``, ``pd_demo``) can swap backends without code changes.
    Phase 5a keeps each predictor call synchronous; richer asynchrony lands
    in 5b.
    """

    def __init__(
        self,
        bundle: DisaggPredictors,
        prefill_predictor_factory: Callable[[], object],
        decode_predictor_factory: Callable[[], object],
        controller: Optional[PDController] = None,
        mp_context: Optional["mp.context.BaseContext"] = None,
    ):
        if bundle.prefill_replicas < 1:
            raise ValueError("prefill_replicas must be >= 1")
        if bundle.decode_replicas < 1:
            raise ValueError("decode_replicas must be >= 1")
        self._bundle = bundle
        self._controller = controller or PDController(
            transfer_model=bundle.transfer_model,
            kv_model_cfg=bundle.kv_model_config,
        )
        self._ctx = mp_context or mp.get_context("spawn")
        self._prefill_factory = prefill_predictor_factory
        self._decode_factory = decode_predictor_factory
        self._decode_queue_mode = getattr(bundle, "decode_queue_mode", "single_replica")
        self._prefill_workers: List[_WorkerHandle] = []
        self._decode_workers: List[_WorkerHandle] = []
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
        self._next_job_id = 0
        self._started = False

    # ---- lifecycle ----
    def start(self) -> None:
        if self._started:
            return
        for _ in range(self._bundle.prefill_replicas):
            self._prefill_workers.append(
                self._spawn(_prefill_worker_main, self._prefill_factory)
            )
        for _ in range(self._bundle.decode_replicas):
            self._decode_workers.append(
                self._spawn(_decode_worker_main, self._decode_factory)
            )
        self._started = True

    def shutdown(self, timeout: float = 5.0) -> None:
        for w in self._prefill_workers + self._decode_workers:
            try:
                w.in_q.put(_SHUTDOWN)
            except Exception:
                pass
        for w in self._prefill_workers + self._decode_workers:
            w.process.join(timeout=timeout)
            if w.process.is_alive():
                w.process.terminate()
                w.process.join(timeout=1.0)
        self._prefill_workers.clear()
        self._decode_workers.clear()
        self._prefill_replica_by_rid.clear()
        self._prefill_running_count.clear()
        self._prefill_slot_reserved.clear()
        self._decode_replica_by_rid.clear()
        self._decode_running_count.clear()
        self._single_decode_running_rids.clear()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def _spawn(self, target, factory) -> _WorkerHandle:
        in_q = self._ctx.Queue()
        out_q = self._ctx.Queue()
        proc = self._ctx.Process(
            target=target, args=(factory, in_q, out_q), daemon=True
        )
        proc.start()
        return _WorkerHandle(process=proc, in_q=in_q, out_q=out_q)

    # ---- introspection (mirror BackendA) ----
    def controller(self) -> PDController:
        return self._controller

    def prefill_pool_size(self) -> int:
        return self._bundle.prefill_replicas

    def decode_pool_size(self) -> int:
        return self._bundle.decode_replicas

    def decode_queue_mode(self) -> str:
        return self._decode_queue_mode

    def bind_prefill_replicas(
        self, reqs: Sequence[PDRequestState]
    ) -> dict[str, int]:
        if not reqs:
            return {}
        self._require_started()
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
            for idx in range(len(self._prefill_workers))
            if self._prefill_running_count[idx]
            < self._prefill_max_running_per_replica
        ]
        if not candidates:
            return None
        replica_idx = min(
            candidates,
            key=lambda idx: (
                self._prefill_running_count[idx],
                self._prefill_workers[idx].busy_until,
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

    def _schedule_prefill_wave(
        self, replica_idx: int, wave: Sequence[PDRequestState], now: float
    ) -> float:
        worker = self._prefill_workers[replica_idx]
        start = max(now, worker.busy_until)
        job_id = self._next_id()
        worker.in_q.put(
            _PrefillJob(
                job_id=job_id,
                input_lengths=tuple(int(req.input_length) for req in wave),
            )
        )
        res = self._await_result(
            worker, job_id, role="prefill", idx=replica_idx
        )
        end = start + res.duration
        worker.busy_until = end
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
        """See BackendA.earliest_pool_time for the full rationale.

        BackendB mirrors BackendA's decode queue modes:

        * ``single_replica``: one decode call bundles the whole cohort onto
          a single worker, so the next call is floored by the LAST decode
          completion (max busy_until).
        * ``per_replica_queue``: requests stick to individual decode workers,
          so independent replicas advance on their own timelines and the next
          admission floor is the earliest free worker (min busy_until).
        """
        if pool == "prefill":
            return min(w.busy_until for w in self._prefill_workers)
        if pool == "decode":
            if self._decode_queue_mode == "per_replica_queue":
                return min(w.busy_until for w in self._decode_workers)
            return max(w.busy_until for w in self._decode_workers)
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
            idx, _ = self._earliest_worker(self._decode_workers)
            return {req.rid: idx for req in reqs}

        replica_indices = list(range(len(self._decode_workers)))
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
                        self._decode_workers[idx].busy_until,
                        idx,
                    ),
                )
                self._decode_replica_by_rid[req.rid] = replica_idx
                planned_count[replica_idx] = planned_count.get(replica_idx, 0) + 1
            mapping[req.rid] = replica_idx
        return mapping

    def decode_replica_time(self, replica_idx: int) -> float:
        return self._decode_workers[replica_idx].busy_until

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

    # ---- internals ----
    def _earliest_worker(
        self, workers: List[_WorkerHandle]
    ) -> Tuple[int, _WorkerHandle]:
        idx = min(range(len(workers)), key=lambda i: workers[i].busy_until)
        return idx, workers[idx]

    def _next_id(self) -> int:
        self._next_job_id += 1
        return self._next_job_id

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("BackendB.start() must be called before use")

    def _await_result(
        self, worker: _WorkerHandle, expected_job_id: int, *, role: str, idx: int
    ) -> _Result:
        """Block until the worker returns; re-raise worker exceptions."""
        res = worker.out_q.get()
        if res.job_id != expected_job_id:
            raise RuntimeError(
                f"{role} worker {idx} returned job {res.job_id}, "
                f"expected {expected_job_id}"
            )
        if res.error is not None:
            raise BackendBWorkerError(
                f"{role} worker {idx} raised:\n{res.error}"
            )
        return res

    # ---- scheduling primitives ----
    def try_admit_prefill(
        self, req: PDRequestState, now: float
    ) -> Tuple[int, float]:
        self._require_started()
        return self.try_admit_prefill_batch([req], now)

    def try_admit_prefill_batch(
        self, reqs: Sequence[PDRequestState], now: float
    ) -> Tuple[int, float]:
        """Partition a scheduler batch into capacity-bounded worker batches.

        Phase guard: only WAITING_PREFILL requests go through the controller
        lifecycle; mid-chunk requests (already RUNNING_PREFILL) bypass state
        transitions but still contribute to the latency estimate.
        """
        if not reqs:
            raise ValueError("try_admit_prefill_batch requires at least one request")
        if len({req.rid for req in reqs}) != len(reqs):
            raise ValueError("prefill batch contains duplicate request ids")
        self._require_started()
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
                buckets, key=lambda idx: self._prefill_workers[idx].busy_until
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

    def advance_to_kv_ready(self, req: PDRequestState, now: float) -> None:
        self._controller.poll_kv_ready(now)

    def try_admit_decode_step(
        self, req: PDRequestState, now: float
    ) -> Tuple[int, float]:
        self._require_started()
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
            worker = self._decode_workers[idx]
        else:
            idx, worker = self._earliest_worker(self._decode_workers)
        start = max(now, worker.busy_until)
        job_id = self._next_id()
        worker.in_q.put(
            _DecodeJob(
                job_id=job_id,
                batch_size=1,
                past_kv=(int(req.current_past_kv_length),),
            )
        )
        res = self._await_result(worker, job_id, role="decode", idx=idx)
        end = start + res.duration
        worker.busy_until = end
        return idx, end

    def try_admit_decode_batch(
        self, reqs: Sequence[PDRequestState], now: float
    ) -> Tuple[int, float]:
        if not reqs:
            raise ValueError("try_admit_decode_batch requires at least one request")
        for req in reqs:
            if req.phase != RequestPhase.RUNNING_DECODE:
                raise ValueError(
                    f"cannot schedule decode batch for rid={req.rid!r} "
                    f"in phase={req.phase.value}"
                )
        self._require_started()
        if self._decode_queue_mode == "per_replica_queue":
            return self._try_admit_decode_batch_per_replica(reqs, now)
        idx, worker = self._earliest_worker(self._decode_workers)
        start = max(now, worker.busy_until)
        past_kv = tuple(int(r.current_past_kv_length) for r in reqs)
        job_id = self._next_id()
        worker.in_q.put(
            _DecodeJob(job_id=job_id, batch_size=len(reqs), past_kv=past_kv)
        )
        res = self._await_result(worker, job_id, role="decode", idx=idx)
        end = start + res.duration
        worker.busy_until = end
        return idx, end

    def _try_admit_decode_batch_per_replica(
        self, reqs: Sequence[PDRequestState], now: float
    ) -> Tuple[int, float]:
        mapping = self.bind_decode_replicas(reqs)
        replica_order = sorted(
            range(len(self._decode_workers)),
            key=lambda i: self._decode_workers[i].busy_until,
        )
        if not replica_order:
            raise ValueError("decode pool requires at least one replica")

        buckets: dict[int, list[PDRequestState]] = defaultdict(list)
        for req in reqs:
            buckets[mapping[req.rid]].append(req)

        first_replica_idx: Optional[int] = None
        max_end = now
        for replica_idx in replica_order:
            bucket = buckets.get(replica_idx)
            if not bucket:
                continue
            worker = self._decode_workers[replica_idx]
            start = max(now, worker.busy_until)
            past_kv = tuple(int(r.current_past_kv_length) for r in bucket)
            job_id = self._next_id()
            worker.in_q.put(
                _DecodeJob(job_id=job_id, batch_size=len(bucket), past_kv=past_kv)
            )
            res = self._await_result(worker, job_id, role="decode", idx=replica_idx)
            end = start + res.duration
            worker.busy_until = end
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

    def admit_decode_for_replica(
        self, replica_idx: int, rids: AbstractSet[str], now: float
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
