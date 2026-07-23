from abc import ABC, abstractmethod
from collections import deque
from typing import AbstractSet, Callable, Deque, Iterable, List, Optional

from hisim.simulation.pd_transfer import KVModelConfig, TransferModel
from hisim.simulation.pd_types import PDRequestState, RequestPhase


class ExecutionBackend(ABC):
    """Adapter interface implemented by Backend A (single-process virtual time)
    and Backend B (two-process real schedulers).
    """

    @abstractmethod
    def submit_prefill_batch(
        self, reqs: Iterable[PDRequestState], now: float
    ) -> None: ...

    @abstractmethod
    def progress_decode(
        self, reqs: Iterable[PDRequestState], now: float
    ) -> None: ...

    @abstractmethod
    def advance_clock(self, now: float) -> None: ...

    @abstractmethod
    def handoff_kv(self, req: PDRequestState, now: float) -> None: ...


class PDController:
    """Backend-agnostic PD request lifecycle manager.

    Owns three queues: PREFILL_WAITING, KV_TRANSIT, DECODE_WAITING.
    Drives PDRequestState transitions; does NOT execute work itself.
    """

    def __init__(self, transfer_model: TransferModel, kv_model_cfg: KVModelConfig):
        self._transfer_model = transfer_model
        self._kv_model_cfg = kv_model_cfg
        self._prefill_waiting: Deque[PDRequestState] = deque()
        self._kv_transit: List[PDRequestState] = []
        self._decode_waiting: Deque[PDRequestState] = deque()
        self._kv_link_busy_until = 0.0

    # ---- introspection (used by tests / metrics) ----
    def prefill_waiting_count(self) -> int:
        return len(self._prefill_waiting)

    def kv_transit_count(self) -> int:
        return len(self._kv_transit)

    def decode_waiting_count(self) -> int:
        return len(self._decode_waiting)

    # ---- lifecycle hooks ----
    def on_request_arrival(self, req: PDRequestState, now: float) -> None:
        self._require_phase(req, RequestPhase.WAITING_PREFILL, "request arrival")
        if any(queued.rid == req.rid for queued in self._prefill_waiting):
            raise ValueError(f"duplicate prefill arrival for rid={req.rid!r}")
        self._prefill_waiting.append(req)

    def admit_prefill(self, capacity: int, now: float) -> List[PDRequestState]:
        admitted: List[PDRequestState] = []
        while capacity > 0 and self._prefill_waiting:
            req = self._prefill_waiting.popleft()
            self._require_phase(req, RequestPhase.WAITING_PREFILL, "prefill admission")
            req.phase = RequestPhase.RUNNING_PREFILL
            req.prefill_start_time = now
            admitted.append(req)
            capacity -= 1
        return admitted

    def on_prefill_done(
        self, req: PDRequestState, now: float, kv_ready_time: float
    ) -> None:
        self._require_phase(req, RequestPhase.RUNNING_PREFILL, "prefill completion")
        if kv_ready_time < now:
            raise ValueError(
                f"kv_ready_time ({kv_ready_time}) must be >= now ({now}); "
                "backend reported an inconsistent KV handoff time."
            )
        req.phase = RequestPhase.KV_TRANSIT
        req.prefill_end_time = now
        req.kv_ready_time = kv_ready_time
        self._kv_transit.append(req)

    def compute_kv_ready_time(self, req: PDRequestState, now: float) -> float:
        """Helper for backends that use the bandwidth transfer model directly
        (e.g., Backend A). Backend B should ignore this and report the real
        handoff completion time from its transfer process.
        """
        transfer_dur = self._transfer_model.estimate(
            req.input_length, self._kv_model_cfg
        )
        start = max(now, self._kv_link_busy_until)
        self._kv_link_busy_until = start + transfer_dur
        return self._kv_link_busy_until

    def compute_batch_kv_ready_time(self, total_tokens: int, now: float) -> float:
        """Batch-level KV transfer estimate.

        Treats the whole batch's KV data as a single serial transfer stream:
        t = latency + sum(all request tokens) * bytes_per_token / bw.
        All requests in the batch share the same kv_ready_time.
        """
        transfer_dur = self._transfer_model.estimate(total_tokens, self._kv_model_cfg)
        start = max(now, self._kv_link_busy_until)
        self._kv_link_busy_until = start + transfer_dur
        return self._kv_link_busy_until

    def poll_kv_ready(self, now: float) -> List[PDRequestState]:
        ready: List[PDRequestState] = []
        remaining: List[PDRequestState] = []
        for req in self._kv_transit:
            self._require_phase(req, RequestPhase.KV_TRANSIT, "KV-ready polling")
            if req.kv_ready_time is not None and req.kv_ready_time <= now:
                req.phase = RequestPhase.WAITING_DECODE
                self._decode_waiting.append(req)
                ready.append(req)
            else:
                remaining.append(req)
        self._kv_transit = remaining
        return ready

    def on_prefill_token_sampled(
        self, req: PDRequestState, now: float
    ) -> None:
        """Credit the first output token sampled from final-prefill logits.

        Sampling does not run a decode forward: it neither consumes decode
        capacity nor materializes another token in the KV cache.  The existing
        ``decode_step_count`` field is the total generated-token count used for
        OSL completion, despite its historical name.
        """
        self._require_phase(req, RequestPhase.KV_TRANSIT, "prefill token sampling")
        if req.output_length > 0:
            req.decode_step_count += 1
        if req.decode_step_count >= req.output_length:
            self._kv_transit = [
                queued for queued in self._kv_transit if queued.rid != req.rid
            ]
            req.phase = RequestPhase.FINISHED
            req.decode_end_time = now

    def admit_decode(self, capacity: int, now: float) -> List[PDRequestState]:
        admitted: List[PDRequestState] = []
        while capacity > 0 and self._decode_waiting:
            req = self._decode_waiting.popleft()
            self._require_phase(req, RequestPhase.WAITING_DECODE, "decode admission")
            req.phase = RequestPhase.RUNNING_DECODE
            req.decode_start_time = now
            req.current_past_kv_length = req.input_length
            admitted.append(req)
            capacity -= 1
        return admitted

    def admit_decode_targeted(
        self,
        rids: AbstractSet[str],
        now: float,
        max_count: Optional[int] = None,
        token_budget: Optional[int] = None,
        token_cost: Optional[Callable[[PDRequestState], int]] = None,
    ) -> List[PDRequestState]:
        """Admit only the requested rids from the decode-waiting queue.

        This lets the SGLang hook stamp ``decode_start_time`` only for the
        requests that are actually present in the current decode batch, instead
        of greedily popping unrelated waiters from the queue.

        If ``max_count`` is given, at most that many requests are admitted;
        excess matching rids stay in the waiting queue for the next round.

        If ``token_budget`` is given (together with ``token_cost``), a request
        is only admitted while the running sum of ``token_cost(req)`` for
        already-admitted requests stays within budget. This lets callers cap
        admission by KV-memory footprint, not just by request count, so a
        replica already near its real HBM capacity stops accepting further
        long-context requests instead of silently overcommitting it. Requests
        that don't fit stay queued for a later round -- exactly like requests
        that don't fit under ``max_count``.
        """
        if not rids:
            return []
        admitted: List[PDRequestState] = []
        remaining: Deque[PDRequestState] = deque()
        tokens_used = 0
        while self._decode_waiting:
            req = self._decode_waiting.popleft()
            self._require_phase(req, RequestPhase.WAITING_DECODE, "targeted decode admission")
            if req.rid in rids:
                over_count = max_count is not None and len(admitted) >= max_count
                over_budget = False
                cost = 0
                if not over_count and token_budget is not None and token_cost is not None:
                    cost = token_cost(req)
                    over_budget = tokens_used + cost > token_budget
                if over_count or over_budget:
                    remaining.append(req)
                    continue
                req.phase = RequestPhase.RUNNING_DECODE
                req.decode_start_time = now
                req.current_past_kv_length = req.input_length
                admitted.append(req)
                tokens_used += cost
            else:
                remaining.append(req)
        self._decode_waiting = remaining
        return admitted

    def on_decode_step_done(
        self, reqs: Iterable[PDRequestState], now: float
    ) -> None:
        for req in reqs:
            self._require_phase(req, RequestPhase.RUNNING_DECODE, "decode completion")
            # Standard HiSim sets ignore_eos=True and treats output_length as
            # the workload OSL.  One virtual decode step therefore advances
            # exactly one token; SGLang output_ids must not overwrite this
            # simulation counter.
            req.decode_step_count += 1
            req.current_past_kv_length += 1
            if req.decode_step_count >= req.output_length:
                req.phase = RequestPhase.FINISHED
                req.decode_end_time = now

    def terminate_request(self, req: PDRequestState, now: float) -> None:
        """Remove an externally aborted/cancelled request from PD state.

        Standard HiSim requests use ``ignore_eos=True`` and finish naturally
        when their OSL is reached.  This separate, idempotent transition is
        only for explicit external cancellation and removes the request from
        every controller queue before marking it finished.
        """
        self._prefill_waiting = deque(
            queued for queued in self._prefill_waiting if queued.rid != req.rid
        )
        self._kv_transit = [
            queued for queued in self._kv_transit if queued.rid != req.rid
        ]
        self._decode_waiting = deque(
            queued for queued in self._decode_waiting if queued.rid != req.rid
        )
        if req.phase == RequestPhase.FINISHED:
            return
        req.phase = RequestPhase.FINISHED
        if req.decode_start_time is not None:
            req.decode_end_time = now

    def reset_for_retract(self, req: PDRequestState, now: float) -> None:
        """Re-queue a request that SGLang retracted for KV-cache pressure.

        Real SGLang can evict an in-flight decode request's KV cache when the
        pool is full (``ScheduleBatch.retract_decode``), then re-queue the
        *same* rid so it re-enters prefill and rebuilds KV over (original
        prompt + already-generated output). Crucially, real SGLang does NOT
        discard the tokens already generated -- only their KV cache is
        rebuilt -- so this reset must only touch PD *phase* and prefill-side
        bookkeeping. ``decode_step_count`` / ``current_past_kv_length`` /
        ``output_length`` (total generation progress) are left untouched:
        resetting them would make HiSim simulate more decode steps than
        SGLang will actually run, double-counting work already done.

        Purges every phase-specific controller queue the request might
        currently sit in (whichever one HiSim's virtual PD clock has it in at
        the moment real SGLang retracts it -- KV_TRANSIT, WAITING_DECODE, or
        RUNNING_DECODE are all possible since the two clocks are decoupled),
        then resets phase to WAITING_PREFILL so the next extend batch admits
        it exactly like a fresh arrival.
        """
        self._prefill_waiting = deque(
            queued for queued in self._prefill_waiting if queued.rid != req.rid
        )
        self._kv_transit = [
            queued for queued in self._kv_transit if queued.rid != req.rid
        ]
        self._decode_waiting = deque(
            queued for queued in self._decode_waiting if queued.rid != req.rid
        )
        if req.phase == RequestPhase.FINISHED:
            raise ValueError(
                f"cannot retract already-finished request rid={req.rid!r}; "
                "this indicates PD state desynced from SGLang before retract"
            )
        req.phase = RequestPhase.WAITING_PREFILL

    @staticmethod
    def _require_phase(
        req: PDRequestState, expected: RequestPhase, operation: str
    ) -> None:
        if req.phase != expected:
            raise ValueError(
                f"invalid PD phase for {operation}: rid={req.rid!r}, "
                f"expected={expected.value}, actual={req.phase.value}"
            )
