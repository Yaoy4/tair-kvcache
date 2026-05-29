from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Iterable, List

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

    # ---- introspection (used by tests / metrics) ----
    def prefill_waiting_count(self) -> int:
        return len(self._prefill_waiting)

    def kv_transit_count(self) -> int:
        return len(self._kv_transit)

    def decode_waiting_count(self) -> int:
        return len(self._decode_waiting)

    # ---- lifecycle hooks ----
    def on_request_arrival(self, req: PDRequestState, now: float) -> None:
        req.phase = RequestPhase.WAITING_PREFILL
        self._prefill_waiting.append(req)

    def admit_prefill(self, capacity: int, now: float) -> List[PDRequestState]:
        admitted: List[PDRequestState] = []
        while capacity > 0 and self._prefill_waiting:
            req = self._prefill_waiting.popleft()
            req.phase = RequestPhase.RUNNING_PREFILL
            req.prefill_start_time = now
            admitted.append(req)
            capacity -= 1
        return admitted

    def on_prefill_done(self, req: PDRequestState, now: float) -> None:
        req.phase = RequestPhase.KV_TRANSIT
        req.prefill_end_time = now
        transfer_dur = self._transfer_model.estimate(
            req.input_length, self._kv_model_cfg
        )
        req.kv_ready_time = now + transfer_dur
        self._kv_transit.append(req)

    def poll_kv_ready(self, now: float) -> List[PDRequestState]:
        ready: List[PDRequestState] = []
        remaining: List[PDRequestState] = []
        for req in self._kv_transit:
            if req.kv_ready_time is not None and req.kv_ready_time <= now:
                req.phase = RequestPhase.WAITING_DECODE
                self._decode_waiting.append(req)
                ready.append(req)
            else:
                remaining.append(req)
        self._kv_transit = remaining
        return ready

    def admit_decode(self, capacity: int, now: float) -> List[PDRequestState]:
        admitted: List[PDRequestState] = []
        while capacity > 0 and self._decode_waiting:
            req = self._decode_waiting.popleft()
            req.phase = RequestPhase.RUNNING_DECODE
            req.decode_start_time = now
            req.current_past_kv_length = req.input_length
            admitted.append(req)
            capacity -= 1
        return admitted

    def on_decode_step_done(
        self, reqs: Iterable[PDRequestState], now: float
    ) -> None:
        for req in reqs:
            req.decode_step_count += 1
            req.current_past_kv_length += 1
            if req.decode_step_count >= req.output_length:
                req.phase = RequestPhase.FINISHED
