from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RequestPhase(Enum):
    WAITING_PREFILL = "waiting_prefill"
    RUNNING_PREFILL = "running_prefill"
    KV_TRANSIT = "kv_transit"
    WAITING_DECODE = "waiting_decode"
    RUNNING_DECODE = "running_decode"
    FINISHED = "finished"


@dataclass
class PDRequestState:
    rid: str
    arrival_time: float
    phase: RequestPhase = RequestPhase.WAITING_PREFILL
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    kv_ready_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    decode_step_count: int = 0
    current_past_kv_length: int = 0
    output_length: int = 0
