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
    # Timestamp at which the request entered the server-side scheduling
    # queue.  This is deliberately separate from ``arrival_time`` (request
    # creation / end-to-end baseline): prefill queue wait must not include
    # client or transport time.  Older callers may omit it; metrics then fall
    # back to ``arrival_time`` for backward compatibility.
    prefill_queue_start_time: Optional[float] = None
    phase: RequestPhase = RequestPhase.WAITING_PREFILL
    input_length: int = 0
    # Full prompt token count, stable across every chunk of a chunked
    # prefill -- unlike ``input_length``, which the hook deliberately
    # overwrites to each chunk's OWN token count for per-chunk latency
    # accuracy (one forward pass ~= chunk_len tokens). Admission-time KV
    # budgeting (``_decode_token_cost``) needs the TRUE total footprint a
    # request will occupy once fully prefilled, which for any chunked
    # (multi-forward-pass) prefill is NOT the same as the first chunk's
    # ``input_length``. Left as None when the caller doesn't have (or
    # doesn't need) the distinction, in which case ``input_length`` is used
    # as a fallback -- exact for single-chunk requests and for every
    # pre-existing test/caller that only ever sets ``input_length``.
    total_input_length: Optional[int] = None
    # Scheduling metadata for the current prefill chunk.  The hook refreshes
    # ``prefill_is_final_chunk`` before every extend call; backends stamp the
    # replica and replica-local fused-batch id.
    prefill_is_final_chunk: bool = True
    prefill_replica_idx: Optional[int] = None
    prefill_batch_id: Optional[int] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    kv_ready_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    decode_end_time: Optional[float] = None
    decode_step_count: int = 0
    current_past_kv_length: int = 0
    output_length: int = 0

