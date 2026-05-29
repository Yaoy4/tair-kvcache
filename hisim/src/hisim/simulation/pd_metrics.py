"""Backend-agnostic stage-duration helpers for PD disaggregation.

Pure semantic layer — must not import SGLang or backend-specific code.
Used by both Backend A and (future) Backend B adapters to translate
PDRequestState timestamps into RequestStats stage fields, and ultimately
into calc_metrics percentile aggregates.

Stage durations:
- prefill_queue_wait = prefill_start_time - arrival_time
- kv_transfer_time   = kv_ready_time - prefill_end_time
- decode_queue_wait  = decode_start_time - kv_ready_time

Any missing timestamp yields 0.0 for that field. Negative differences
are clamped to 0.0 so downstream percentile aggregation stays well-defined
even if a backend reports slightly inconsistent timings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from hisim.simulation.pd_types import PDRequestState
    from hisim.simulation.types import RequestStats


def _delta(end, start) -> float:
    if end is None or start is None:
        return 0.0
    d = float(end) - float(start)
    return d if d > 0.0 else 0.0


def compute_stage_durations(state: "PDRequestState") -> Dict[str, float]:
    return {
        "prefill_queue_wait": _delta(state.prefill_start_time, state.arrival_time),
        "kv_transfer_time": _delta(state.kv_ready_time, state.prefill_end_time),
        "decode_queue_wait": _delta(state.decode_start_time, state.kv_ready_time),
    }


def populate_request_stats(
    stats: "RequestStats", state: "PDRequestState"
) -> None:
    d = compute_stage_durations(state)
    stats.prefill_queue_wait = d["prefill_queue_wait"]
    stats.kv_transfer_time = d["kv_transfer_time"]
    stats.decode_queue_wait = d["decode_queue_wait"]
