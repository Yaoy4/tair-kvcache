"""Small, version-tolerant helpers for SGLang request lifecycle signals.

The supported SGLang releases expose completion and abort information under
slightly different attribute names.  Keeping the compatibility logic here
makes the hook orchestration testable without importing SGLang.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Set


def request_reports_finished(req: Any) -> bool:
    """Return whether a concrete SGLang request reports terminal state."""
    for name in ("finished_reason", "finish_reason"):
        if getattr(req, name, None) is not None:
            return True

    for name in ("finished", "is_finished", "is_aborted"):
        value = getattr(req, name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value is not None and bool(value):
            return True
    return False


def abort_request_ids(message: Any) -> Set[str]:
    """Extract request IDs from a single- or batch-abort message."""
    if "abort" not in message.__class__.__name__.lower():
        return set()

    result: Set[str] = set()
    for name in ("rid", "rids", "request_id", "request_ids"):
        value = getattr(message, name, None)
        if value is None:
            continue
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, Iterable):
            result.update(str(item) for item in value if item is not None)
    return result


def request_actual_output_length(req: Any) -> int | None:
    """Read the number of output tokens retained by SGLang's stop policy."""
    for name in ("output_ids_through_stop", "output_ids"):
        value = getattr(req, name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value is not None:
            try:
                return len(value)
            except TypeError:
                continue
    return None


def reconcile_decode_progress(state: Any, actual_output_length: int) -> None:
    """Synchronize PD KV/decode counters after a SGLang decode result.

    A speculative step may accept multiple tokens, whereas the virtual backend
    schedules one forward step.  The next predictor call must nevertheless see
    the complete KV length produced by SGLang.
    """
    actual = max(int(actual_output_length), 0)
    state.decode_step_count = actual
    state.current_past_kv_length = int(state.input_length) + actual
