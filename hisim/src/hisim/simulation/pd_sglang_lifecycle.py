"""Small, version-tolerant helpers for SGLang request lifecycle signals.

The supported SGLang releases expose completion and abort information under
slightly different attribute names.  Keeping the compatibility logic here
makes the hook orchestration testable without importing SGLang.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Set


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
