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


def retracted_request_ids(
    waiting_queue: Iterable[Any], candidate_rids: Set[str]
) -> Set[str]:
    """Find requests in ``waiting_queue`` that SGLang just retracted.

    When SGLang's KV-cache pool is full it evicts an in-flight decode
    request's KV cache and re-queues the same ``rid`` (stamping
    ``req.is_retracted = True``) so it re-enters prefill and rebuilds KV.
    That flag is cleared again once a fresh extend batch actually picks the
    request back up, so membership in ``waiting_queue`` with the flag still
    set is a precise, version-tolerant retraction signal.

    ``candidate_rids`` should be the rids that were in the running (decode)
    batch immediately before the call that may have triggered retraction.
    Scoping to that set ensures an already-retracted request still sitting in
    the waiting queue from a *previous* call (not yet re-admitted to a fresh
    prefill batch) is never reprocessed a second time.
    """
    if not candidate_rids:
        return set()
    result: Set[str] = set()
    for req in waiting_queue:
        rid = getattr(req, "rid", None)
        if rid is None or rid not in candidate_rids:
            continue
        if getattr(req, "is_retracted", False):
            result.add(rid)
    return result
