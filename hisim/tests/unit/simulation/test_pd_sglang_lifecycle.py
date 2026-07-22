from types import SimpleNamespace

from hisim.simulation.pd_sglang_lifecycle import (
    abort_request_ids,
    retracted_request_ids,
)


def test_abort_request_ids_supports_single_and_batch_messages():
    AbortReq = type("AbortReq", (), {})
    single = AbortReq()
    single.rid = "r1"
    assert abort_request_ids(single) == {"r1"}

    BatchAbortReq = type("BatchAbortReq", (), {})
    batch = BatchAbortReq()
    batch.rids = ["r2", "r3"]
    assert abort_request_ids(batch) == {"r2", "r3"}
    assert abort_request_ids(SimpleNamespace(rid="not-an-abort")) == set()


def _req(rid, is_retracted=False):
    return SimpleNamespace(rid=rid, is_retracted=is_retracted)


def test_retracted_request_ids_flags_only_candidates_stamped_retracted():
    waiting_queue = [
        _req("r1", is_retracted=True),
        _req("r2", is_retracted=False),
        _req("not-a-candidate", is_retracted=True),
    ]
    assert retracted_request_ids(waiting_queue, {"r1", "r2"}) == {"r1"}


def test_retracted_request_ids_empty_candidates_short_circuits():
    waiting_queue = [_req("r1", is_retracted=True)]
    assert retracted_request_ids(waiting_queue, set()) == set()


def test_retracted_request_ids_ignores_stale_flag_outside_candidates():
    # A request retracted by a *previous* call and still waiting for a fresh
    # prefill batch must not be reprocessed just because it still carries
    # is_retracted=True; it must not appear as a "new" retraction unless it
    # was actually in the candidate (running-batch) set for *this* call.
    waiting_queue = [_req("already-handled", is_retracted=True)]
    assert retracted_request_ids(waiting_queue, {"someone-else"}) == set()


def test_retracted_request_ids_handles_objects_without_rid_or_flag():
    waiting_queue = [SimpleNamespace(), _req("r1", is_retracted=True)]
    assert retracted_request_ids(waiting_queue, {"r1"}) == {"r1"}
