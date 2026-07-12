from types import SimpleNamespace

from hisim.simulation.pd_sglang_lifecycle import (
    abort_request_ids,
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
