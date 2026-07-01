"""p1-00: chunked-prefill detection guard.

The PD glue assumes one extend batch per request. SGLang's chunked prefill
spreads a request across several extend batches (flagging non-final chunks with
``is_chunked != 0``), which would corrupt PD input_length / kv_ready / decode
base. ``has_unsupported_chunked`` lets the hook detect this and warn loudly
instead of silently emitting bad PD numbers. These tests pin the predicate so
the hook's guard cannot silently regress.
"""
from hisim.simulation.pd_timeline import has_unsupported_chunked


class _FakeReq:
    def __init__(self, is_chunked=0):
        self.is_chunked = is_chunked


class _NoAttrReq:
    """A request object that does not expose is_chunked at all."""


def test_unchunked_batch_is_not_flagged():
    reqs = [_FakeReq(0), _FakeReq(0), _FakeReq(0)]
    assert has_unsupported_chunked(reqs) is False


def test_any_chunked_request_flags_the_batch():
    reqs = [_FakeReq(0), _FakeReq(1), _FakeReq(0)]
    assert has_unsupported_chunked(reqs) is True


def test_negative_chunk_marker_also_flags():
    # SGLang uses non-zero (not strictly positive) to mark non-final chunks.
    assert has_unsupported_chunked([_FakeReq(-1)]) is True


def test_missing_is_chunked_attr_defaults_to_unchunked():
    assert has_unsupported_chunked([_NoAttrReq(), _NoAttrReq()]) is False


def test_empty_batch_is_not_flagged():
    assert has_unsupported_chunked([]) is False
