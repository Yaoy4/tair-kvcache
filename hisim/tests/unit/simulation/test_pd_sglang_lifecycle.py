from types import SimpleNamespace

from hisim.simulation.pd_sglang_lifecycle import (
    abort_request_ids,
    reconcile_decode_progress,
    request_actual_output_length,
    request_reports_finished,
)


def test_request_reports_finished_accepts_reason_and_method_variants():
    assert request_reports_finished(SimpleNamespace(finished_reason="eos"))
    assert request_reports_finished(SimpleNamespace(finish_reason="stop"))

    class MethodVariant:
        def finished(self):
            return object()

    assert request_reports_finished(MethodVariant())
    assert not request_reports_finished(SimpleNamespace(finished_reason=None))


def test_request_reports_finished_ignores_incompatible_method_signature():
    class Variant:
        def is_finished(self, required_arg):
            return True

    assert not request_reports_finished(Variant())


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


def test_request_actual_output_length_prefers_stop_trimmed_tokens():
    req = SimpleNamespace(
        output_ids_through_stop=[1, 2],
        output_ids=[1, 2, 3, 4],
    )
    assert request_actual_output_length(req) == 2
    assert request_actual_output_length(SimpleNamespace(output_ids=[1, 2, 3])) == 3
    assert request_actual_output_length(SimpleNamespace()) is None


def test_reconcile_decode_progress_handles_multi_token_step():
    state = SimpleNamespace(
        input_length=100,
        decode_step_count=1,
        current_past_kv_length=101,
    )
    reconcile_decode_progress(state, 4)
    assert state.decode_step_count == 4
    assert state.current_past_kv_length == 104
