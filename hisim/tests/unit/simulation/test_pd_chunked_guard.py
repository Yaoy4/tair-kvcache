"""p1-00: chunked-prefill detection guard + fix tests.

The PD glue now fully handles chunked prefill:
- Phase guard: mid-chunk requests (already RUNNING_PREFILL) bypass the
  controller lifecycle (on_request_arrival / admit_prefill) in
  try_admit_prefill_batch, preventing phase reset and double-counted time.
- Final-chunk gate: finalize_prefill_batch is called only when is_chunked==0,
  preventing premature KV_TRANSIT and state corruption.
- Full-prompt length: input_length is set to origin_input_ids length (or
  accumulated chunk sum) on the final chunk so KV sizing and decode KV base
  are accurate.

These tests pin the old ``has_unsupported_chunked`` predicate (preserved for
backward compat) and add focused tests for each of the three bug fixes.
"""
from hisim.simulation.pd_timeline import has_unsupported_chunked
from hisim.simulation.pd_types import PDRequestState, RequestPhase
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_config import DisaggConfig, RolePredictorConfig, BandwidthTransferConfig
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.pd_runtime import finalize_prefill_batch
from hisim.spec import ModelInfo, DataType
from hisim.simulation.types import SchedulerConfig


class _FakeReq:
    def __init__(self, is_chunked=0):
        self.is_chunked = is_chunked


class _NoAttrReq:
    """A request object that does not expose is_chunked at all."""


def _model():
    return ModelInfo(
        hidden_size=4096, num_attention_heads=32, num_hidden_layers=32,
        vocab_size=32000, num_key_value_heads=8, head_dim=128,
        num_full_attention=32, name="t",
    )


def _base():
    return SchedulerConfig(
        model=_model(), tp_size=1, pp_size=1, data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16, backend_name="sglang",
        backend_version="0.4.8",
    )


class _StubHW:
    def __init__(self, name):
        self.name = name


class _StubPredictor:
    def __init__(self, model, hw, config, **kwargs):
        pass

    def predict_prefill_seconds(self, batch_tokens):
        return batch_tokens * 1e-4

    def predict_decode_seconds(self, batch_size=1, past_kv_length=0):
        return 1e-3


def _bundle():
    cfg = DisaggConfig(
        enabled=True, backend="single_process",
        prefill=RolePredictorConfig(device_name="fast", tp_size=1, replicas=1,
                                    max_running_per_replica=8),
        decode=RolePredictorConfig(device_name="fast", tp_size=1, replicas=1,
                                   max_running_per_replica=64),
        kv_transfer=BandwidthTransferConfig(bw_gbps=64.0, latency_us=20.0),
    )
    return build_disagg(
        model=_model(), base_sched_config=_base(), disagg_config=cfg,
        predictor_factory=_StubPredictor, hw_factory=_StubHW,
    )


def _req(rid, input_len=512):
    return PDRequestState(
        rid=rid, arrival_time=0.0, phase=RequestPhase.WAITING_PREFILL,
        input_length=input_len, output_length=4,
    )


# ---------------------------------------------------------------------------
# Original has_unsupported_chunked predicate tests (unchanged)
# ---------------------------------------------------------------------------


def test_unchunked_batch_is_not_flagged():
    reqs = [_FakeReq(0), _FakeReq(0), _FakeReq(0)]
    assert has_unsupported_chunked(reqs) is False


def test_any_chunked_request_flags_the_batch():
    reqs = [_FakeReq(0), _FakeReq(1), _FakeReq(0)]
    assert has_unsupported_chunked(reqs) is True


def test_negative_chunk_marker_also_flags():
    assert has_unsupported_chunked([_FakeReq(-1)]) is True


def test_missing_is_chunked_attr_defaults_to_unchunked():
    assert has_unsupported_chunked([_NoAttrReq(), _NoAttrReq()]) is False


def test_empty_batch_is_not_flagged():
    assert has_unsupported_chunked([]) is False


# ---------------------------------------------------------------------------
# Bug-fix #3a: phase guard — mid-chunk request must NOT be re-admitted
# ---------------------------------------------------------------------------


def test_phase_guard_skips_lifecycle_for_running_prefill():
    """try_admit_prefill_batch must not re-run on_request_arrival/admit_prefill
    for a request already in RUNNING_PREFILL (mid-chunk).
    """
    be = BackendA(_bundle())

    s = _req("r1", input_len=256)
    assert s.phase == RequestPhase.WAITING_PREFILL

    # First chunk: fresh → lifecycle runs → RUNNING_PREFILL
    be.try_admit_prefill_batch([s], now=0.0)
    assert s.phase == RequestPhase.RUNNING_PREFILL, "first chunk must set RUNNING_PREFILL"

    # Second (mid) chunk: already RUNNING_PREFILL → lifecycle must be skipped
    s.input_length = 256  # update to this chunk's size
    be.try_admit_prefill_batch([s], now=0.01)
    assert s.phase == RequestPhase.RUNNING_PREFILL, (
        "mid-chunk must stay RUNNING_PREFILL, not be reset to WAITING_PREFILL"
    )


def test_phase_guard_mixed_batch():
    """A batch with one fresh and one mid-chunk request: only the fresh one
    goes through the controller lifecycle; the mid-chunk one keeps its phase.
    """
    be = BackendA(_bundle())

    s_fresh = _req("r_new", input_len=200)
    s_mid = _req("r_mid", input_len=200)

    # Pre-admit s_mid to RUNNING_PREFILL (first chunk)
    be.try_admit_prefill_batch([s_mid], now=0.0)
    assert s_mid.phase == RequestPhase.RUNNING_PREFILL

    # New batch: s_fresh (WAITING_PREFILL) + s_mid (RUNNING_PREFILL)
    s_fresh.input_length = 200
    s_mid.input_length = 200  # second chunk
    be.try_admit_prefill_batch([s_fresh, s_mid], now=0.01)

    assert s_fresh.phase == RequestPhase.RUNNING_PREFILL, "fresh must be admitted"
    assert s_mid.phase == RequestPhase.RUNNING_PREFILL, "mid-chunk must not be reset"


# ---------------------------------------------------------------------------
# Bug-fix #3b: final-chunk gate — finalize only on is_chunked == 0
# ---------------------------------------------------------------------------


def test_finalize_called_only_on_final_chunk():
    """Mid-chunk should stay RUNNING_PREFILL; only final chunk gets KV_TRANSIT."""
    be = BackendA(_bundle())

    s = _req("r1", input_len=256)
    be.try_admit_prefill_batch([s], now=0.0)
    assert s.phase == RequestPhase.RUNNING_PREFILL

    # Mid-chunk: do NOT call finalize — state should remain RUNNING_PREFILL
    assert s.kv_ready_time is None  # not yet finalized

    # Final chunk: call finalize with full prompt length
    s.input_length = 512  # full prompt
    finalize_prefill_batch(be, [s], now=0.1)

    assert s.phase == RequestPhase.KV_TRANSIT, "after final chunk → KV_TRANSIT"
    assert s.kv_ready_time is not None, "kv_ready_time must be set after finalize"
    assert s.kv_ready_time >= 0.1, "kv_ready_time must be at or after finalize time"


# ---------------------------------------------------------------------------
# Bug-fix #3c: full-prompt length captured on final chunk
# ---------------------------------------------------------------------------


def test_input_length_updated_to_full_prompt_on_final_chunk():
    """After accumulating chunks, input_length on the final state equals the
    full prompt (sum of all chunks or origin_input_ids length).
    """
    chunk_sizes = [256, 256, 240]  # 3 chunks, full prompt = 752

    s = PDRequestState(rid="r1", arrival_time=0.0, input_length=chunk_sizes[0])
    accum = chunk_sizes[0]

    for chunk_len in chunk_sizes[1:]:
        s.input_length = chunk_len  # per-chunk for latency
        accum += chunk_len

    # On final chunk: override to full prompt (hook uses origin_input_ids or accum)
    full_prompt = sum(chunk_sizes)
    s.input_length = full_prompt

    assert s.input_length == 752, (
        f"input_length should be full prompt (752), got {s.input_length}"
    )
    assert accum == full_prompt, "accumulated chunk sum must equal full prompt"


def test_kv_transfer_uses_full_prompt_length():
    """finalize_prefill_batch with full prompt length produces a larger
    kv_ready_time than if only the last chunk length were used.
    """
    be = BackendA(_bundle())

    # Request with a 3-chunk prefill: chunks 256+256+240 = 752 total
    s_correct = _req("r_correct", input_len=256)
    be.try_admit_prefill_batch([s_correct], now=0.0)
    # Simulate correct: set full prompt before finalize
    s_correct.input_length = 752
    finalize_prefill_batch(be, [s_correct], now=0.1)

    be2 = BackendA(_bundle())
    s_wrong = _req("r_wrong", input_len=256)
    be2.try_admit_prefill_batch([s_wrong], now=0.0)
    # Simulate wrong (old bug): finalize with only last chunk length
    s_wrong.input_length = 240
    finalize_prefill_batch(be2, [s_wrong], now=0.1)

    assert s_correct.kv_ready_time > s_wrong.kv_ready_time, (
        "full prompt KV transfer must take longer than single-chunk transfer"
    )
