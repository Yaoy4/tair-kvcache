"""Phase 2b.1 — tests for AICPredictorAdapter.

The adapter wraps an `InferTimePredictor` (e.g. AIConfiguratorTimePredictor)
and exposes the simple `predict_prefill_seconds(batch_tokens)` /
`predict_decode_seconds(batch_size, past_kv_length=…)` protocol that
BackendA expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from hisim.simulation.pd_aic_adapter import AICPredictorAdapter
from hisim.time_predictor.base import FakeRequest, ScheduleBatch


@dataclass
class _RecordingPredictor:
    """Stand-in for InferTimePredictor that records every call."""
    return_value: float = 0.001
    calls: List[ScheduleBatch] = field(default_factory=list)

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        self.calls.append(batch)
        return self.return_value


def test_predict_prefill_builds_single_request_batch():
    base = _RecordingPredictor(return_value=0.005)
    adapter = AICPredictorAdapter(base)

    out = adapter.predict_prefill_seconds(512)

    assert out == 0.005
    assert len(base.calls) == 1
    batch = base.calls[0]
    assert batch.batch_size == 1
    assert batch.reqs[0].input_length == 512
    assert batch.reqs[0].past_kv_length == 0
    assert batch.is_prefill()


def test_predict_prefill_batch_preserves_request_lengths():
    base = _RecordingPredictor(return_value=0.006)
    adapter = AICPredictorAdapter(base)

    out = adapter.predict_prefill_batch_seconds([100, 300, 500])

    assert out == 0.006
    batch = base.calls[0]
    assert batch.batch_size == 3
    assert [req.input_length for req in batch.reqs] == [100, 300, 500]
    assert batch.is_prefill()


def test_predict_decode_builds_decode_batch_with_uniform_past_kv():
    base = _RecordingPredictor(return_value=0.0008)
    adapter = AICPredictorAdapter(base)

    out = adapter.predict_decode_seconds(batch_size=4, past_kv_length=128)

    assert out == 0.0008
    batch = base.calls[0]
    assert batch.batch_size == 4
    assert batch.is_decode()
    assert all(r.input_length == 1 for r in batch.reqs)
    assert [r.past_kv_length for r in batch.reqs] == [128, 128, 128, 128]


def test_predict_decode_accepts_per_request_past_kv_list():
    base = _RecordingPredictor()
    adapter = AICPredictorAdapter(base)

    adapter.predict_decode_seconds(batch_size=3, past_kv_length=[10, 20, 30])

    batch = base.calls[0]
    assert [r.past_kv_length for r in batch.reqs] == [10, 20, 30]


def test_predict_decode_rejects_mismatched_list_length():
    adapter = AICPredictorAdapter(_RecordingPredictor())
    with pytest.raises(ValueError):
        adapter.predict_decode_seconds(batch_size=3, past_kv_length=[10, 20])


def test_predict_decode_default_past_kv_is_zero():
    base = _RecordingPredictor()
    adapter = AICPredictorAdapter(base)
    adapter.predict_decode_seconds(batch_size=2)
    batch = base.calls[0]
    assert [r.past_kv_length for r in batch.reqs] == [0, 0]


def test_predict_prefill_rejects_non_positive_tokens():
    adapter = AICPredictorAdapter(_RecordingPredictor())
    with pytest.raises(ValueError):
        adapter.predict_prefill_seconds(0)
    with pytest.raises(ValueError):
        adapter.predict_prefill_seconds(-5)


def test_predict_decode_rejects_non_positive_batch_size():
    adapter = AICPredictorAdapter(_RecordingPredictor())
    with pytest.raises(ValueError):
        adapter.predict_decode_seconds(batch_size=0)


def test_predict_prefill_seconds_raises_on_negative_oom_sentinel():
    """A negative predictor result is an OOM sentinel, not a real time delta.

    Silently consuming it would corrupt the simulated clock, so the adapter
    must fail loudly instead (see pd_aic_adapter.py's guard comment).
    """
    base = _RecordingPredictor(return_value=-0.25)
    adapter = AICPredictorAdapter(base)

    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict_prefill_seconds(512)

    message = str(excinfo.value)
    assert "OOM sentinel" in message
    assert "negative prefill" in message
    assert "result=-0.250000s" in message
    assert "num_requests=1" in message
    assert "batch_tokens=512" in message
    assert "max_input_length=512" in message


def test_predict_prefill_batch_seconds_raises_on_negative_oom_sentinel():
    base = _RecordingPredictor(return_value=-1.5)
    adapter = AICPredictorAdapter(base)

    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict_prefill_batch_seconds([100, 300, 500])

    message = str(excinfo.value)
    assert "OOM sentinel" in message
    assert "num_requests=3" in message
    assert "batch_tokens=900" in message
    assert "max_input_length=500" in message


def test_predict_decode_seconds_raises_on_negative_oom_sentinel():
    """Mirrors the prefill guard: decode admission is supposed to keep

    batches within decode_kv_capacity_per_replica before ever reaching the
    predictor, so this should only fire if that gate has a genuine gap.
    """
    base = _RecordingPredictor(return_value=-0.001)
    adapter = AICPredictorAdapter(base)

    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict_decode_seconds(batch_size=4, past_kv_length=128)

    message = str(excinfo.value)
    assert "OOM sentinel" in message
    assert "negative decode" in message
    assert "result=-0.001000s" in message
    assert "batch_size=4" in message
    assert "max_past_kv=128" in message
    assert "mean_past_kv=128.0" in message
    assert "decode_kv_capacity_per_replica" in message


def test_predict_decode_seconds_raises_on_negative_oom_sentinel_with_varied_past_kv():
    base = _RecordingPredictor(return_value=-2.0)
    adapter = AICPredictorAdapter(base)

    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict_decode_seconds(batch_size=3, past_kv_length=[10, 20, 30])

    message = str(excinfo.value)
    assert "batch_size=3" in message
    assert "max_past_kv=30" in message
    assert "mean_past_kv=20.0" in message
