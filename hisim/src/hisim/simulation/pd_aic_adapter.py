"""Phase 2b.1 — AICPredictorAdapter.

Adapts a HiSim `InferTimePredictor` (e.g. `AIConfiguratorTimePredictor`)
to the lightweight protocol that `BackendA` expects:

    predict_prefill_seconds(batch_tokens: int) -> float
    predict_decode_seconds(batch_size: int, past_kv_length=...) -> float

This file has zero SGLang imports.  It does not own a predictor — it
borrows one and shapes its ScheduleBatch input.
"""
from __future__ import annotations

from typing import Iterable, Protocol, Sequence, Union

from hisim.time_predictor.base import FakeRequest, ScheduleBatch


class _PredictorLike(Protocol):
    def predict_infer_time(self, batch: ScheduleBatch) -> float: ...


PastKV = Union[int, Sequence[int]]


class AICPredictorAdapter:
    """Shape ScheduleBatch inputs for an InferTimePredictor."""

    def __init__(self, base: _PredictorLike) -> None:
        self._base = base

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    def predict_prefill_seconds(self, batch_tokens: int) -> float:
        if batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be > 0, got {batch_tokens}")
        batch = ScheduleBatch(
            reqs=[FakeRequest(input_length=int(batch_tokens), past_kv_length=0)]
        )
        return self._base.predict_infer_time(batch)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def predict_decode_seconds(
        self,
        batch_size: int,
        past_kv_length: PastKV = 0,
    ) -> float:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if isinstance(past_kv_length, int):
            pkv = [int(past_kv_length)] * batch_size
        else:
            pkv = [int(x) for x in past_kv_length]
            if len(pkv) != batch_size:
                raise ValueError(
                    f"past_kv_length list length {len(pkv)} != batch_size {batch_size}"
                )

        batch = ScheduleBatch(
            reqs=[FakeRequest(input_length=1, past_kv_length=p) for p in pkv]
        )
        return self._base.predict_infer_time(batch)
