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
        return self.predict_prefill_batch_seconds([int(batch_tokens)])

    def predict_prefill_batch_seconds(
        self, input_lengths: Sequence[int]
    ) -> float:
        """Predict one fused prefill batch without collapsing its requests."""
        lengths = [int(length) for length in input_lengths]
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError(
                f"input_lengths must contain positive lengths, got {lengths!r}"
            )
        batch = ScheduleBatch(
            reqs=[
                FakeRequest(input_length=length, past_kv_length=0)
                for length in lengths
            ]
        )
        result = self._base.predict_infer_time(batch)
        if result < 0:
            raise RuntimeError(
                "AIConfigurator predictor returned a negative prefill "
                f"latency (OOM sentinel): result={result:.6f}s "
                f"num_requests={len(lengths)} batch_tokens={sum(lengths)} "
                f"max_input_length={max(lengths)}. This means the modeled "
                "batch's memory footprint (weights + activations + KV "
                "cache) exceeds one GPU's real HBM capacity -- see "
                "aiconfigurator's InferenceSummary.check_oom(). Consuming "
                "a negative value as a literal time delta would silently "
                "corrupt the owning replica's simulated clock, so this is "
                "raised loudly instead; prefill admission should be made "
                "token-budget aware if this fires in practice."
            )
        return result

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
        result = self._base.predict_infer_time(batch)
        if result < 0:
            # The underlying predictor negates its latency estimate as an
            # OOM sentinel when a batch's memory footprint would exceed one
            # GPU's real HBM (aiconfigurator's InferenceSummary.check_oom()).
            # Silently consuming that negative value as a literal time delta
            # previously corrupted the owning decode replica's simulated
            # clock permanently (it could even go absolutely negative),
            # producing widespread negative per-token/TPOT/E2E latencies for
            # every request sharing that replica's later batches. Backend
            # A's decode admission is now KV-token-capacity aware
            # specifically to keep batches within real HBM and never reach
            # this state -- if it fires anyway that is a genuine gap in the
            # admission gate, so fail loudly with full diagnostics instead
            # of silently corrupting simulated time.
            raise RuntimeError(
                "AIConfigurator predictor returned a negative decode "
                f"latency (OOM sentinel): result={result:.6f}s "
                f"batch_size={batch_size} max_past_kv={max(pkv)} "
                f"mean_past_kv={sum(pkv) / len(pkv):.1f}. Decode admission "
                "should have refused this batch via "
                "decode_kv_capacity_per_replica before reaching the "
                "predictor -- check that the DisaggPredictors bundle wires "
                "decode_kv_capacity_per_replica through for this topology."
            )
        return result
