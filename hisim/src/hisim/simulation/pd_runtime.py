"""Phase 2b.3 — PD runtime builder.

Thin wrapper that lets the SGLang hook construct a BackendA in one call
without having to know about predictor factories or HW lookup. By default it
wraps real `AIConfiguratorTimePredictor` instances in `AICPredictorAdapter`,
but `predictor_factory` and `hw_factory` are injectable for unit tests.

This module is hook-agnostic (no SGLang imports) so it can be exercised by
fast unit tests.
"""
from __future__ import annotations

from typing import Callable, Optional

from hisim.spec.model import ModelInfo
from hisim.simulation.pd_aic_adapter import AICPredictorAdapter
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_config import DisaggConfig
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.types import SchedulerConfig

PredictorFactory = Callable[..., object]
HWFactory = Callable[[str], object]


def _default_predictor_factory(model, hw, config, **kwargs):
    """Construct a real AIC predictor and wrap it in the BackendA protocol."""
    # Imported lazily so unit tests can avoid the AIC perf DB dependency.
    from hisim.time_predictor import AIConfiguratorTimePredictor

    base = AIConfiguratorTimePredictor(model, hw=hw, config=config, **kwargs)
    return AICPredictorAdapter(base)


def build_backend_a(
    *,
    model: ModelInfo,
    base_sched_config: SchedulerConfig,
    disagg_config: DisaggConfig,
    predictor_factory: Optional[PredictorFactory] = None,
    hw_factory: Optional[HWFactory] = None,
) -> BackendA:
    """Build a BackendA from the given configs.

    Raises ValueError if `disagg_config.enabled` is False.
    """
    bundle = build_disagg(
        model=model,
        base_sched_config=base_sched_config,
        disagg_config=disagg_config,
        predictor_factory=predictor_factory or _default_predictor_factory,
        hw_factory=hw_factory,
    )
    return BackendA(bundle)


# ---------------------------------------------------------------------------
# Hook glue (Phase 2b.4a)
# ---------------------------------------------------------------------------


def admit_prefill_batch_latency(backend, states, now: float) -> float:
    """Admit every request in `states` onto the backend's prefill pool.

    Returns the predicted batch latency = (max prefill end time) - now.
    Returns 0.0 for an empty batch.
    """
    if not states:
        return 0.0
    max_end = now
    for s in states:
        _, end_t = backend.try_admit_prefill(s, now)
        if end_t > max_end:
            max_end = end_t
    return max_end - now
