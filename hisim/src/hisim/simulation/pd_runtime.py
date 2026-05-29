"""Phase 2b.3 — PD runtime builder.

Thin wrapper that lets the SGLang hook construct a BackendA in one call
without having to know about predictor factories or HW lookup. By default it
wraps real `AIConfiguratorTimePredictor` instances in `AICPredictorAdapter`,
but `predictor_factory` and `hw_factory` are injectable for unit tests.

This module is hook-agnostic (no SGLang imports) so it can be exercised by
fast unit tests.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

from hisim.spec.model import ModelInfo
from hisim.simulation.pd_aic_adapter import AICPredictorAdapter
from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_backend_protocol import PDBackendProtocol
from hisim.simulation.pd_config import DisaggConfig
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.pd_types import PDRequestState
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
# BackendB construction (Phase 5c.1)
# ---------------------------------------------------------------------------


# Type aliases for the BackendB build path.
RoleFactory = Callable[[ModelInfo, SchedulerConfig, object], Callable[[], object]]
"""Builds a picklable per-role predictor factory.

Signature: ``(model, base_sched_config, role_cfg) -> picklable_factory``.
The returned factory must be top-level / picklable so it can be sent into a
spawned worker, and calling it must produce an object satisfying the
adapter protocol used by ``BackendB`` workers (``predict_prefill_seconds`` /
``predict_decode_seconds``).
"""


def _noop_bundle_predictor(*_args, **_kwargs):
    """BackendB never calls the bundle's prefill/decode predictors — workers
    build their own. Returning ``None`` here avoids a wasteful AIC DB load in
    the parent process while still letting ``build_disagg`` populate the
    transfer model, KV bytes-per-token, and replica counts.
    """
    return None


def _default_role_factory(
    model: ModelInfo,
    base_sched_config: SchedulerConfig,
    role,
):
    # Lazy import: AICPredictorFactory pulls hisim.spec which is heavy.
    from hisim.simulation.pd_aic_factory import AICPredictorFactory

    return AICPredictorFactory(
        model=model,
        base_sched_config=base_sched_config,
        role=role,
    )


def build_backend_b(
    *,
    model: ModelInfo,
    base_sched_config: SchedulerConfig,
    disagg_config: DisaggConfig,
    role_factory: Optional[RoleFactory] = None,
    hw_factory: Optional[HWFactory] = None,
    mp_context=None,
):
    """Build a BackendB from the given configs.

    The returned BackendB is **not** started; the caller must invoke
    ``.start()`` (and own ``.shutdown()`` cleanup — see 5c.2).

    Raises ValueError if ``disagg_config.enabled`` is False.
    """
    # Lazy import to keep test envs that never touch multiprocessing clean.
    from hisim.simulation.pd_backend_b import BackendB

    bundle = build_disagg(
        model=model,
        base_sched_config=base_sched_config,
        disagg_config=disagg_config,
        predictor_factory=_noop_bundle_predictor,
        hw_factory=hw_factory,
    )
    role_factory = role_factory or _default_role_factory
    prefill_factory = role_factory(model, base_sched_config, disagg_config.prefill)
    decode_factory = role_factory(model, base_sched_config, disagg_config.decode)
    return BackendB(
        bundle=bundle,
        prefill_predictor_factory=prefill_factory,
        decode_predictor_factory=decode_factory,
        mp_context=mp_context,
    )


# ---------------------------------------------------------------------------
# Backend dispatcher (Phase 5c.1)
# ---------------------------------------------------------------------------


def build_pd_backend(
    *,
    model: ModelInfo,
    base_sched_config: SchedulerConfig,
    disagg_config: DisaggConfig,
    predictor_factory: Optional[PredictorFactory] = None,
    hw_factory: Optional[HWFactory] = None,
    role_factory: Optional[RoleFactory] = None,
    mp_context=None,
) -> PDBackendProtocol:
    """Build the PD backend selected by ``disagg_config.backend``.

    * ``single_process`` → :func:`build_backend_a` (uses ``predictor_factory``).
    * ``two_process``    → :func:`build_backend_b` (uses ``role_factory``).

    The returned backend is not started — caller owns lifecycle. This split
    keeps ``sglang_hook`` agnostic to which backend was selected, while
    letting it call ``.start()`` only when needed (BackendB).

    Raises ValueError if ``disagg_config.enabled`` is False or
    ``disagg_config.backend`` is unrecognized.
    """
    if not disagg_config.enabled:
        raise ValueError("build_pd_backend called with DisaggConfig.enabled=False")

    backend_name = disagg_config.backend
    if backend_name == "single_process":
        return build_backend_a(
            model=model,
            base_sched_config=base_sched_config,
            disagg_config=disagg_config,
            predictor_factory=predictor_factory,
            hw_factory=hw_factory,
        )
    if backend_name == "two_process":
        return build_backend_b(
            model=model,
            base_sched_config=base_sched_config,
            disagg_config=disagg_config,
            role_factory=role_factory,
            hw_factory=hw_factory,
            mp_context=mp_context,
        )
    raise ValueError(
        f"unknown disagg.backend: {backend_name!r} "
        f"(expected 'single_process' or 'two_process')"
    )


# ---------------------------------------------------------------------------
# Hook glue (Phase 2b.4a)
# ---------------------------------------------------------------------------


def admit_prefill_batch_latency(
    backend: PDBackendProtocol,
    states: Sequence[PDRequestState],
    now: float,
) -> float:
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


def decode_batch_latency(
    backend: PDBackendProtocol,
    states: Sequence[PDRequestState],
    now: float,
) -> float:
    """Schedule one decode step for the whole batch via BackendA.

    Returns the predicted step latency = end_time - now. Empty batches yield 0.
    """
    if not states:
        return 0.0
    _, end_t = backend.try_admit_decode_batch(states, now)
    return end_t - now


def finalize_prefill_batch(
    backend: PDBackendProtocol,
    states: Iterable[PDRequestState],
    now: float,
) -> None:
    """After a prefill batch completes at virtual time `now`, move each
    request to KV_TRANSIT with its computed kv_ready_time.
    """
    for s in states:
        kv_ready = backend.compute_kv_ready_time(s, now)
        backend.on_prefill_done(s, now, kv_ready)


def drain_kv_ready_and_admit_decode(
    backend: PDBackendProtocol, now: float, capacity: int
) -> List[PDRequestState]:
    """Promote any KV-ready requests to WAITING_DECODE, then admit up to
    `capacity` of them into RUNNING_DECODE. Returns the admitted list.
    """
    ctrl = backend.controller()
    ctrl.poll_kv_ready(now)
    return ctrl.admit_decode(capacity=capacity, now=now)
