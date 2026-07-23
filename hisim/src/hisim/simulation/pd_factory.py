"""PD predictor + transfer-model factory.

Given a DisaggConfig, builds two role-specific predictors (prefill / decode)
and derives kv_bytes_per_token via the existing utils helpers — never
hand-picked. The predictor and HW-lookup constructors are injectable so unit
tests can avoid loading real AIConfigurator perf databases.

This module is the wiring layer; it depends on hisim.spec / hisim.time_predictor
but stays free of sglang imports so it can be exercised from both Backend A and
Backend B without dragging the SGLang runtime into tests.
"""

from __future__ import annotations

import dataclasses
import copy
from typing import Callable, Optional

from hisim.spec import AcceleratorInfo, DataType, ModelInfo
from hisim.simulation.types import SchedulerConfig
from hisim.simulation.utils import (
    calc_kv_cache_cell_elems,
    estimate_kv_cache_pool_capacity,
)
from hisim.simulation.pd_config import DisaggConfig, RolePredictorConfig
from hisim.simulation.pd_transfer import BandwidthTransferModel, KVModelConfig
from hisim.utils import get_logger

logger = get_logger("hisim")


PredictorFactory = Callable[..., object]
HWFactory = Callable[[str], object]


@dataclasses.dataclass
class DisaggPredictors:
    prefill: object
    decode: object
    kv_bytes_per_token: int
    kv_model_config: KVModelConfig
    transfer_model: BandwidthTransferModel
    prefill_replicas: int = 1
    decode_replicas: int = 1
    decode_queue_mode: str = "single_replica"
    prefill_max_running_per_replica: Optional[int] = None
    decode_max_running_per_replica: Optional[int] = None
    # Real per-accelerator KV-token budget for ONE decode replica (tp/pp-size
    # aware; same formula as override_initialize's aggregate KV estimate,
    # computed once here for the decode role specifically). None means the
    # caller opted out of KV-aware decode admission (e.g. older unit tests
    # that build DisaggPredictors by hand) -- Backend A treats that as
    # "no per-replica KV gating", matching the pre-existing behaviour.
    decode_kv_capacity_per_replica: Optional[int] = None
    # Backward-compatible alias used by older tests/callers.  New backends
    # prefer the role-specific fields above.
    max_running_per_replica: int = (1 << 31) - 1


def _as_dtype(value) -> Optional[DataType]:
    if value is None or isinstance(value, DataType):
        return value
    return DataType(str(value).upper())


def calc_kv_bytes_per_token(
    model: ModelInfo,
    tp_size: int,
    pp_size: int,
    kv_cache_dtype: DataType,
) -> int:
    """Derive bytes-per-token for the KV cache via the canonical utils helpers.

    Mirrors ConfigManager.get_kv_cache_bytes but parameterised so PD can use
    per-role tp_size without depending on the global ConfigManager singleton.
    """
    cell_elems = calc_kv_cache_cell_elems(model, tp_size, pp_size)
    return int(cell_elems * _as_dtype(kv_cache_dtype).bytes)


_HW_FACTORY_WARNED: set[str] = set()


def _default_hw_factory(name: str):
    hw = AcceleratorInfo.find_by_hw_name(name)
    if hw is not None:
        return hw

    # Some perf-db device names (for example "b60" or any newly-added device
    # under aiconfigurator/systems/data/) may not be present in HiSim's static
    # accelerator registry. For predictor-only simulation, fall back to a known
    # hw profile and preserve the requested device name so AIConfigurator still
    # queries the intended perf database. To register the device first-class in
    # HiSim, add it to hisim/src/hisim/spec/accelerator/info.py.
    all_hws = AcceleratorInfo.list_all_hws()
    fallback = AcceleratorInfo.find_by_hw_name("h20_sxm")
    if fallback is None and all_hws:
        fallback = next(iter(all_hws.values()))
    if fallback is None:
        raise ValueError(
            f"Unknown accelerator name {name!r}. Available: "
            f"{sorted(all_hws.keys())}"
        )
    if name not in _HW_FACTORY_WARNED:
        _HW_FACTORY_WARNED.add(name)
        import warnings
        warnings.warn(
            f"Accelerator {name!r} not in HiSim AcceleratorInfo registry;"
            f" using {fallback.name!r} profile as hw label. AIC perf-db lookup"
            f" still uses {name!r}. To remove this warning, register the"
            f" device in hisim/spec/accelerator/info.py.",
            stacklevel=2,
        )
    cloned = copy.deepcopy(fallback)
    cloned.name = name
    return cloned


def _build_role_sched_config(
    base: SchedulerConfig,
    role: RolePredictorConfig,
    model: ModelInfo,
) -> SchedulerConfig:
    sched = copy.copy(base)
    sched.model = model
    sched.tp_size = role.tp_size
    sched.ep_size = role.ep_size
    sched.dp_size = role.dp_size
    sched.pp_size = role.pp_size
    sched.max_running_requests = role.max_running_per_replica
    if role.data_type is not None:
        sched.data_type = _as_dtype(role.data_type)
    if role.kv_cache_data_type is not None:
        sched.kv_cache_data_type = _as_dtype(role.kv_cache_data_type)
    if role.backend_version is not None:
        sched.backend_version = role.backend_version
    return sched


def _build_role_predictor(
    *,
    model: ModelInfo,
    base_sched: SchedulerConfig,
    role: RolePredictorConfig,
    predictor_factory: PredictorFactory,
    hw_factory: HWFactory,
):
    hw = hw_factory(role.device_name)
    sched = _build_role_sched_config(base_sched, role, model)
    kwargs = {}
    if role.database_path is not None:
        kwargs["database_path"] = role.database_path
    kwargs["prefill_scale_factor"] = role.prefill_scale_factor
    kwargs["decode_scale_factor"] = role.decode_scale_factor
    kwargs["prefill_overhead_ms"] = role.prefill_overhead_ms
    kwargs["decode_overhead_ms"] = role.decode_overhead_ms
    return predictor_factory(model, hw=hw, config=sched, **kwargs)


def build_disagg(
    *,
    model: ModelInfo,
    base_sched_config: SchedulerConfig,
    disagg_config: DisaggConfig,
    predictor_factory: Optional[PredictorFactory] = None,
    hw_factory: Optional[HWFactory] = None,
) -> DisaggPredictors:
    if not disagg_config.enabled:
        raise ValueError("build_disagg called with DisaggConfig.enabled=False")
    if disagg_config.prefill is None or disagg_config.decode is None:
        raise ValueError("DisaggConfig.prefill and .decode must be set")
    if disagg_config.kv_transfer is None:
        raise ValueError("DisaggConfig.kv_transfer must be set")

    if predictor_factory is None:
        # Lazy import to avoid forcing heavy deps at module import time.
        from hisim.time_predictor import AIConfiguratorTimePredictor
        predictor_factory = AIConfiguratorTimePredictor
    if hw_factory is None:
        hw_factory = _default_hw_factory

    prefill_pred = _build_role_predictor(
        model=model,
        base_sched=base_sched_config,
        role=disagg_config.prefill,
        predictor_factory=predictor_factory,
        hw_factory=hw_factory,
    )
    decode_pred = _build_role_predictor(
        model=model,
        base_sched=base_sched_config,
        role=disagg_config.decode,
        predictor_factory=predictor_factory,
        hw_factory=hw_factory,
    )

    # Real per-accelerator KV-token budget for ONE decode replica, using the
    # exact same hw/tp/pp inputs the decode predictor itself was built with.
    # This is deliberately independent of decode_pred's internals (pure calc
    # from utils.py, no I/O) so it stays correct even when predictor_factory
    # is swapped out for a fake/test double. estimate_kv_cache_pool_capacity
    # internally loads a real AIConfigurator perf model from `model`/`hw`,
    # which requires a fully-realistic ModelInfo -- callers that pass a
    # minimal/synthetic ModelInfo (e.g. unit tests exercising only the role-
    # wiring logic with a fake predictor_factory) can't satisfy that, so any
    # failure here degrades gracefully to None (== "KV-aware decode
    # admission disabled"), matching the pre-existing behaviour for those
    # callers instead of breaking them.
    decode_hw = hw_factory(disagg_config.decode.device_name)
    decode_role_sched = _build_role_sched_config(
        base_sched_config, disagg_config.decode, model
    )
    try:
        decode_kv_capacity_per_replica: Optional[int] = (
            estimate_kv_cache_pool_capacity(model, decode_hw, decode_role_sched)
        )
    except Exception as exc:
        logger.warning(
            "estimate_kv_cache_pool_capacity failed for decode role "
            "(model=%r device=%r); disabling KV-aware decode admission "
            "for this bundle (falling back to count-only capacity): %s",
            getattr(model, "name", model),
            disagg_config.decode.device_name,
            exc,
        )
        decode_kv_capacity_per_replica = None

    # Derive KV bytes-per-token from the prefill role (KV is produced there).
    prefill_role = disagg_config.prefill
    kv_dtype = (
        _as_dtype(prefill_role.kv_cache_data_type)
        if prefill_role.kv_cache_data_type is not None
        else base_sched_config.kv_cache_data_type
    )
    if kv_dtype is None:
        raise ValueError(
            "kv_cache_data_type must be set on base SchedulerConfig or prefill role"
        )
    kv_bytes = calc_kv_bytes_per_token(
        model=model,
        tp_size=prefill_role.tp_size,
        pp_size=prefill_role.pp_size,
        kv_cache_dtype=kv_dtype,
    )
    kv_model_cfg = KVModelConfig(kv_bytes_per_token=kv_bytes)
    transfer = BandwidthTransferModel(
        bw_gbps=disagg_config.kv_transfer.bw_gbps,
        latency_us=disagg_config.kv_transfer.latency_us,
    )
    return DisaggPredictors(
        prefill=prefill_pred,
        decode=decode_pred,
        kv_bytes_per_token=kv_bytes,
        kv_model_config=kv_model_cfg,
        transfer_model=transfer,
        prefill_replicas=disagg_config.prefill.replicas,
        decode_replicas=disagg_config.decode.replicas,
        decode_queue_mode=disagg_config.decode_queue_mode,
        prefill_max_running_per_replica=(
            disagg_config.prefill.max_running_per_replica
        ),
        decode_max_running_per_replica=(
            disagg_config.decode.max_running_per_replica
        ),
        decode_kv_capacity_per_replica=decode_kv_capacity_per_replica,
        max_running_per_replica=disagg_config.decode.max_running_per_replica,
    )
