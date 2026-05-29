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
from hisim.simulation.utils import calc_kv_cache_cell_elems
from hisim.simulation.pd_config import DisaggConfig, RolePredictorConfig
from hisim.simulation.pd_transfer import BandwidthTransferModel, KVModelConfig


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


def _default_hw_factory(name: str):
    hw = AcceleratorInfo.find_by_hw_name(name)
    if hw is None:
        raise ValueError(
            f"Unknown accelerator name {name!r}. Available: "
            f"{sorted(AcceleratorInfo.list_all_hws().keys())}"
        )
    return hw


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
    )
