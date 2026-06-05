"""Backend-agnostic config types for PD disaggregation.

These mirror the inputs HiSim already understands (predictor knobs + KV transfer)
and intentionally avoid any SGLang or runtime-execution concepts. Per-role
fields enable heterogeneous deployments (e.g., H100 prefill + H20 decode) by
giving each role its own predictor inputs.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

DEFAULT_MAX_RUNNING = (1 << 31) - 1


@dataclass
class RolePredictorConfig:
    """Predictor inputs for one PD role (prefill or decode).

    The HiSim runtime scheduler does NOT execute TP/EP/DP/PP — these are
    inputs to AIConfigurator's perf-database lookup. One PD role corresponds
    to one AIConfiguratorTimePredictor instance.
    """

    device_name: str
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    data_type: Optional[str] = None
    kv_cache_data_type: Optional[str] = None
    database_path: Optional[str] = None
    backend_version: Optional[str] = None
    replicas: int = 1
    max_running_per_replica: int = DEFAULT_MAX_RUNNING
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in ("tp_size", "ep_size", "dp_size", "pp_size",
                     "replicas", "max_running_per_replica"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")


@dataclass
class BandwidthTransferConfig:
    """KV transfer config consumed by BandwidthTransferModel in Backend A."""

    bw_gbps: float
    latency_us: float

    def __post_init__(self) -> None:
        if self.bw_gbps <= 0:
            raise ValueError(f"bw_gbps must be > 0, got {self.bw_gbps}")
        if self.latency_us < 0:
            raise ValueError(f"latency_us must be >= 0, got {self.latency_us}")


DisaggBackend = Literal["single_process", "two_process"]
DecodeQueueMode = Literal["single_replica", "per_replica_queue"]
_ALLOWED_BACKENDS = ("single_process", "two_process")
_ALLOWED_DECODE_QUEUE_MODES = ("single_replica", "per_replica_queue")


@dataclass
class DisaggConfig:
    """Top-level disaggregation config attached to SchedulerConfig.

    When ``enabled`` is False, all role/transfer fields may be None and the
    rest of HiSim behaves as today (aggregated single-engine).
    """

    enabled: bool = False
    backend: DisaggBackend = "single_process"
    decode_queue_mode: DecodeQueueMode = "single_replica"
    prefill: Optional[RolePredictorConfig] = None
    decode: Optional[RolePredictorConfig] = None
    kv_transfer: Optional[BandwidthTransferConfig] = None

    def __post_init__(self) -> None:
        if self.backend not in _ALLOWED_BACKENDS:
            raise ValueError(
                f"backend must be one of {_ALLOWED_BACKENDS}, got {self.backend!r}"
            )
        if self.decode_queue_mode not in _ALLOWED_DECODE_QUEUE_MODES:
            raise ValueError(
                "decode_queue_mode must be one of "
                f"{_ALLOWED_DECODE_QUEUE_MODES}, got {self.decode_queue_mode!r}"
            )
        if self.enabled:
            if self.prefill is None:
                raise ValueError("disagg enabled but prefill role config missing")
            if self.decode is None:
                raise ValueError("disagg enabled but decode role config missing")
            if self.kv_transfer is None:
                raise ValueError("disagg enabled but kv_transfer config missing")
