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

    The HiSim runtime scheduler does NOT execute TP/EP/DP/PP; these describe
    the topology inside one predictor-backed service replica. ``replicas`` is
    the number of independently scheduled service replicas to which requests
    are routed. Runtime affinity is therefore indexed by ``replicas``, not by
    ``dp_size`` or by an SGLang ``dp_rank`` owned by another scheduler process.
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
    prefill_overhead_ms: float = 0.0
    decode_overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("tp_size", "ep_size", "dp_size", "pp_size",
                     "replicas", "max_running_per_replica"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("prefill_overhead_ms", "decode_overhead_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


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

    def decode_admission_capacity(self) -> int:
        """Native SGLang running-request cap compatible with decode routing."""
        if not self.enabled or self.decode is None:
            return DEFAULT_MAX_RUNNING
        replicas = (
            self.decode.replicas
            if self.decode_queue_mode == "per_replica_queue"
            else 1
        )
        return self.decode.max_running_per_replica * replicas

    def prefill_admission_capacity(self) -> int:
        """Native running-request cap compatible with prefill routing.

        Chunked-prefill requests retain their slot across scheduler iterations,
        so the native scheduler must never activate more requests than the
        complete prefill pool can hold.
        """
        if not self.enabled or self.prefill is None:
            return DEFAULT_MAX_RUNNING
        return (
            self.prefill.max_running_per_replica
            * self.prefill.replicas
        )

    def total_replica_count(self) -> int:
        """Total number of declared prefill + decode replicas (devices).

        Backend A ("single_process") runs every declared replica through ONE
        real (mocked) engine process, so any native, auto-estimated resource
        budget derived from a single device's capacity (e.g. the KV cache
        pool) would otherwise silently represent just one of the declared
        devices, regardless of topology. Callers scale such budgets by this
        count so the shared pool represents the aggregate of all declared
        devices instead of under-provisioning multi-replica topologies.
        Returns 1 when disaggregation is disabled (aggregated single-engine).
        """
        if not self.enabled or self.prefill is None or self.decode is None:
            return 1
        return self.prefill.replicas + self.decode.replicas

    def combined_running_request_capacity(self) -> int:
        """Shared native `max_running_requests` budget for the merged engine.

        Backend A ("single_process") funnels both P and D role traffic
        through ONE real SGLang scheduler with a single, undifferentiated
        `max_running_requests` slot pool -- native SGLang has no concept of
        "prefill-phase" vs "decode-phase" occupancy, it just tracks total
        running requests. Sizing that shared pool at only one role's
        capacity (or the smaller of the two) lets whichever role wasn't used
        to size it starve the other: decode requests hold their slot for the
        whole generation (duration scales with output length), so once they
        fill the pool, new arrivals can't even be admitted for prefill --
        regardless of real KV-cache token headroom. Summing both roles'
        declared capacities lets the merged pool hold up to
        ``prefill_admission_capacity()`` concurrent prefills AND up to
        ``decode_admission_capacity()`` concurrent decodes at the same time,
        matching how independent P/D hardware would behave.
        Returns the shared ``DEFAULT_MAX_RUNNING`` fallback when disagg is
        disabled (aggregated single-engine, no P/D split to sum).
        """
        if not self.enabled or self.prefill is None or self.decode is None:
            return DEFAULT_MAX_RUNNING
        return self.prefill_admission_capacity() + self.decode_admission_capacity()
