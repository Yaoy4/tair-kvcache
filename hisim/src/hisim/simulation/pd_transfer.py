from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class KVModelConfig:
    """Backend-agnostic KV transfer inputs derived from model + scheduler config."""

    kv_bytes_per_token: int


class TransferModel(ABC):
    """Abstract KV-cache transfer time model used by PD disaggregation."""

    @abstractmethod
    def estimate(self, seq_len: int, model_cfg: KVModelConfig) -> float:
        """Return KV transfer duration in seconds for `seq_len` tokens."""


class BandwidthTransferModel(TransferModel):
    """Simple latency + bandwidth model: t = latency + bytes / bw."""

    def __init__(self, bw_gbps: float, latency_us: float):
        if bw_gbps <= 0:
            raise ValueError(f"bw_gbps must be > 0, got {bw_gbps}")
        if latency_us < 0:
            raise ValueError(f"latency_us must be >= 0, got {latency_us}")
        self.bw_gbps = bw_gbps
        self.latency_us = latency_us

    def estimate(self, seq_len: int, model_cfg: KVModelConfig) -> float:
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")
        bytes_total = seq_len * model_cfg.kv_bytes_per_token
        bw_bytes_per_s = self.bw_gbps * 1e9
        return self.latency_us * 1e-6 + bytes_total / bw_bytes_per_s
