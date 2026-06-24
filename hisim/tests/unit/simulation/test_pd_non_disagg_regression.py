"""Phase 4b — Non-disagg regression guard.

Goal: pin down the contract that ``disagg.enabled=False`` is byte-identical
to "PD code does not exist". Every PD entry point must refuse to construct,
every config plumbing path must default off, and the metrics layer must
remain a no-op for the aggregated path.

These tests are deliberately small and structural. They will fail loudly if
any later refactor accidentally:

  * flips the disagg default to on,
  * lets a JSON/CLI path enable disagg without an explicit ``enabled=true``,
  * makes ``build_disagg`` or ``build_backend_a`` silently usable while disabled,
  * lets the PD stage-duration RequestStats fields drift away from zero defaults,
  * mutates ``calc_metrics`` so the disagg keys become missing or non-zero
    for an aggregated run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hisim.simulation.pd_config import DisaggConfig
from hisim.simulation.pd_factory import build_disagg
from hisim.simulation.pd_runtime import build_backend_a
from hisim.simulation.sim_args import SimulationArgs, _disagg_from_dict
from hisim.simulation.types import RequestStats, SchedulerConfig
from hisim.simulation.utils import calc_metrics
from hisim.spec import DataType, ModelInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> ModelInfo:
    return ModelInfo(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        num_key_value_heads=8,
        head_dim=128,
        num_full_attention=32,
        name="phase4b-stub",
    )


def _make_base_sched() -> SchedulerConfig:
    return SchedulerConfig(
        model=_make_model(),
        tp_size=1,
        pp_size=1,
        data_type=DataType.FP16,
        kv_cache_data_type=DataType.FP16,
        backend_name="sglang",
        backend_version="0.4.8",
    )


def _make_stats(rid: str, *, ttft: float = 0.05, tpot: float = 0.01,
                output_length: int = 8) -> RequestStats:
    return RequestStats(
        rid=rid,
        last_event_time=0.05 + tpot * (output_length - 1),
        input_length=256,
        output_length=output_length,
        queue_start=0.0,
        queue_end=0.0,
        created_time=0.0,
        gen_token_latencies=[ttft] + [tpot] * max(0, output_length - 1),
    )


# ---------------------------------------------------------------------------
# 1. Defaults are off.
# ---------------------------------------------------------------------------


def test_disagg_config_default_is_disabled_with_no_roles():
    cfg = DisaggConfig()
    assert cfg.enabled is False
    assert cfg.prefill is None
    assert cfg.decode is None
    assert cfg.kv_transfer is None


def test_simulation_args_default_disagg_is_disabled():
    args = SimulationArgs()
    assert args.disagg.enabled is False
    assert args.disagg.prefill is None
    assert args.disagg.decode is None
    assert args.disagg.kv_transfer is None


# ---------------------------------------------------------------------------
# 2. JSON paths default to disabled and never silently enable.
# ---------------------------------------------------------------------------


def test_from_json_without_disagg_block_is_disabled(tmp_path: Path):
    p = tmp_path / "no_disagg.json"
    p.write_text(json.dumps({
        "platform": {"accelerator": {"name": "H20"}},
        "scheduler": {"tp_size": 1, "backend_name": "sglang"},
    }))
    args = SimulationArgs.from_json(str(p))
    assert args.disagg.enabled is False
    assert args.disagg.prefill is None
    assert args.disagg.decode is None


def test_from_json_with_explicit_disabled_block_ignores_stray_keys(tmp_path: Path):
    """Even a fully-populated disagg block must stay off when enabled=false.
    Guards against a refactor where {prefill, decode, kv_transfer} presence
    is used as a synonym for 'on'."""
    p = tmp_path / "disagg_disabled.json"
    p.write_text(json.dumps({
        "scheduler": {"tp_size": 1, "backend_name": "sglang"},
        "disagg": {
            "enabled": False,
            "backend": "single_process",
            "prefill": {"device_name": "H20", "tp_size": 1, "replicas": 2},
            "decode": {"device_name": "H20", "tp_size": 1, "replicas": 2},
            "kv_transfer": {"bw_gbps": 100.0, "latency_us": 10.0},
        },
    }))
    args = SimulationArgs.from_json(str(p))
    assert args.disagg.enabled is False
    # Stray sibling keys MUST NOT bleed into the resulting object.
    assert args.disagg.prefill is None
    assert args.disagg.decode is None
    assert args.disagg.kv_transfer is None


def test_disagg_from_dict_empty_is_disabled():
    assert _disagg_from_dict({}).enabled is False


# ---------------------------------------------------------------------------
# 3. Build entry points refuse when disabled (hook must explicitly opt in).
# ---------------------------------------------------------------------------


def test_build_disagg_refuses_when_disabled():
    with pytest.raises(ValueError, match="enabled=False"):
        build_disagg(
            model=_make_model(),
            base_sched_config=_make_base_sched(),
            disagg_config=DisaggConfig(),  # enabled=False default
        )


def test_build_backend_a_refuses_when_disabled():
    with pytest.raises(ValueError, match="enabled=False"):
        build_backend_a(
            model=_make_model(),
            base_sched_config=_make_base_sched(),
            disagg_config=DisaggConfig(),
        )


# ---------------------------------------------------------------------------
# 4. RequestStats stage fields are zero-default and survive metrics.
# ---------------------------------------------------------------------------


def test_request_stats_pd_stage_fields_default_to_zero():
    s = _make_stats("baseline")
    assert s.prefill_queue_wait == 0.0
    assert s.kv_transfer_time == 0.0
    assert s.decode_queue_wait == 0.0


def test_calc_metrics_disagg_keys_present_and_zero_for_aggregated_run():
    """calc_metrics always emits the PD percentile keys (since Phase 3a).
    For a non-disagg run they must equal 0.0 — never NaN, never absent —
    so downstream dashboards and CSV exports keep a stable schema."""
    stats = [_make_stats(f"r{i}") for i in range(5)]
    m = calc_metrics(stats)
    for key in (
        "mean_prefill_queue_ms", "p50_prefill_queue_ms",
        "p95_prefill_queue_ms", "p99_prefill_queue_ms",
        "mean_kv_transfer_ms", "p50_kv_transfer_ms",
        "mean_decode_queue_ms", "p50_decode_queue_ms",
    ):
        assert key in m, f"missing metric key: {key}"
        assert m[key] == 0.0, f"{key} should be 0.0 for non-disagg run, got {m[key]}"
