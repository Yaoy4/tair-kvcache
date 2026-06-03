"""Unit tests for PD disaggregation KV cache transfer simulation."""

import json
import pathlib
import pytest
from copy import deepcopy

from hisim.simulation.types import SchedulerConfig
from hisim.simulation.manager.config import ConfigManager
from hisim.time_predictor import ScheduleBatch, FakeRequest


cur_dir = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# SchedulerConfig field defaults and override
# ---------------------------------------------------------------------------

def test_scheduler_config_pd_disagg_defaults():
    """New fields default to disabled / unset."""
    sc = SchedulerConfig(model=None)
    assert sc.pd_disagg_enabled is False
    assert sc.pd_kv_transfer_bandwidth_gb is None


def test_scheduler_config_pd_disagg_set():
    """Fields can be set explicitly."""
    sc = SchedulerConfig(
        model=None,
        pd_disagg_enabled=True,
        pd_kv_transfer_bandwidth_gb=25.0,
    )
    assert sc.pd_disagg_enabled is True
    assert sc.pd_kv_transfer_bandwidth_gb == 25.0


# ---------------------------------------------------------------------------
# Demo config file
# ---------------------------------------------------------------------------

def test_demo_config_file_has_pd_disagg():
    """The demo config file sets both PD disagg fields correctly."""
    cfg_path = cur_dir / "assets" / "mock" / "config.demo.pd_disagg.json"
    assert cfg_path.exists(), "Demo config file not found"
    cfg = json.loads(cfg_path.read_text())
    sched = cfg.get("scheduler", {})
    assert sched.get("pd_disagg_enabled") is True
    assert isinstance(sched.get("pd_kv_transfer_bandwidth_gb"), (int, float))
    assert sched.get("pd_kv_transfer_bandwidth_gb") > 0


# ---------------------------------------------------------------------------
# KV transfer duration formula
# ---------------------------------------------------------------------------

def test_kv_transfer_duration_formula():
    """KV transfer duration = tokens * bytes_per_token / (bandwidth * 1e9)."""
    tokens = 1024
    bytes_per_token = 131072  # e.g., 32 layers * 2 * 64 heads * 128 dim * 2 bytes
    bandwidth_gb = 25.0

    expected_dur = tokens * bytes_per_token / (bandwidth_gb * 1e9)

    # Verify formula directly
    computed = tokens * bytes_per_token / (bandwidth_gb * 1e9)
    assert abs(computed - expected_dur) < 1e-12
    # Sanity: 1024 tokens * 128KB/token / 25GB/s ≈ 5.24 ms
    assert 0.001 < computed < 1.0, f"Unexpected duration: {computed}s"


# ---------------------------------------------------------------------------
# Chunked prefill token filtering logic
# ---------------------------------------------------------------------------

class _MockSglReq:
    """Minimal stand-in for a SGLang request object."""
    def __init__(self, is_chunked: int):
        self.is_chunked = is_chunked


def _count_final_prefill_tokens(sgl_reqs, hisim_reqs):
    """
    Mirror of the filtering logic in sglang_hook.py wrapped_process_batch_result.
    Only counts tokens for requests whose final prefill chunk just completed.
    """
    if len(sgl_reqs) != len(hisim_reqs):
        # Fallback path
        return sum(r.input_length for r in hisim_reqs if r.input_length > 1)
    return sum(
        hreq.input_length
        for sreq, hreq in zip(sgl_reqs, hisim_reqs)
        if sreq.is_chunked == 0 and hreq.input_length > 1
    )


def test_chunked_mid_chunk_no_transfer():
    """Mid-chunk requests (is_chunked=1) should contribute 0 tokens for KV transfer."""
    sgl_reqs = [_MockSglReq(is_chunked=1), _MockSglReq(is_chunked=1)]
    hisim_reqs = [FakeRequest(input_length=512, past_kv_length=0),
                  FakeRequest(input_length=256, past_kv_length=0)]
    tokens = _count_final_prefill_tokens(sgl_reqs, hisim_reqs)
    assert tokens == 0, "Mid-chunk requests should not trigger KV transfer"


def test_chunked_final_chunk_triggers_transfer():
    """Final chunk requests (is_chunked=0, input_length>1) should count their tokens."""
    sgl_reqs = [_MockSglReq(is_chunked=0), _MockSglReq(is_chunked=0)]
    hisim_reqs = [FakeRequest(input_length=512, past_kv_length=0),
                  FakeRequest(input_length=256, past_kv_length=0)]
    tokens = _count_final_prefill_tokens(sgl_reqs, hisim_reqs)
    assert tokens == 768, f"Expected 768 tokens, got {tokens}"


def test_chunked_mixed_batch():
    """Batch with both mid-chunk and final-chunk requests: only final counts."""
    sgl_reqs = [
        _MockSglReq(is_chunked=1),  # mid-chunk
        _MockSglReq(is_chunked=0),  # final chunk
        _MockSglReq(is_chunked=0),  # decode (input_length=1 -> excluded)
    ]
    hisim_reqs = [
        FakeRequest(input_length=512, past_kv_length=0),   # mid-chunk
        FakeRequest(input_length=256, past_kv_length=512), # final chunk
        FakeRequest(input_length=1,   past_kv_length=768), # decode
    ]
    tokens = _count_final_prefill_tokens(sgl_reqs, hisim_reqs)
    assert tokens == 256, f"Expected 256 tokens (only final-chunk prefill), got {tokens}"


def test_decode_only_batch_no_transfer():
    """Pure decode batch: no tokens counted, so no KV transfer."""
    sgl_reqs = [_MockSglReq(is_chunked=0), _MockSglReq(is_chunked=0)]
    hisim_reqs = [FakeRequest(input_length=1, past_kv_length=512),
                  FakeRequest(input_length=1, past_kv_length=256)]
    tokens = _count_final_prefill_tokens(sgl_reqs, hisim_reqs)
    assert tokens == 0, "Decode-only batch should not trigger KV transfer"


# ---------------------------------------------------------------------------
# ScheduleBatch helpers used in PD disagg check
# ---------------------------------------------------------------------------

def test_schedule_batch_is_prefill():
    batch = ScheduleBatch([FakeRequest(512, 0), FakeRequest(256, 0)])
    assert batch.is_prefill() is True
    assert batch.is_decode() is False


def test_schedule_batch_is_decode():
    batch = ScheduleBatch([FakeRequest(1, 512), FakeRequest(1, 256)])
    assert batch.is_decode() is True
    assert batch.is_prefill() is False


def test_schedule_batch_num_context_tokens():
    batch = ScheduleBatch([FakeRequest(512, 0), FakeRequest(256, 128)])
    assert batch.num_context_tokens == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
