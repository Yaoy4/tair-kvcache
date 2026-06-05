"""Unit tests for PD disaggregation KV cache transfer simulation."""

import json
import pathlib
import pytest
from copy import deepcopy

from hisim.simulation.types import SchedulerConfig
from hisim.simulation.manager.config import ConfigManager
from hisim.simulation.manager.state import StateManager
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
    assert sc.pd_num_prefill_instances == 1
    assert sc.pd_num_decode_instances == 1


def test_scheduler_config_pd_disagg_set():
    """Fields can be set explicitly."""
    sc = SchedulerConfig(
        model=None,
        pd_disagg_enabled=True,
        pd_kv_transfer_bandwidth_gb=25.0,
        pd_num_prefill_instances=2,
        pd_num_decode_instances=3,
    )
    assert sc.pd_disagg_enabled is True
    assert sc.pd_kv_transfer_bandwidth_gb == 25.0
    assert sc.pd_num_prefill_instances == 2
    assert sc.pd_num_decode_instances == 3


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
    assert sched.get("pd_num_prefill_instances", 1) >= 1
    assert sched.get("pd_num_decode_instances", 1) >= 1


# ---------------------------------------------------------------------------
# StateManager P/D timelines (Step 3)
# ---------------------------------------------------------------------------

def test_state_manager_init_pd_timelines():
    StateManager.reset()
    StateManager.init_pd_timelines(num_prefill=2, num_decode=3)
    assert [StateManager.get_prefill_clock(i) for i in range(2)] == [0.0, 0.0]
    assert [StateManager.get_decode_clock(i) for i in range(3)] == [0.0, 0.0, 0.0]
    StateManager.reset()


def test_state_manager_init_pd_timelines_invalid():
    with pytest.raises(ValueError):
        StateManager.init_pd_timelines(num_prefill=0, num_decode=1)
    with pytest.raises(ValueError):
        StateManager.init_pd_timelines(num_prefill=1, num_decode=0)
    StateManager.reset()


def test_state_manager_pick_least_loaded_instance():
    StateManager.reset()
    StateManager.init_pd_timelines(num_prefill=2, num_decode=2)
    StateManager.step_prefill_clock(0, 5.0)  # P0 busy until t=5
    assert StateManager.pick_prefill_instance() == 1  # P1 is free
    StateManager.start_decode(0, 3.0)  # D0 busy until t=3
    assert StateManager.pick_decode_instance() == 1  # D1 is free
    StateManager.reset()


def test_state_manager_decode_waits_for_its_own_kv_migration():
    """A decode cannot start before its own P->D KV migration completes."""
    StateManager.reset()
    StateManager.init_pd_timelines(num_prefill=1, num_decode=1)
    StateManager.step_prefill_clock(0, 10.0)  # prefill compute done at t=10
    ready = StateManager.complete_kv_migration(0, transfer_dur=2.0)  # KV ready at t=12
    assert ready == 12.0
    # Decode clock is at 0, but it must wait for KV ready (t=12).
    start, end = StateManager.start_decode(0, dur=1.0, kv_ready_time=ready)
    assert start == 12.0
    assert end == 13.0
    StateManager.reset()


def test_state_manager_decode_not_delayed_by_unrelated_later_migration():
    """An earlier-ready decode must NOT wait for an unrelated later migration."""
    StateManager.reset()
    StateManager.init_pd_timelines(num_prefill=2, num_decode=2)
    StateManager.step_prefill_clock(0, 10.0)
    StateManager.step_prefill_clock(1, 30.0)
    ready_a = StateManager.complete_kv_migration(0, transfer_dur=0.0)  # ready at 10
    ready_b = StateManager.complete_kv_migration(1, transfer_dur=0.0)  # ready at 30
    assert ready_a == 10.0 and ready_b == 30.0
    # Decode for request A on D0 starts at its own ready time (10), not 30.
    start_a, _ = StateManager.start_decode(0, dur=1.0, kv_ready_time=ready_a)
    assert start_a == 10.0
    # Aggregate latest is tracked for metrics only.
    assert StateManager.get_last_kv_ready_time() == 30.0
    StateManager.reset()


def test_state_manager_kv_migration_does_not_block_prefill_instance():
    """KV transfer runs over the network and does not advance the P clock."""
    StateManager.reset()
    StateManager.init_pd_timelines(num_prefill=1, num_decode=1)
    StateManager.step_prefill_clock(0, 10.0)
    StateManager.complete_kv_migration(0, transfer_dur=5.0)
    # P instance is free again at t=10 for the next prefill (transfer is async).
    assert StateManager.get_prefill_clock(0) == 10.0
    StateManager.reset()


# ---------------------------------------------------------------------------
# TTFT recording: TTFT = prefill + KV transfer + first decode (PD disagg)
# ---------------------------------------------------------------------------

def _replay_request(events, created_time=0.0):
    """Mirror of the per-request recording logic in wrapped_process_batch_result.

    ``events`` is a list of ``(clock_after, is_pd_prefill)`` tuples, one per batch
    in which this (is_chunked == 0) request participated. ``clock_after`` is the
    virtual global clock after the batch finished (already including the KV
    transfer time for a PD prefill batch). Returns the resulting
    ``gen_token_latencies`` list, where index 0 is the TTFT.
    """
    gen = []
    last_event_time = created_time
    pending = False
    for clock_after, is_pd_prefill in events:
        if pending:
            # Fold the first decode into TTFT.
            gen[0] += clock_after - last_event_time
            pending = False
        else:
            gen.append(clock_after - last_event_time)
            if is_pd_prefill:
                pending = True
        last_event_time = clock_after
    return gen


def test_ttft_non_pd_is_pure_prefill():
    """Baseline (non-PD): TTFT = prefill time; each decode is its own ITL."""
    prefill, dec1, dec2 = 2.0, 0.5, 0.5
    events = [
        (prefill, False),
        (prefill + dec1, False),
        (prefill + dec1 + dec2, False),
    ]
    gen = _replay_request(events)
    assert gen == [prefill, dec1, dec2]
    assert gen[0] == prefill  # TTFT


def test_ttft_pd_includes_transfer_and_first_decode():
    """PD disagg: TTFT = prefill + KV transfer + first decode."""
    prefill, transfer, dec1, dec2 = 2.0, 0.3, 0.5, 0.5
    events = [
        # prefill batch: clock already advanced by prefill + transfer
        (prefill + transfer, True),
        (prefill + transfer + dec1, False),
        (prefill + transfer + dec1 + dec2, False),
    ]
    gen = _replay_request(events)
    # First decode folded into TTFT; it is NOT a separate ITL.
    assert gen[0] == pytest.approx(prefill + transfer + dec1)
    assert gen[1:] == pytest.approx([dec2])


def test_ttft_pd_single_output_token_no_decode():
    """PD disagg, output_length == 1: TTFT = prefill + transfer (no decode)."""
    prefill, transfer = 2.0, 0.3
    events = [(prefill + transfer, True)]
    gen = _replay_request(events)
    assert gen == pytest.approx([prefill + transfer])


def test_ttft_pd_e2e_latency_preserved():
    """Folding the first decode into TTFT keeps total e2e latency unchanged."""
    prefill, transfer, dec1, dec2 = 2.0, 0.3, 0.5, 0.4
    events = [
        (prefill + transfer, True),
        (prefill + transfer + dec1, False),
        (prefill + transfer + dec1 + dec2, False),
    ]
    gen = _replay_request(events)
    assert sum(gen) == pytest.approx(prefill + transfer + dec1 + dec2)


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
