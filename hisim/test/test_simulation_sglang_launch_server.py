import argparse

import pytest
import torch

from env import check_framework
from hisim.simulation.sglang import launch_server


def _make_args(**overrides):
    defaults = {
        "device": "cuda",
        "attention_backend": None,
        "decode_attention_backend": None,
        "prefill_attention_backend": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
def test_sets_flashinfer_for_unsupported_blackwell(monkeypatch):
    import sglang.srt.server_args as sgl_server_args

    raw_args = _make_args()

    monkeypatch.setattr(launch_server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sgl_server_args, "is_blackwell", lambda: True)
    monkeypatch.setattr(sgl_server_args, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(sgl_server_args, "is_flashinfer_available", lambda: True)

    launch_server._maybe_set_safe_attention_backend(raw_args)

    assert raw_args.attention_backend == "flashinfer"


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
def test_sets_triton_when_flashinfer_is_unavailable(monkeypatch):
    import sglang.srt.server_args as sgl_server_args

    raw_args = _make_args()

    monkeypatch.setattr(launch_server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sgl_server_args, "is_blackwell", lambda: True)
    monkeypatch.setattr(sgl_server_args, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(sgl_server_args, "is_flashinfer_available", lambda: False)

    launch_server._maybe_set_safe_attention_backend(raw_args)

    assert raw_args.attention_backend == "triton"


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
def test_keeps_explicit_attention_backend(monkeypatch):
    import sglang.srt.server_args as sgl_server_args

    raw_args = _make_args(attention_backend="fa3")

    monkeypatch.setattr(launch_server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sgl_server_args, "is_blackwell", lambda: True)
    monkeypatch.setattr(sgl_server_args, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(sgl_server_args, "is_flashinfer_available", lambda: False)

    launch_server._maybe_set_safe_attention_backend(raw_args)

    assert raw_args.attention_backend == "fa3"


class _FakeReqToTokenPool:
    def __init__(self, device="cpu"):
        self.req_to_token = torch.zeros((2, 8), dtype=torch.int32, device=device)

    def write(self, indices, values):
        self.req_to_token[indices] = values


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
def test_cpu_pool_uses_safe_cache_write():
    pool = _FakeReqToTokenPool(device="cpu")

    assert not launch_server._can_use_triton_write_cache_indices(
        out_cache_loc=torch.tensor([3, 4], dtype=torch.int64),
        req_pool_indices_tensor=torch.tensor([0], dtype=torch.int64),
        prefix_lens_tensor=torch.tensor([2], dtype=torch.int64),
        seq_lens_tensor=torch.tensor([4], dtype=torch.int64),
        extend_lens_tensor=torch.tensor([2], dtype=torch.int64),
        prefix_tensors=[torch.tensor([1, 2], dtype=torch.int64)],
        req_to_token_pool=pool,
    )


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
def test_safe_cache_write_fallback_populates_pool():
    pool = _FakeReqToTokenPool(device="cpu")

    launch_server._write_cache_indices_fallback(
        out_cache_loc=torch.tensor([3, 4], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
        prefix_lens_cpu=torch.tensor([2], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
        extend_lens_cpu=torch.tensor([2], dtype=torch.int64),
        prefix_tensors=[torch.tensor([1, 2], dtype=torch.int64)],
        req_to_token_pool=pool,
    )

    assert pool.req_to_token[0, :4].tolist() == [1, 2, 3, 4]
