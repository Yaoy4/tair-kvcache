import os
import json
import sys
import argparse
import torch
import hisim.hook as hisim_hook
from hisim.simulation.sglang import sgl_kernel_hook, sglang_hook
from hisim.simulation.sim_args import SimulationArgs
from hisim.utils import get_logger


# hook the sglang implementation
if not torch.cuda.is_available():
    # CPU Platform
    hisim_hook.install_module_hooks([sgl_kernel_hook.M_SGLangKernelLoadUtilHook])
hisim_hook.install_class_hooks(
    [
        sglang_hook.C_SchedulerHook,
        sglang_hook.C_ModelRunnerHook,
        sglang_hook.C_TokenizerManagerHook,
        sglang_hook.C_StorageBackendFactory,
        sglang_hook.C_HiCacheController,
        sglang_hook.C_HiRadixCacheHook,
    ]
)


logger = get_logger("hisim")


def _write_cache_indices_fallback(
    out_cache_loc: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool,
) -> None:
    target = req_to_token_pool.req_to_token
    target_device = target.device
    target_dtype = target.dtype

    pt = 0
    for i in range(req_pool_indices_cpu.shape[0]):
        req_idx = req_pool_indices_cpu[i].item()
        prefix_len = prefix_lens_cpu[i].item()
        seq_len = seq_lens_cpu[i].item()
        extend_len = extend_lens_cpu[i].item()

        prefix_values = prefix_tensors[i].to(
            device=target_device, dtype=target_dtype, non_blocking=True
        )
        req_to_token_pool.write((req_idx, slice(0, prefix_len)), prefix_values)

        extend_values = out_cache_loc[pt : pt + extend_len].to(
            device=target_device, dtype=target_dtype, non_blocking=True
        )
        req_to_token_pool.write(
            (req_idx, slice(prefix_len, seq_len)),
            extend_values,
        )
        pt += extend_len


def _can_use_triton_write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool,
) -> bool:
    target = req_to_token_pool.req_to_token
    return (
        target.is_cuda
        and out_cache_loc.is_cuda
        and req_pool_indices_tensor.is_cuda
        and prefix_lens_tensor.is_cuda
        and seq_lens_tensor.is_cuda
        and extend_lens_tensor.is_cuda
        and all(t.is_cuda for t in prefix_tensors)
    )


def _install_safe_write_cache_indices() -> None:
    from sglang.srt.mem_cache import common as mem_cache_common

    original_write_cache_indices = mem_cache_common.write_cache_indices
    if getattr(original_write_cache_indices, "_hisim_safe_write", False):
        return

    def wrapped_write_cache_indices(
        out_cache_loc: torch.Tensor,
        req_pool_indices_tensor: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        prefix_lens_tensor: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        extend_lens_tensor: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        prefix_tensors: list[torch.Tensor],
        req_to_token_pool,
    ):
        if _can_use_triton_write_cache_indices(
            out_cache_loc,
            req_pool_indices_tensor,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            prefix_tensors,
            req_to_token_pool,
        ):
            return original_write_cache_indices(
                out_cache_loc,
                req_pool_indices_tensor,
                req_pool_indices_cpu,
                prefix_lens_tensor,
                prefix_lens_cpu,
                seq_lens_tensor,
                seq_lens_cpu,
                extend_lens_tensor,
                extend_lens_cpu,
                prefix_tensors,
                req_to_token_pool,
            )

        return _write_cache_indices_fallback(
            out_cache_loc,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            seq_lens_cpu,
            extend_lens_cpu,
            prefix_tensors,
            req_to_token_pool,
        )

    wrapped_write_cache_indices._hisim_safe_write = True
    mem_cache_common.write_cache_indices = wrapped_write_cache_indices


def _has_explicit_attention_backend(raw_args: argparse.Namespace) -> bool:
    return any(
        getattr(raw_args, attr, None) is not None
        for attr in (
            "attention_backend",
            "decode_attention_backend",
            "prefill_attention_backend",
        )
    )


def _maybe_set_safe_attention_backend(raw_args: argparse.Namespace) -> None:
    if _has_explicit_attention_backend(raw_args):
        return

    device = getattr(raw_args, "device", None)
    if device is not None and device != "cuda":
        return

    if not torch.cuda.is_available():
        return

    from sglang.srt.server_args import (
        is_blackwell,
        is_flashinfer_available,
        is_sm100_supported,
    )

    if not is_blackwell() or is_sm100_supported():
        return

    raw_args.attention_backend = (
        "flashinfer" if is_flashinfer_available() else "triton"
    )
    logger.warning(
        "Default TRTLLM MHA backend requires SM100. Falling back to %s.",
        raw_args.attention_backend,
    )


# Ref: https://github.com/sgl-project/sglang/blob/v0.5.6.post2/python/sglang/launch_server.py
if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree
    from sglang.version import __version__ as sglang_version
    from hisim.simulation.sglang.version import COMPATIBLE_VERSIONS

    if sglang_version not in COMPATIBLE_VERSIONS:
        logger.warning(
            f"Current SGLang version {sglang_version} is not in the compatible versions "
            f"{COMPATIBLE_VERSIONS}, so errors may occur."
        )

    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    _install_safe_write_cache_indices()

    raw_args = parser.parse_args(sys.argv[1:])
    _maybe_set_safe_attention_backend(raw_args)
    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    config_path = os.getenv("HISIM_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from {config_path}")
    elif simulation_args.config_path:
        os.environ["HISIM_CONFIG_PATH"] = simulation_args.config_path
    else:
        config_path = "/tmp/hisim/config.json"
        logger.info(f"Export config to {config_path}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(simulation_args.to_dict(), f)
        os.environ["HISIM_CONFIG_PATH"] = config_path

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
