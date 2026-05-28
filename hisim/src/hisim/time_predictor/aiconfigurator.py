from hisim.spec.data_type import DataType
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo
import inspect
import math
from functools import wraps
import numpy as np
import os
import re
import sys
from typing import Optional

from aiconfigurator.sdk import models
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
    DatabaseMode,
)
try:
    from aiconfigurator.sdk.common import SupportedModels
except ImportError:
    SupportedModels = None
from aiconfigurator.sdk.config import RuntimeConfig, ModelConfig
from aiconfigurator.sdk.inference_session import InferenceSession
import aiconfigurator.sdk.perf_database as aic_perf_database
try:
    from aiconfigurator.sdk.perf_database import get_database, get_system_config_path
except ImportError:
    from aiconfigurator.sdk.perf_database import get_database, get_systems_paths

    def get_system_config_path():
        systems_paths = get_systems_paths()
        return systems_paths[0] if systems_paths else None

from hisim.simulation.types import (
    SchedulerConfig,
    PlatformConfig,
)
from hisim.time_predictor import (
    InferTimePredictor,
    ScheduleBatch,
    FakeRequest,
)
from hisim.utils import get_logger


logger = get_logger("hisim")


def _first_available_enum_member(enum_cls, *member_names):
    for member_name in member_names:
        member = getattr(enum_cls, member_name, None)
        if member is None:
            member = enum_cls.__members__.get(member_name)
        if member is not None:
            return member
    raise AttributeError(
        f"{enum_cls.__name__} does not provide any of: {', '.join(member_names)}"
    )


def _install_enum_alias(enum_cls, alias_name: str, canonical_name: str) -> bool:
    members = enum_cls.__members__
    if alias_name in members:
        return False

    canonical_member = members.get(canonical_name)
    if canonical_member is None:
        return False

    enum_cls._member_map_[alias_name] = canonical_member
    return True


def _install_aic_legacy_dtype_aliases() -> None:
    # Older AIConfigurator SDK builds expose float16 but not bfloat16, while newer perf
    # database files may use "bfloat16" as the dtype token. Alias them before database load.
    for enum_cls in (
        GEMMQuantMode,
        KVCacheQuantMode,
        FMHAQuantMode,
        MoEQuantMode,
    ):
        _install_enum_alias(enum_cls, "bfloat16", "float16")


def _backfill_aic_system_spec(system_spec: dict) -> None:
    gpu_spec = system_spec.get("gpu")
    if not isinstance(gpu_spec, dict):
        return

    if "float16_tc_flops" not in gpu_spec and "bfloat16_tc_flops" in gpu_spec:
        gpu_spec["float16_tc_flops"] = gpu_spec["bfloat16_tc_flops"]


def _install_aic_system_spec_compatibility() -> None:
    original_correct_data = aic_perf_database.PerfDatabase._correct_data
    if getattr(original_correct_data, "_hisim_system_spec_compat", False):
        return

    @wraps(original_correct_data)
    def wrapped_correct_data(self, *args, **kwargs):
        _backfill_aic_system_spec(self.system_spec)
        return original_correct_data(self, *args, **kwargs)

    wrapped_correct_data._hisim_system_spec_compat = True
    aic_perf_database.PerfDatabase._correct_data = wrapped_correct_data


_GET_DATABASE_SUPPORTS_SYSTEMS_PATHS = (
    "systems_paths" in inspect.signature(get_database).parameters
)
_GEMM_FLOAT16_MODE = _first_available_enum_member(
    GEMMQuantMode, "float16", "bfloat16"
)
_GEMM_FP8_MODE = _first_available_enum_member(GEMMQuantMode, "fp8", "fp8_block")
_KV_FLOAT16_MODE = _first_available_enum_member(
    KVCacheQuantMode, "float16", "bfloat16"
)
_FMHA_FLOAT16_MODE = _first_available_enum_member(
    FMHAQuantMode, "float16", "bfloat16"
)
_MOE_FLOAT16_MODE = _first_available_enum_member(MoEQuantMode, "float16", "bfloat16")


# Map the common data types to AIConfigurator data types.
MAP_DTYPE_TO_GEMMQuantMode = {
    DataType.FP16: _GEMM_FLOAT16_MODE,
    DataType.BF16: _GEMM_FLOAT16_MODE,
    DataType.FP8: _GEMM_FP8_MODE,
    DataType.INT8: GEMMQuantMode.int8_wo,
    DataType.FP4: GEMMQuantMode.nvfp4,
    DataType.INT4: GEMMQuantMode.int4_wo,
    DataType.FP16_TENSOR: _GEMM_FLOAT16_MODE,
    DataType.BF16_TENSOR: _GEMM_FLOAT16_MODE,
    DataType.FP8_TENSOR: _GEMM_FP8_MODE,
    DataType.INT8_TENSOR: GEMMQuantMode.int8_wo,
    DataType.FP4_TENSOR: GEMMQuantMode.nvfp4,
    DataType.INT4_TENSOR: GEMMQuantMode.int4_wo,
}

MAP_DTYPE_TO_KVCacheQuantMode = {
    DataType.FP16: _KV_FLOAT16_MODE,
    DataType.BF16: _KV_FLOAT16_MODE,
    DataType.FP8: KVCacheQuantMode.fp8,
    DataType.INT8: KVCacheQuantMode.int8,
}

MAP_DTYPE_TO_FMHAQuantMode = {
    DataType.FP16: _FMHA_FLOAT16_MODE,
    DataType.BF16: _FMHA_FLOAT16_MODE,
    DataType.FP8: FMHAQuantMode.fp8,
}

MAP_DTYPE_TO_MoEQuantMode = {
    DataType.FP16: _MOE_FLOAT16_MODE,
    DataType.BF16: _MOE_FLOAT16_MODE,
    DataType.FP8: MoEQuantMode.fp8,
    DataType.INT8: MoEQuantMode.fp8,
    DataType.FP4: MoEQuantMode.nvfp4,
    DataType.INT4: MoEQuantMode.int4_wo,
}

MAP_DTYPE_TO_CommQunatMode = {
    DataType.FP16: CommQuantMode.half,
    DataType.BF16: CommQuantMode.half,
    DataType.FP8: CommQuantMode.fp8,
    DataType.INT8: CommQuantMode.int8,
}


def _load_perf_database(
    system: str,
    backend: str,
    version: str,
    systems_path: str,
    database_mode: DatabaseMode,
):
    _install_aic_legacy_dtype_aliases()
    _install_aic_system_spec_compatibility()

    if _GET_DATABASE_SUPPORTS_SYSTEMS_PATHS:
        return get_database(
            system=system,
            backend=backend,
            version=version,
            systems_paths=systems_path,
            database_mode=database_mode.name,
        )
    return get_database(
        system=system,
        backend=backend,
        version=version,
        systems_dir=systems_path,
    )

INTERCONNECT_MODE_DEFAULT_LATENCY_US = {
    "none": 0.0,
    "nvlink": 1.0,
    "pcie": 3.0,
    "ib": 5.0,
    "infiniband": 5.0,
    "roce": 7.0,
    "ethernet": 10.0,
}

INTERCONNECT_MODE_BANDWIDTH_EFFICIENCY = {
    "none": 1.0,
    "nvlink": 1.0,
    "pcie": 0.9,
    "ib": 0.9,
    "infiniband": 0.9,
    "roce": 0.85,
    "ethernet": 0.8,
}


# XGBoost
def _import_xgboost():
    """
    Try importing a usable XGBoost Python package.

    Note: This repository vendors the XGBoost source tree at `<repo_root>/xgboost/`, which can shadow
    the real Python package depending on cwd/sys.path. We try:
    - Normal import
    - Repository-local Python package import (adds `<repo_root>/xgboost/python-package` to sys.path)
    """

    def _is_usable(mod) -> bool:
        return hasattr(mod, "XGBRegressor") and hasattr(mod, "DMatrix")

    try:
        import xgboost as _xgb  # type: ignore

        if _is_usable(_xgb):
            return _xgb
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_py_pkg = os.path.join(repo_root, "xgboost", "python-package")
    if os.path.isdir(local_py_pkg) and local_py_pkg not in sys.path:
        sys.path.insert(0, local_py_pkg)

    import xgboost as _xgb2  # type: ignore

    if not _is_usable(_xgb2):
        raise RuntimeError(
            "Imported 'xgboost' but it doesn't look like the python package (missing XGBRegressor/DMatrix). "
            "If you intended to use pip/conda xgboost, install it; if you intended to use the vendored source tree, "
            "make sure it is built properly."
        )
    return _xgb2


def _parse_bucket_from_model_path(path: str) -> tuple[int, int]:
    """
    Parse bucket range from model filename.
    Expected: '*bs<lo>_<hi>*' or '*bs<lo>-<hi>*' or '*bs<lo>to<hi>*'
    """
    base = os.path.basename(path)
    m = re.search(r"bs\s*(\d+)\s*[_\-]\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"bs\s*(\d+)\s*to\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse bucket range from model filename: {base}")
    lo = int(m.group(1))
    hi = int(m.group(2))
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError(f"Invalid bucket range in model filename: {base}")
    return lo, hi


def _load_bucket_models(model_dir: str):
    """
    Load all bucket models from a directory.
    Returns list of (lo, hi, reg) sorted by bucket width then lo.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return []
    xgb = _import_xgboost()
    models = []
    for fn in os.listdir(model_dir):
        if not (fn.endswith(".json") or fn.endswith(".model")):
            continue
        p = os.path.join(model_dir, fn)
        try:
            lo, hi = _parse_bucket_from_model_path(p)
        except Exception:
            continue
        reg = xgb.XGBRegressor()
        reg.load_model(p)
        models.append((lo, hi, reg, p))
        logger.info(f"Loaded bucket XGBoost model bs{lo}_{hi}: {p}")
    models.sort(key=lambda t: ((t[1] - t[0]), t[0], t[1]))
    return models


def _pick_bucket_model(models, bs: int):
    for lo, hi, reg, p in models:
        if lo <= bs <= hi:
            return lo, hi, reg, p
    return None


def _build_xgb_feature_maxbs_2(reqs: list[FakeRequest], *, max_bs: int) -> np.ndarray:
    """
    Feature shape (max_bs, 2):
    - rows [0..max_bs-1]: if request exists -> [1, past_kv_length] else [0, 0]
    - NOTE: AIC latency is NOT included in features for ratio model.
    """
    if max_bs <= 0:
        raise ValueError("max_bs must be > 0")
    feats = np.zeros((max_bs, 2), dtype=np.float32)
    n = min(len(reqs), max_bs)
    for i in range(n):
        feats[i, 0] = 1.0
        feats[i, 1] = float(reqs[i].past_kv_length)
    return feats


def _xgb_predict_ratio(reqs: list[FakeRequest], xgb_reg, *, max_bs: int) -> float:
    if len(reqs) > max_bs:
        raise ValueError(
            f"batch_size={len(reqs)} > max_bs={max_bs} is not supported by the (max_bs,2) feature spec."
        )
    x = (
        _build_xgb_feature_maxbs_2(reqs, max_bs=max_bs)
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    pred_ratio = float(xgb_reg.predict(x.reshape(1, -1))[0])
    pred_ratio = max(pred_ratio, 1e-6)
    return pred_ratio


def _pick_generation_attn_key(latency_dict: dict) -> str:
    if not latency_dict:
        raise ValueError("Empty latency_dict")

    exact_candidates = [
        "generation_attn",
        "gen_attn",
        "gen_attention",
        "generation_attention",
        "attn",
    ]
    for k in exact_candidates:
        if k in latency_dict:
            return k

    keys = list(latency_dict.keys())
    gen_attn = [
        k for k in keys if ("attn" in str(k).lower() and "gen" in str(k).lower())
    ]
    if gen_attn:
        return max(gen_attn, key=lambda kk: float(latency_dict.get(kk, 0.0)))

    attn_keys = [k for k in keys if "attn" in str(k).lower()]
    if attn_keys:
        return max(attn_keys, key=lambda kk: float(latency_dict.get(kk, 0.0)))

    raise KeyError(f"Cannot find generation attn key from latency_dict keys={keys}")


def _split_gen_attn_other(latency_dict: dict) -> tuple[str, float, float, float]:
    gen_k = _pick_generation_attn_key(latency_dict)
    gen_ms = float(latency_dict[gen_k])
    total_ms = float(sum(float(v) for v in latency_dict.values()))
    other_ms = float(total_ms - gen_ms)
    return gen_k, gen_ms, other_ms, total_ms


def _parse_decode_bs_range_from_xgb_model_path(
    xgb_decode_model_path: str,
) -> tuple[int, int]:
    """
    Parse decode batch size range from xgboost model filename.

    Expected patterns (examples):
    - xgb_decode_bs6_32.json  -> (6, 32)
    - .../xgb_decode_bs6-32.model -> (6, 32)
    - .../xgb_decode_bs6to32.json -> (6, 32)

    If not found, returns (-1, -1).
    """
    base = os.path.basename(xgb_decode_model_path)
    m = re.search(r"bs\s*(\d+)\s*[_\-]\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"bs\s*(\d+)\s*to\s*(\d+)", base, flags=re.IGNORECASE)
    if not m:
        return -1, -1
    lo = int(m.group(1))
    hi = int(m.group(2))
    if lo <= 0 or hi <= 0 or lo > hi:
        return -1, -1
    return lo, hi


def get_perf_model(sched_config: SchedulerConfig, model: ModelInfo) -> models.BaseModel:
    model_config = ModelConfig(
        pp_size=sched_config.pp_size,
        tp_size=sched_config.tp_size,
        moe_tp_size=sched_config.tp_size,  # FIXME
        moe_ep_size=sched_config.ep_size,
        attention_dp_size=sched_config.dp_size,  # FIXME
        gemm_quant_mode=MAP_DTYPE_TO_GEMMQuantMode.get(
            sched_config.data_type, _GEMM_FLOAT16_MODE
        ),
        moe_quant_mode=MAP_DTYPE_TO_MoEQuantMode.get(
            sched_config.data_type, _MOE_FLOAT16_MODE
        ),
        kvcache_quant_mode=MAP_DTYPE_TO_KVCacheQuantMode.get(
            sched_config.kv_cache_data_type, _KV_FLOAT16_MODE
        ),
        fmha_quant_mode=MAP_DTYPE_TO_FMHAQuantMode.get(
            sched_config.kv_cache_data_type, _FMHA_FLOAT16_MODE
        ),
        comm_quant_mode=MAP_DTYPE_TO_CommQunatMode.get(
            sched_config.data_type, CommQuantMode.half
        ),
        workload_distribution="power_law_1.2",
    )

    if SupportedModels is None:
        model_source = model.model_path or model.name
        if not model_source:
            raise ValueError("Model path is required for the installed AIConfigurator version.")
        return models.get_model(
            model_source,
            model_config,
            sched_config.backend_name,
        )

    if model.model_type in [
        "qwen",
        "qwen2",
        "qwen3",
        "llama",
        "chatglm",
    ]:
        # aiconfigurator.sdk.backends.trtllm_backend._get_memory_usage() requires the SupportedModels.
        SupportedModels.update(
            {
                model.name: [
                    "LLAMA",
                    model.num_hidden_layers,
                    model.num_attention_heads,
                    model.num_key_value_heads,
                    model.head_dim,
                    model.hidden_size,
                    model.intermediate_size,
                    model.vocab_size,
                    model.max_seq_len,
                    0,
                    0,
                    0,
                    None,
                ]
            }
        )
    elif model.model_type in ["deepseek_v3", "kimi_k2"]:
        SupportedModels.update(
            {
                model.name: [
                    "DEEPSEEK",
                    model.num_hidden_layers,
                    model.num_attention_heads,
                    model.num_key_value_heads,
                    model.head_dim,
                    model.hidden_size,
                    model.intermediate_size,
                    model.vocab_size,
                    model.max_seq_len,
                    model.num_experts_per_tok,
                    model.n_routed_experts,
                    model.moe_intermediate_size,
                    None,
                ]
            }
        )
    elif model.model_type in ["qwen3_moe"]:
        SupportedModels.update(
            {
                model.name: [
                    "MOE",
                    model.num_hidden_layers,
                    model.num_attention_heads,
                    model.num_key_value_heads,
                    model.head_dim,
                    model.hidden_size,
                    model.intermediate_size,
                    model.vocab_size,
                    model.max_seq_len,
                    model.num_experts_per_tok,
                    model.n_routed_experts,
                    model.moe_intermediate_size,
                    None,
                ]
            }
        )
    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")
    return models.get_model(
        model_name=model.name,
        model_config=model_config,
        backend_name=sched_config.backend_name,
    )


class AIConfiguratorTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        platform_config: Optional[PlatformConfig] = None,
        database_path: Optional[str] = None,
        database_mode: DatabaseMode | str = DatabaseMode.SILICON,
        xgb_model_path: Optional[str] = None,
        prefill_scale_factor: float = 1,
        decode_scale_factor: float = 1,
    ):
        super().__init__(model, hw, config)

        self.platform_config = platform_config
        self.prefill_scale_factor = prefill_scale_factor
        self.decode_scale_factor = decode_scale_factor

        if isinstance(database_mode, str):
            database_mode = self._get_database_mode(database_mode)

        systems_path = (
            database_path if database_path is not None else get_system_config_path()
        )
        database = _load_perf_database(
            system=hw.name,
            backend=config.backend_name,
            version=config.backend_version,
            systems_path=systems_path,
            database_mode=database_mode,
        )

        if database is None:
            raise ValueError("Failed to initialize the database.")

        database.set_default_database_mode(database_mode)

        # --- Replace the original function to support more flexible request input. --- #

        db_nearest_1d_point_helper = database._nearest_1d_point_helper

        def wrapped_nearest_1d_point_helper(
            x: int, values: list[int], inner_only: bool = False
        ):
            # Disable the inner_only by default
            return db_nearest_1d_point_helper(x, values, inner_only)

        database._nearest_1d_point_helper = wrapped_nearest_1d_point_helper

        # --- End --- #

        self._session = InferenceSession(
            model=get_perf_model(config, model),
            backend=get_backend(self.config.backend_name),
            database=database,
        )

        # Load XGBoost bucket models
        self.xgb_bucket_models = []
        if xgb_model_path is not None:
            self.xgb_bucket_models = _load_bucket_models(xgb_model_path)

    def _estimate_tp_comm_time_ms(self, batch: ScheduleBatch) -> float:
        if self.platform_config is None or self.config.tp_size <= 1:
            return 0.0

        hidden_size = getattr(self.model, "hidden_size", 0) or 0
        num_layers = getattr(self.model, "num_hidden_layers", 0) or 0
        if hidden_size <= 0 or num_layers <= 0:
            return 0.0

        tokens = batch.batch_size if batch.is_decode() else batch.num_context_tokens
        if tokens <= 0:
            return 0.0

        # Approximate TP communication as two ring collectives per transformer layer.
        collective_count = 2 * num_layers
        dtype_bytes = self.config.data_type.bytes if self.config.data_type else 2
        payload_bytes = tokens * hidden_size * dtype_bytes
        tp_size = self.config.tp_size

        num_nodes = max(int(self.platform_config.num_nodes or 1), 1)
        num_device_per_node = max(int(self.platform_config.num_device_per_node or 1), 1)
        total_devices = num_nodes * num_device_per_node

        if total_devices < tp_size:
            bandwidth, latency_us = self._get_link_profile(cross_node=num_nodes > 1)
            return collective_count * self._estimate_collective_time_ms(
                tp_size, payload_bytes, bandwidth, latency_us
            )

        nodes_used = max(1, math.ceil(tp_size / num_device_per_node))
        local_group_size = min(tp_size, num_device_per_node)
        local_bandwidth, local_latency_us = self._get_link_profile(cross_node=False)

        total_ms = self._estimate_collective_time_ms(
            local_group_size, payload_bytes, local_bandwidth, local_latency_us
        )

        if nodes_used > 1:
            inter_bandwidth, inter_latency_us = self._get_link_profile(cross_node=True)
            total_ms += self._estimate_collective_time_ms(
                nodes_used, payload_bytes, inter_bandwidth, inter_latency_us
            )
            total_ms += self._estimate_collective_time_ms(
                local_group_size, payload_bytes, local_bandwidth, local_latency_us
            )

        return collective_count * total_ms

    def _estimate_collective_time_ms(
        self,
        participants: int,
        payload_bytes: float,
        bandwidth: Optional[float],
        latency_us: float,
    ) -> float:
        if participants <= 1 or bandwidth is None or bandwidth <= 0:
            return 0.0

        ring_factor = 2 * (participants - 1) / participants
        transfer_ms = payload_bytes * ring_factor / bandwidth * 1e3
        launch_ms = 2 * (participants - 1) * latency_us / 1e3
        return transfer_ms + launch_ms

    def _get_link_profile(self, *, cross_node: bool) -> tuple[Optional[float], float]:
        mode = self._get_interconnect_mode()
        configured_bandwidth = self.platform_config.interconnect_bandwidth
        configured_latency_us = self.platform_config.interconnect_latency_us
        device = self.platform_config.device
        hw = device if isinstance(device, AcceleratorInfo) else None

        if cross_node:
            bandwidth = configured_bandwidth
            if bandwidth is None and hw is not None:
                bandwidth = hw.inter_node_bw
            bandwidth = self._apply_interconnect_mode_efficiency(
                bandwidth, cross_node=True
            )
            latency_us = (
                configured_latency_us
                if configured_latency_us is not None
                else INTERCONNECT_MODE_DEFAULT_LATENCY_US.get(mode, 5.0)
            )
            return bandwidth, latency_us

        if self.platform_config.num_nodes <= 1:
            bandwidth = configured_bandwidth
            if bandwidth is None and hw is not None:
                bandwidth = hw.intra_node_bw or hw.inter_node_bw
            bandwidth = self._apply_interconnect_mode_efficiency(
                bandwidth, cross_node=False
            )
            latency_us = (
                configured_latency_us
                if configured_latency_us is not None
                else INTERCONNECT_MODE_DEFAULT_LATENCY_US.get(mode, 1.0)
            )
            return bandwidth, latency_us

        bandwidth = hw.intra_node_bw if hw is not None else None
        if bandwidth is None:
            bandwidth = configured_bandwidth
        latency_us = 1.0 if bandwidth is not None else (
            configured_latency_us
            if configured_latency_us is not None
            else INTERCONNECT_MODE_DEFAULT_LATENCY_US.get(mode, 1.0)
        )
        return bandwidth, latency_us

    def _apply_interconnect_mode_efficiency(
        self, bandwidth: Optional[float], *, cross_node: bool
    ) -> Optional[float]:
        if bandwidth is None:
            return None
        if not cross_node and self.platform_config.num_nodes > 1:
            return bandwidth

        mode = self._get_interconnect_mode()
        efficiency = INTERCONNECT_MODE_BANDWIDTH_EFFICIENCY.get(mode, 1.0)
        return bandwidth * efficiency

    def _get_interconnect_mode(self) -> str:
        mode = self.platform_config.interconnect_mode or "none"
        return mode.lower()

    def _get_database_mode(self, mode: str) -> DatabaseMode:
        return {
            "SILICON": DatabaseMode.SILICON,
            "HYBRID": DatabaseMode.HYBRID,
            "EMPIRICAL": DatabaseMode.EMPIRICAL,
            "SOL": DatabaseMode.SOL,
            "SOL_FULL": DatabaseMode.SOL_FULL,
        }.get(mode.upper(), DatabaseMode.SILICON)

    def ctx_attn_flops_ratio_with_avg(self, reqs: list[FakeRequest]) -> float:
        if len(reqs) == 1:
            return 1.0
        mean_past = np.mean([req.past_kv_length for req in reqs])
        mean_input = np.mean([req.input_length for req in reqs])
        avg_flops = (mean_past + mean_past + mean_input) * mean_input / 2 * len(reqs)

        actual_flops = 0
        for req in reqs:
            actual_flops += (
                (req.past_kv_length + req.past_kv_length + req.input_length)
                * req.input_length
                / 2
            )

        return actual_flops / avg_flops

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        infer_time = 0
        if batch.is_decode():
            # Decode: output sequence length (osl) = 2, input sequence length (isl) = mean(past_kv_length)
            isl = int(np.mean([req.past_kv_length for req in batch.reqs]))
            gen_attn_scale = -1
            # Try to use XGBoost bucket model
            if len(self.xgb_bucket_models) > 0:
                picked = _pick_bucket_model(self.xgb_bucket_models, batch.batch_size)
                if picked is not None:
                    _lo, _hi, _reg, _p = picked
                    xgb_reg = _reg
                    xgb_max_bs = int(_hi)

                    # If XGBoost is enabled, it predicts the ratio: aic_gen_attn_ms / measured_gen_attn_ms.
                    # To make AIC's generation attention closer to measured, scale it by (1 / pred_ratio).
                    pred_ratio = _xgb_predict_ratio(
                        batch.reqs, xgb_reg, max_bs=xgb_max_bs
                    )
                    gen_attn_scale = 1.0 / pred_ratio

            if gen_attn_scale < 0:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size, isl=isl, osl=2
                )
            else:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size,
                    isl=isl,
                    osl=2,
                    gen_seq_imbalance_correction_scale=float(gen_attn_scale),
                )
            summary = self._session.run_static(runtime_config, mode="static_gen")
            latency_dict = summary.get_generation_latency_dict()

        else:
            # Prefill: output sequence length (osl) = 1, input sequence length (isl) = mean(past_kv + input), prefix = mean(past_kv)
            mean_past = np.mean([req.past_kv_length for req in batch.reqs])
            mean_input = np.mean([req.input_length for req in batch.reqs])
            isl = int(mean_past + mean_input)
            prefix = int(mean_past)
            runtime_config = RuntimeConfig(
                batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1
            )

            seq_imbalance_correction_scale = self.ctx_attn_flops_ratio_with_avg(
                batch.reqs
            )
            if seq_imbalance_correction_scale >= 0.4:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size,
                    isl=isl,
                    prefix=prefix,
                    osl=1,
                    seq_imbalance_correction_scale=seq_imbalance_correction_scale,
                )
            else:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1
                )

            summary = self._session.run_static(runtime_config, mode="static_ctx")
            latency_dict = summary.get_context_latency_dict()
        infer_time = sum(latency_dict.values())
        is_oom = summary.check_oom()
        if is_oom:
            logger.warning("Out of memory detected during estimation.")
        if batch.is_decode():
            infer_time *= self.decode_scale_factor
        else:
            infer_time *= self.prefill_scale_factor
        if is_oom:
            return -abs(infer_time) / 1e3
        infer_time += self._estimate_tp_comm_time_ms(batch)
        return infer_time / 1e3
