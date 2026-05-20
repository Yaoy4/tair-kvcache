import json
import os
from pathlib import Path
import tempfile
from types import MethodType
from copy import deepcopy
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo

from hisim.simulation.manager import ConfigManager
from hisim.time_predictor import (
    AIConfiguratorTimePredictor,
    ScheduleBatch,
    FakeRequest,
)
from hisim.simulation.types import (
    SchedulerConfig,
    PlatformConfig,
)


cur_dir = Path(__file__).parent


def _get_h100_aic_hw():
    hw = deepcopy(AcceleratorInfo.find_by_hw_name("H100"))
    hw.name = "h100_sxm"
    return hw


class _FakeSummary:
    def __init__(self, latency_dict: dict[str, float], oom: bool = False):
        self._latency_dict = latency_dict
        self._oom = oom

    def get_context_latency_dict(self):
        return self._latency_dict

    def check_oom(self):
        return self._oom


class _FakeSession:
    def __init__(self, summary: _FakeSummary):
        self._summary = summary

    def run_static(self, runtime_config, mode: str):
        assert mode == "static_ctx"
        return self._summary


def _make_comm_only_predictor(
    model: ModelInfo, config: SchedulerConfig, platform_config: PlatformConfig
):
    predictor = AIConfiguratorTimePredictor.__new__(AIConfiguratorTimePredictor)
    predictor.model = model
    predictor.config = config
    predictor.platform_config = platform_config
    return predictor


def test_time_predictor():
    model = ModelInfo.from_modelscope_id("Qwen/Qwen3-8B")
    hw = _get_h100_aic_hw()
    config = SchedulerConfig(
        model=model, backend_name="sglang", backend_version="0.5.6.post2"
    )
    for clz in [
        AIConfiguratorTimePredictor,
    ]:
        predictor = clz(model, hw, config)

        # Prefill
        reqs = [
            FakeRequest(512, 512),
            FakeRequest(1024, 0),
            FakeRequest(512, 0),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0

        # Decode
        reqs = [
            FakeRequest(1, 1024),
            FakeRequest(1, 1024),
            FakeRequest(1, 1024),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0


def test_time_predictor_interconnect_penalty():
    model = ModelInfo.from_modelscope_id("Qwen/Qwen3-8B")
    hw = _get_h100_aic_hw()
    config = SchedulerConfig(
        model=model,
        tp_size=2,
        backend_name="sglang",
        backend_version="0.5.6.post2",
    )

    nvlink = PlatformConfig(
        device=hw,
        num_nodes=1,
        num_device_per_node=2,
        interconnect_mode="nvlink",
        interconnect_bandwidth_gb=100,
        interconnect_latency_us=1,
    )
    ethernet = PlatformConfig(
        device=hw,
        num_nodes=2,
        num_device_per_node=1,
        interconnect_mode="ethernet",
        interconnect_bandwidth_gb=12.5,
        interconnect_latency_us=10,
    )

    reqs = [
        FakeRequest(32, 0),
        FakeRequest(32, 0),
    ]
    batch = ScheduleBatch(reqs)

    nvlink_predictor = AIConfiguratorTimePredictor(
        model, hw, config, platform_config=nvlink
    )
    ethernet_predictor = AIConfiguratorTimePredictor(
        model, hw, config, platform_config=ethernet
    )

    nvlink_latency = nvlink_predictor.predict_infer_time(batch)
    ethernet_latency = ethernet_predictor.predict_infer_time(batch)

    assert ethernet_latency > nvlink_latency


def test_platform_config_interconnect_bandwidth_unit():
    platform = PlatformConfig(device="mock", interconnect_bandwidth_gb=100)
    assert platform.interconnect_bandwidth == 100 * 1e9


def test_platform_config_legacy_interconnect_bandwidth_gbps():
    config_data = {
        "platform": {
            "accelerator": {"name": "H100"},
            "interconnect_bandwidth_gbps": 100,
        }
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as config_file:
        json.dump(config_data, config_file)
        config_path = config_file.name

    original_config_path = os.environ.get("HISIM_CONFIG_PATH")
    ConfigManager._platform_config = None
    try:
        os.environ["HISIM_CONFIG_PATH"] = config_path
        platform = ConfigManager.get_platform_config()
    finally:
        if original_config_path is None:
            os.environ.pop("HISIM_CONFIG_PATH", None)
        else:
            os.environ["HISIM_CONFIG_PATH"] = original_config_path
        ConfigManager._platform_config = None
        os.unlink(config_path)

    assert platform.interconnect_bandwidth_gb == 12.5


def test_time_predictor_topology_penalty():
    model = ModelInfo.from_modelscope_id("Qwen/Qwen3-8B")
    hw = AcceleratorInfo.find_by_hw_name("H100")
    config = SchedulerConfig(
        model=model,
        tp_size=4,
        backend_name="sglang",
        backend_version="0.5.6.post2",
    )
    batch = ScheduleBatch([FakeRequest(32, 0) for _ in range(4)])

    two_node = PlatformConfig(
        device=hw,
        num_nodes=2,
        num_device_per_node=2,
        interconnect_mode="ethernet",
        interconnect_bandwidth_gb=100,
        interconnect_latency_us=1,
    )
    four_node = PlatformConfig(
        device=hw,
        num_nodes=4,
        num_device_per_node=1,
        interconnect_mode="ethernet",
        interconnect_bandwidth_gb=100,
        interconnect_latency_us=1,
    )

    two_node_comm = _make_comm_only_predictor(
        model, config, two_node
    )._estimate_tp_comm_time_ms(batch)
    four_node_comm = _make_comm_only_predictor(
        model, config, four_node
    )._estimate_tp_comm_time_ms(batch)

    assert four_node_comm > two_node_comm


def test_time_predictor_interconnect_mode_penalty():
    model = ModelInfo.from_modelscope_id("Qwen/Qwen3-8B")
    hw = AcceleratorInfo.find_by_hw_name("H100")
    config = SchedulerConfig(
        model=model,
        tp_size=2,
        backend_name="sglang",
        backend_version="0.5.6.post2",
    )
    batch = ScheduleBatch([FakeRequest(32, 0), FakeRequest(32, 0)])

    nvlink = PlatformConfig(
        device=hw,
        num_nodes=1,
        num_device_per_node=2,
        interconnect_mode="nvlink",
        interconnect_bandwidth_gb=100,
    )
    ethernet = PlatformConfig(
        device=hw,
        num_nodes=1,
        num_device_per_node=2,
        interconnect_mode="ethernet",
        interconnect_bandwidth_gb=100,
    )

    nvlink_comm = _make_comm_only_predictor(
        model, config, nvlink
    )._estimate_tp_comm_time_ms(batch)
    ethernet_comm = _make_comm_only_predictor(
        model, config, ethernet
    )._estimate_tp_comm_time_ms(batch)

    assert ethernet_comm > nvlink_comm


def test_time_predictor_oom_stays_negative():
    model = ModelInfo.from_modelscope_id("Qwen/Qwen3-8B")
    config = SchedulerConfig(
        model=model,
        tp_size=2,
        backend_name="sglang",
        backend_version="0.5.6.post2",
    )
    predictor = AIConfiguratorTimePredictor.__new__(AIConfiguratorTimePredictor)
    predictor.model = model
    predictor.config = config
    predictor.prefill_scale_factor = 1.0
    predictor.decode_scale_factor = 1.0
    predictor.xgb_bucket_models = []
    predictor._session = _FakeSession(_FakeSummary({"compute": 1.0}, oom=True))
    predictor._estimate_tp_comm_time_ms = MethodType(
        lambda self, batch: 1000.0, predictor
    )

    latency = predictor.predict_infer_time(ScheduleBatch([FakeRequest(32, 0)]))

    assert latency < 0


def test_h100_accelerator_info():
    hw = AcceleratorInfo.find_by_hw_name("H100")
    assert hw is not None
    assert hw.hbm_capacity_gb == 80
    assert hw.hbm_bandwidth_gb == 3350
    assert hw.intra_node_bandwidth_gb == 450
    assert hw.inter_node_bandwidth_gb == 25


def test_b60_accelerator_info():
    hw = AcceleratorInfo.find_by_hw_name("B60")
    assert hw is not None
    assert hw.vendor == "Intel"
    assert hw.hbm_capacity_gb == 24
    assert hw.hbm_bandwidth_gb == 456
    assert hw.intra_node_bandwidth_gb == 32
    assert hw.inter_node_bandwidth_gb == 50
    assert AcceleratorInfo.find_by_hw_name("Intel B60") == hw


def test_b60_demo_config_uses_aic_data():
    config_path = cur_dir / "assets/mock/config.demo.b60_pcie.json"
    with config_path.open() as f:
        config = json.load(f)

    assert config["platform"]["accelerator"]["name"] == "Intel B60"
    assert config["platform"]["num_nodes"] == 1
    assert config["platform"]["num_device_per_node"] == 8
    assert config["platform"]["interconnect_mode"] == "pcie"
    assert config["platform"]["interconnect_bandwidth_gb"] == 32
    assert config["platform"]["interconnect_latency_us"] == 10
    assert config["predictor"]["device_name"] == "b60"
    assert config["scheduler"]["backend_name"] == "vllm"
    assert config["scheduler"]["backend_version"] == "0.12.0"


def test_rtx_pro_6000_server_accelerator_info():
    hw = AcceleratorInfo.find_by_hw_name("RTX 6000 PRO")
    assert hw is not None
    assert hw.vendor == "NVIDIA"
    assert round(hw.hbm_capacity_gb, 9) == 102.641958912
    assert hw.hbm_bandwidth_gb == 1792
    assert hw.intra_node_bandwidth_gb == 64
    assert hw.inter_node_bandwidth_gb == 50
    assert round(hw.tensor_flops("BF16") / 1e12, 1) == 467.8
    assert round(hw.tensor_flops("FP8") / 1e12, 1) == 935.6
    assert round(hw.tensor_flops("FP4") / 1e12, 1) == 1871.2
    assert AcceleratorInfo.find_by_hw_name("rtx_pro_6000_server") == hw


def test_rtx_pro_6000_server_demo_config_uses_aic_data():
    config_path = cur_dir / "assets/mock/config.demo.rtx_pro_6000_server_pcie.json"
    with config_path.open() as f:
        config = json.load(f)

    assert config["platform"]["accelerator"]["name"] == "RTX 6000 PRO"
    assert config["platform"]["num_nodes"] == 1
    assert config["platform"]["num_device_per_node"] == 2
    assert config["platform"]["interconnect_mode"] == "pcie"
    assert config["platform"]["interconnect_bandwidth_gb"] == 64
    assert config["platform"]["interconnect_latency_us"] == 10
    assert config["predictor"]["device_name"] == "rtx_pro_6000_server"
    assert config["scheduler"]["tp_size"] == 2
    assert config["scheduler"]["data_type"] == "FP8"
    assert config["scheduler"]["kv_cache_data_type"] == "FP8"
    assert config["scheduler"]["backend_name"] == "sglang"
    assert config["scheduler"]["backend_version"] == "0.5.10"


if __name__ == "__main__":
    test_time_predictor()
    test_time_predictor_interconnect_penalty()
    test_platform_config_interconnect_bandwidth_unit()
    test_platform_config_legacy_interconnect_bandwidth_gbps()
    test_time_predictor_topology_penalty()
    test_time_predictor_interconnect_mode_penalty()
    test_time_predictor_oom_stays_negative()
    test_h100_accelerator_info()
    test_b60_accelerator_info()
    test_b60_demo_config_uses_aic_data()
    test_rtx_pro_6000_server_accelerator_info()
    test_rtx_pro_6000_server_demo_config_uses_aic_data()
