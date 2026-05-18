from pathlib import Path
from types import MethodType
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo

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


def test_time_predictor():
    model = ModelInfo.from_modelscope_id("Qwen/Qwen3-8B")
    hw = AcceleratorInfo.find_by_hw_name("H20")
    # H20 don't exist in aiconfigurator's system database yet.
    hw.name = "h100_sxm"  # (tmp) AIConfigurator internal device name
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
    hw = AcceleratorInfo.find_by_hw_name("H20")
    hw.name = "h100_sxm"
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


if __name__ == "__main__":
    test_time_predictor()
    test_time_predictor_interconnect_penalty()
    test_platform_config_interconnect_bandwidth_unit()
    test_time_predictor_oom_stays_negative()
