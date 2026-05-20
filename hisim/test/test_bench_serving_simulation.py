import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from hisim.simulation import bench_serving


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        return text.split()


def _fake_simulation_metrics():
    return bench_serving.BenchmarkMetrics(
        bench_mode="simulation",
        completed=2,
        duration=1.5,
        total_input=8,
        total_input_text=8,
        total_input_vision=0,
        total_output=8,
        total_output_retokenized=8,
        request_throughput=1.33,
        input_throughput=5.33,
        output_throughput=5.33,
        output_throughput_retokenized=5.33,
        total_throughput=10.66,
        total_throughput_retokenized=10.66,
        mean_ttft_ms=10.0,
        median_ttft_ms=10.0,
        std_ttft_ms=0.0,
        p99_ttft_ms=10.0,
        mean_tpot_ms=5.0,
        median_tpot_ms=5.0,
        std_tpot_ms=0.0,
        p99_tpot_ms=5.0,
        mean_itl_ms=5.0,
        median_itl_ms=5.0,
        std_itl_ms=0.0,
        p95_itl_ms=5.0,
        p99_itl_ms=5.0,
        max_itl_ms=5.0,
        mean_e2e_latency_ms=20.0,
        median_e2e_latency_ms=20.0,
        std_e2e_latency_ms=0.0,
        p99_e2e_latency_ms=20.0,
        concurrency=0.03,
        max_output_tokens_per_s=8.0,
        max_concurrent_requests=2,
        mean_queue_ms=1.0,
        prefix_cache_reused_ratio=0.0,
        disk_prefetch_ratio=0.0,
    )


async def _fake_request(request_func_input, pbar=None):
    output = bench_serving.RequestFuncOutput.init_new(request_func_input)
    output.success = True
    output.generated_text = "ok"
    output.latency = 0.02
    output.ttft = 0.01
    output.itl = [0.01]
    output.output_len = request_func_input.output_len
    output.start_time = 0.0
    if pbar:
        pbar.update(1)
    return output


class BenchmarkSimulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_simulation_uses_stop_endpoint_and_clears_stale_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            for name in ("metrics.json", "iteration.jsonl", "request.jsonl"):
                (output_dir / name).write_text("stale", encoding="utf-8")

            profile_urls = []

            async def fake_profile(api_url: str):
                profile_urls.append(api_url)
                output = bench_serving.RequestFuncOutput()
                output.success = True
                return output

            bench_serving.set_global_args(
                Namespace(
                    bench_mode="simulation",
                    dataset_name="random",
                    warmup_requests=0,
                    output_details=False,
                    output_file=str(output_dir / "result.jsonl"),
                    disable_stream=False,
                    disable_ignore_eos=False,
                    plot_throughput=False,
                    print_requests=False,
                    profile_activities=["CPU"],
                    profile_num_steps=None,
                    profile_by_stage=False,
                    profile_stages=None,
                    tag=None,
                    backend="sglang",
                    num_prompts=2,
                    sharegpt_output_len=None,
                    random_input_len=4,
                    random_output_len=4,
                    random_range_ratio=1.0,
                )
            )

            input_requests = [
                bench_serving.DatasetRow(prompt="a b c d", prompt_len=4, output_len=4),
                bench_serving.DatasetRow(prompt="e f g h", prompt_len=4, output_len=4),
            ]

            with (
                patch.dict(
                    bench_serving.ASYNC_REQUEST_FUNCS, {"sglang": _fake_request}, clear=False
                ),
                patch.object(bench_serving, "async_request_profile", side_effect=fake_profile),
                patch.object(bench_serving, "load_simulation_metrics", return_value=_fake_simulation_metrics()),
                patch.object(bench_serving, "get_sglang_server_info", return_value=None),
                patch.object(bench_serving.time, "sleep", return_value=None),
                patch.dict(os.environ, {"HISIM_OUTPUT_DIR": str(output_dir)}, clear=False),
            ):
                await bench_serving.benchmark(
                    backend="sglang",
                    api_url="http://unit.test/generate",
                    base_url="http://unit.test",
                    model_id="dummy-model",
                    tokenizer=_FakeTokenizer(),
                    input_requests=input_requests,
                    request_rate=1.0,
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=None,
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                )

            self.assertEqual(
                profile_urls,
                [
                    "http://unit.test/start_profile",
                    "http://unit.test/stop_profile",
                ],
            )
            for name in ("metrics.json", "iteration.jsonl", "request.jsonl"):
                self.assertFalse((output_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
