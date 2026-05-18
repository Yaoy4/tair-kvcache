# HiSim Demo 运行命令

下面是一套可以直接执行的 HiSim demo 命令。命令假设当前目录是 `tair-kvcache` 仓库根目录，HiSim 位于 `./hisim`，模型位于仓库同级目录的 `../models/Qwen3-8B`。

实际跑通时还需要补两点：

1. Python 解释器需要使用已经配好的 `hisim` 虚拟环境。下面统一用 `./hisim/bin/python`。
2. `--dataset-name random` 仍会用到 ShareGPT 样本。当前环境访问 `huggingface.co` 可能失败，因此 benchmark 前先把数据集缓存到本地，并在命令里显式传入 `--dataset-path`。

另外，三个 demo config 已经调整为：

- `single_gpu`：`tp_size=1`
- `dual_gpu_nvlink`：`tp_size=2`
- `two_node_ethernet`：`tp_size=2`

这样 multi-device case 会真正走到通信建模路径，TTFT / TPOT 会受到通信时长影响。

## 1. 单个 demo：single GPU

先开第一个终端，启动 HiSim server：

```bash
cd ./hisim

export HISIM_PYTHON=./hisim/bin/python
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export MODEL_PATH="../../models/Qwen3-8B"
export HISIM_OUTPUT_DIR=./outputs/hisim/demo-single

mkdir -p ./outputs/hisim/demo-single

"${HISIM_PYTHON}" -m hisim.simulation.sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --sim-config-path test/assets/mock/config.demo.single_gpu.json \
  --device cpu \
  --skip-server-warmup
```

再开第二个终端，运行 benchmark：

```bash
cd ./hisim

export HISIM_PYTHON=./hisim/bin/python
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export MODEL_PATH="../../models/Qwen3-8B"
export HISIM_OUTPUT_DIR=./outputs/hisim/demo-single
export SHAREGPT_DATASET=/tmp/ShareGPT_V3_unfiltered_cleaned_split.json

mkdir -p ./outputs/hisim/demo-single

if [ ! -s "${SHAREGPT_DATASET}" ]; then
  curl -L --fail --retry 3 \
    -o "${SHAREGPT_DATASET}" \
    https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

"${HISIM_PYTHON}" -m hisim.simulation.bench_serving \
  --host 127.0.0.1 \
  --warmup-requests 0 \
  --bench-mode simulation \
  --model "${MODEL_PATH}" \
  --backend sglang \
  --dataset-name random \
  --dataset-path "${SHAREGPT_DATASET}" \
  --request-rate 1 \
  --random-input-len 32 \
  --random-output-len 8 \
  --random-range-ratio 1 \
  --num-prompts 2 \
  --output-file ./outputs/hisim_demo_single_bench.json
```

查看输出：

```bash
cat ./outputs/hisim_demo_single_bench.json
cat ./outputs/hisim/demo-single/metrics.json
cat ./outputs/hisim/demo-single/iteration.jsonl
```

## 2. 三组 case 一次跑完

```bash
cd ./hisim

export HISIM_PYTHON=./hisim/bin/python
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export MODEL_PATH="../../models/Qwen3-8B"
export SHAREGPT_DATASET=/tmp/ShareGPT_V3_unfiltered_cleaned_split.json

mkdir -p ./outputs/hisim

if [ ! -s "${SHAREGPT_DATASET}" ]; then
  curl -L --fail --retry 3 \
    -o "${SHAREGPT_DATASET}" \
    https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

for CASE in single_gpu dual_gpu_nvlink two_node_ethernet
do
  pkill -f "hisim.simulation.sglang.launch_server" || true
  sleep 2

  export HISIM_OUTPUT_DIR="./outputs/hisim/${CASE}"
  mkdir -p "${HISIM_OUTPUT_DIR}"

  "${HISIM_PYTHON}" -m hisim.simulation.sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --sim-config-path "test/assets/mock/config.demo.${CASE}.json" \
    --device cpu \
    --skip-server-warmup \
    >"./outputs/${CASE}_server.log" 2>&1 &

  "${HISIM_PYTHON}" - <<'PY'
import requests, time, sys
for _ in range(120):
    try:
        r = requests.get("http://127.0.0.1:30000", timeout=2)
        if r.status_code < 500:
            print("server ready")
            sys.exit(0)
    except Exception:
        time.sleep(1)
print("server not ready")
sys.exit(1)
PY

  "${HISIM_PYTHON}" -m hisim.simulation.bench_serving \
    --host 127.0.0.1 \
    --warmup-requests 0 \
    --bench-mode simulation \
    --model "${MODEL_PATH}" \
    --backend sglang \
    --dataset-name random \
    --dataset-path "${SHAREGPT_DATASET}" \
    --request-rate 1 \
    --random-input-len 32 \
    --random-output-len 8 \
    --random-range-ratio 1 \
    --num-prompts 2 \
    --output-file "./outputs/${CASE}_bench.json"

  echo "===== ${CASE} ====="
  cat "./outputs/${CASE}_bench.json"
  cat "./outputs/hisim/${CASE}/metrics.json"
done
```

## 3. 三个 demo config 文件位置

```bash
./hisim/test/assets/mock/config.demo.single_gpu.json
./hisim/test/assets/mock/config.demo.dual_gpu_nvlink.json
./hisim/test/assets/mock/config.demo.two_node_ethernet.json
```
