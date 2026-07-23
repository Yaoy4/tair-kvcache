# 260721-PD合并情况下的代码逻辑测试报告

> Hisim：ThJiang_Dev
>
> AIC：aiconfigurator-e0735cc

- `tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  {
    "platform": {
      "accelerator": {
        "name": "rtx_pro_6000_server"
      },
      "disk_read_bandwidth_gb": 4,
      "disk_write_bandwidth_gb": 4,
      "memory_read_bandwidth_gb": 64,
      "memory_write_bandwidth_gb": 64
    },
    "predictor": {
      "name": "aiconfigurator",
      "database_path": "/mnt/nfs02/users/zeyuanwa/project/aiconfigurator-e0735cc/src/aiconfigurator/systems",
      "device_name": "rtx_pro_6000_server",
      "prefill_scale_factor": 1.0,
      "decode_scale_factor": 1.0
    },
    "scheduler": {
      "tp_size": 1,
      "ep_size": 1,
      "dp_size": 1,
      "data_type": "FP16",
      "kv_cache_data_type": "FP16",
      "backend_name": "sglang",
      "backend_version": "0.5.10"
    },
    "disagg": {
      "enabled": false,
      "backend": "single_process",
      "decode_queue_mode": "single_replica",
      "prefill": {
        "device_name": "rtx_pro_6000_server",
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "replicas": 1,
        "data_type": "FP16",
        "kv_cache_data_type": "FP16",
        "prefill_scale_factor": 1.0,
        "decode_scale_factor": 1.0,
        "database_path": "/mnt/nfs02/users/zeyuanwa/project/aiconfigurator-e0735cc/src/aiconfigurator/systems",
        "backend_version": "0.5.10"
      },
      "decode": {
        "device_name": "rtx_pro_6000_server",
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "replicas": 1,
        "data_type": "FP16",
        "kv_cache_data_type": "FP16",
        "prefill_scale_factor": 1.0,
        "decode_scale_factor": 1.0,
        "database_path": "/mnt/nfs02/users/zeyuanwa/project/aiconfigurator-e0735cc/src/aiconfigurator/systems",
        "backend_version": "0.5.10"
      },
      "kv_transfer": {
        "bw_gbps": 128,
        "latency_us": 20
      }
    }
  }
  ```

  

## 1 PD合并，TP=1，DP=1

> 从这里开始，1-4使用的commit均为：61ce9f4d5cb58e3ed3c64192dc3d6c881283d9a7
>
> 由于Tianhao的PD分离实现没有改动原本PD合并的实现，因此与后续的实验结果不冲突，可以正常对照

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": false
      
  "scheduler": {
      "tp_size": 1
  ```

- server指令

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup
  ```

- client指令

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  mkdir -p "$HOME/project/cases_same_seed_flush_cache_32kv_GBps"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=1"
  
        python3 -m hisim.simulation.bench_serving \
          --bench-mode simulation \
          --backend sglang \
          --host 127.0.0.1 \
          --port "$HISIM_PORT" \
          --model "$MODEL_PATH" \
          --dataset-name random \
          --num-prompts 200 \
          --warmup-requests 0 \
          --request-rate "$rr" \
          --random-input-len "$il" \
          --random-output-len "$ol" \
          --random-range-ratio 1 \
          --seed 1 \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed_flush_cache_32kv_GBps/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed_flush_cache_32kv_GBps/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

### **性能表**

> mean

| RR   | IL     | OL    | TTFT (ms)  | TPOT (ms/token) | E2E (ms)     |
| ---- | ------ | ----- | ---------- | --------------- | ------------ |
| 1    | 1,024  | 1,024 | 63.46      | 15.00           | 15,413.43    |
| 1    | 1,024  | 4,096 | 76.60      | 50.17           | 205,541.95   |
| 1    | 4,096  | 1,024 | 237.49     | 37.47           | 38,577.40    |
| 1    | 4,096  | 4,096 | 88,771.82  | 63.19           | 347,585.78   |
| 1    | 16,384 | 1,024 | 220,544.16 | 86.08           | 308,861.92   |
| 1    | 16,384 | 4,096 | 898,359.86 | 87.19           | 1,255,775.46 |
| 8    | 1,024  | 1,024 | 98.11      | 45.05           | 46,188.36    |
| 8    | 1,024  | 4,096 | 98.11      | 71.98           | 294,879.88   |
| 8    | 4,096  | 1,024 | 36,813.21  | 72.24           | 110,787.78   |
| 8    | 4,096  | 4,096 | 224,777.95 | 66.78           | 498,345.56   |
| 8    | 16,384 | 1,024 | 303,843.79 | 86.08           | 392,161.54   |
| 8    | 16,384 | 4,096 | 981,659.49 | 87.19           | 1,339,075.09 |
| 64   | 1,024  | 1,024 | 2,860.74   | 50.82           | 54,863.25    |
| 64   | 1,024  | 4,096 | 72,276.13  | 52.42           | 286,957.13   |
| 64   | 4,096  | 1,024 | 47,225.67  | 72.24           | 121,200.23   |
| 64   | 4,096  | 4,096 | 235,190.41 | 66.78           | 508,758.01   |
| 64   | 16,384 | 1,024 | 314,256.24 | 86.08           | 402,574.00   |
| 64   | 16,384 | 4,096 | 992,071.94 | 87.19           | 1,349,487.54 |

- 

## 2 PD合并，TP=2，DP=1

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": false,
      
  "scheduler": {
      "tp_size": 2,
  ```

- server指令

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup
  ```

- client指令

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  mkdir -p "$HOME/project/cases_same_seed_flush_cache_32kv_GBps"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=1"
  
        python3 -m hisim.simulation.bench_serving \
          --bench-mode simulation \
          --backend sglang \
          --host 127.0.0.1 \
          --port "$HISIM_PORT" \
          --model "$MODEL_PATH" \
          --dataset-name random \
          --num-prompts 200 \
          --warmup-requests 0 \
          --request-rate "$rr" \
          --random-input-len "$il" \
          --random-output-len "$ol" \
          --random-range-ratio 1 \
          --seed 1 \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed_flush_cache_32kv_GBps/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed_flush_cache_32kv_GBps/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

### **性能表**

> mean

| RR   | IL     | OL    | TTFT (ms)  | TPOT (ms/token) | E2E (ms)   | 与tp=1时的E2E加速比 |
| ---- | ------ | ----- | ---------- | --------------- | ---------- | ------------------- |
| 1    | 1,024  | 1,024 | 87.45      | 9.49            | 9,793.42   | 1.574x              |
| 1    | 1,024  | 4,096 | 94.96      | 27.15           | 111,265.20 | 1.847x              |
| 1    | 4,096  | 1,024 | 341.58     | 17.34           | 18,083.52  | 2.133x              |
| 1    | 4,096  | 4,096 | 355.74     | 65.44           | 268,351.13 | 1.295x              |
| 1    | 16,384 | 1,024 | 105,079.02 | 116.86          | 225,002.65 | 1.373x              |
| 1    | 16,384 | 4,096 | 371,060.03 | 83.13           | 711,762.89 | 1.764x              |
| 8    | 1,024  | 1,024 | 164.28     | 37.05           | 38,061.81  | 1.214x              |
| 8    | 1,024  | 4,096 | 164.28     | 50.69           | 207,756.78 | 1.419x              |
| 8    | 4,096  | 1,024 | 15,058.72  | 92.62           | 109,903.70 | 1.008x              |
| 8    | 4,096  | 4,096 | 57,596.70  | 70.94           | 348,171.92 | 1.431x              |
| 8    | 16,384 | 1,024 | 188,009.81 | 117.22          | 308,302.22 | 1.272x              |
| 8    | 16,384 | 4,096 | 454,359.65 | 83.13           | 795,062.52 | 1.684x              |
| 64   | 1,024  | 1,024 | 5,040.66   | 41.12           | 47,115.19  | 1.164x              |
| 64   | 1,024  | 4,096 | 5,040.66   | 52.39           | 219,607.04 | 1.307x              |
| 64   | 4,096  | 1,024 | 25,471.17  | 92.62           | 120,316.16 | 1.007x              |
| 64   | 4,096  | 4,096 | 68,009.16  | 70.94           | 358,584.38 | 1.419x              |
| 64   | 16,384 | 1,024 | 198,422.27 | 117.22          | 318,714.68 | 1.263x              |
| 64   | 16,384 | 4,096 | 464,772.11 | 83.13           | 805,474.97 | 1.675x              |



## 3 PD合并，TP=1，DP=1，开启 disable-radix-cache

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": false,
      
  "scheduler": {
      "tp_size": 1,
  ```

- server指令

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --disable-radix-cache
  ```

- client指令

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  mkdir -p "$HOME/project/cases_same_seed_flush_cache_32kv_GBps"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=1"
  
        python3 -m hisim.simulation.bench_serving \
          --bench-mode simulation \
          --backend sglang \
          --host 127.0.0.1 \
          --port "$HISIM_PORT" \
          --model "$MODEL_PATH" \
          --dataset-name random \
          --num-prompts 200 \
          --warmup-requests 0 \
          --request-rate "$rr" \
          --random-input-len "$il" \
          --random-output-len "$ol" \
          --random-range-ratio 1 \
          --seed 1 \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed_flush_cache_32kv_GBps/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed_flush_cache_32kv_GBps/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

### 性能表

> mean

| RR   | IL     | OL    | TTFT (ms)    | TPOT (ms/token) | E2E (ms)     |
| ---- | ------ | ----- | ------------ | --------------- | ------------ |
| 1    | 1,024  | 1,024 | 64.10        | 15.01           | 15,418.73    |
| 1    | 1,024  | 4,096 | 76.15        | 50.19           | 205,586.72   |
| 1    | 4,096  | 1,024 | 240.49       | 37.52           | 38,632.10    |
| 1    | 4,096  | 4,096 | 149,462.62   | 62.44           | 405,155.83   |
| 1    | 16,384 | 1,024 | 223,379.52   | 86.44           | 312,142.83   |
| 1    | 16,384 | 4,096 | 924,631.58   | 67.52           | 1,201,373.89 |
| 8    | 1,024  | 1,024 | 98.63        | 45.10           | 46,231.83    |
| 8    | 1,024  | 4,096 | 79,507.04    | 47.03           | 272,104.05   |
| 8    | 4,096  | 1,024 | 37,950.23    | 70.24           | 109,874.20   |
| 8    | 4,096  | 4,096 | 249,513.27   | 58.95           | 490,957.14   |
| 8    | 16,384 | 1,024 | 306,679.15   | 86.44           | 395,442.46   |
| 8    | 16,384 | 4,096 | 1,007,931.21 | 67.52           | 1,284,673.51 |
| 64   | 1,024  | 1,024 | 2,858.40     | 50.85           | 54,886.38    |
| 64   | 1,024  | 4,096 | 92,814.11    | 45.83           | 280,518.90   |
| 64   | 4,096  | 1,024 | 48,362.69    | 70.24           | 120,286.65   |
| 64   | 4,096  | 4,096 | 259,925.72   | 58.95           | 501,369.60   |
| 64   | 16,384 | 1,024 | 317,091.60   | 86.44           | 405,854.92   |
| 64   | 16,384 | 4,096 | 1,018,343.66 | 67.52           | 1,295,085.97 |

- 大多数短输出 case 符合预期：

  - `IL=1K, OL=1K`：E2E 仅增加 0.03%–0.09%。
  - `IL=4K, OL=1K`：E2E 变化为 +0.14%、-0.82%、-0.75%。

- 与不开 disable-radix-cache 相比，重负载下存在明显的调度敏感性：

  | 场景                 | TTFT 变化            | TPOT 变化 | E2E 变化      |
  | -------------------- | -------------------- | --------- | ------------- |
  | RR=1, IL=4K, OL=4K   | +68.37%              | -1.20%    | +16.56%       |
  | RR=8, IL=1K, OL=4K   | 从 98 ms 增至 79.5 s | -34.66%   | -7.72%        |
  | RR=64, IL=1K, OL=4K  | +28.42%              | -12.56%   | -2.24%        |
  | IL=16K, OL=4K，各 RR | +2.65%～2.92%        | -22.57%   | -4.03%～4.33% |

## 4 PD合并，TP=2，DP=1，开启 disable-radix-cache

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": false,
      
  "scheduler": {
      "tp_size": 2,
  ```

- server指令

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --disable-radix-cache
  ```

- client指令

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  mkdir -p "$HOME/project/cases_same_seed_flush_cache_32kv_GBps"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=1"
  
        python3 -m hisim.simulation.bench_serving \
          --bench-mode simulation \
          --backend sglang \
          --host 127.0.0.1 \
          --port "$HISIM_PORT" \
          --model "$MODEL_PATH" \
          --dataset-name random \
          --num-prompts 200 \
          --warmup-requests 0 \
          --request-rate "$rr" \
          --random-input-len "$il" \
          --random-output-len "$ol" \
          --random-range-ratio 1 \
          --seed 1 \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed_flush_cache_32kv_GBps/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed_flush_cache_32kv_GBps/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

### 性能表

| RR   | IL     | OL    | TTFT (ms)  | TPOT (ms/token) | E2E (ms)   |
| ---- | ------ | ----- | ---------- | --------------- | ---------- |
| 1    | 1,024  | 1,024 | 87.33      | 9.49            | 9,798.66   |
| 1    | 1,024  | 4,096 | 94.46      | 27.17           | 111,337.88 |
| 1    | 4,096  | 1,024 | 347.05     | 17.38           | 18,124.00  |
| 1    | 4,096  | 4,096 | 364.46     | 65.50           | 268,603.85 |
| 1    | 16,384 | 1,024 | 106,134.06 | 114.26          | 223,469.27 |
| 1    | 16,384 | 4,096 | 386,843.86 | 72.50           | 684,021.47 |
| 8    | 1,024  | 1,024 | 163.79     | 37.15           | 38,170.42  |
| 8    | 1,024  | 4,096 | 163.79     | 50.72           | 207,882.57 |
| 8    | 4,096  | 1,024 | 15,083.59  | 92.83           | 110,142.43 |
| 8    | 4,096  | 4,096 | 87,034.19  | 59.07           | 328,999.51 |
| 8    | 16,384 | 1,024 | 189,433.69 | 114.26          | 306,768.89 |
| 8    | 16,384 | 4,096 | 470,143.48 | 72.50           | 767,321.09 |
| 64   | 1,024  | 1,024 | 5,053.47   | 41.17           | 47,179.06  |
| 64   | 1,024  | 4,096 | 5,053.47   | 52.41           | 219,670.90 |
| 64   | 4,096  | 1,024 | 25,496.04  | 92.83           | 120,554.89 |
| 64   | 4,096  | 4,096 | 97,446.64  | 59.07           | 339,411.96 |
| 64   | 16,384 | 1,024 | 199,846.14 | 114.26          | 317,181.35 |
| 64   | 16,384 | 4,096 | 480,555.94 | 72.50           | 777,733.55 |

- 开关 disable-radix-cache 对E2E的影响整体较小

- 与不开 disable-radix-cache 相比，重负载下存在明显的调度敏感性：

  | 场景                     | drc 相对基线 TTFT | TPOT    | E2E         |
  | ------------------------ | ----------------- | ------- | ----------- |
  | RR=8, IL=4K, OL=4K       | +51.11%           | -16.72% | -5.51%      |
  | RR=64, IL=4K, OL=4K      | +43.28%           | -16.72% | -5.35%      |
  | RR=1/8/64, IL=16K, OL=4K | +3.4%～4.3%       | -12.78% | -3.4%～3.9% |



## 5 PD分离，TP=1，DP=1

> 从这里开始使用commit：bdb1514b9c9ba91858946ddaba504dea67818836
>
> 后续开启 disable-radix-cache

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": true,
      
  "disagg": {
      "prefill": {
          "tp_size": 1
  ```

- server指令

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --disable-radix-cache
  ```

- client指令

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  mkdir -p "$HOME/project/cases_same_seed_flush_cache_32kv_GBps"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=1"
  
        python3 -m hisim.simulation.bench_serving \
          --bench-mode simulation \
          --backend sglang \
          --host 127.0.0.1 \
          --port "$HISIM_PORT" \
          --model "$MODEL_PATH" \
          --dataset-name random \
          --num-prompts 200 \
          --warmup-requests 0 \
          --request-rate "$rr" \
          --random-input-len "$il" \
          --random-output-len "$ol" \
          --random-range-ratio 1 \
          --seed 1 \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed_flush_cache_32kv_GBps/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed_flush_cache_32kv_GBps/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

### 性能表

| RR   | IL     | OL    | TTFT (ms)  | TPOT (ms/token) | E2E (ms)     |
| ---- | ------ | ----- | ---------- | --------------- | ------------ |
| 1    | 1,024  | 1,024 | 65.05      | 15.01           | 15,418.05    |
| 1    | 1,024  | 4,096 | 72.70      | 50.17           | 205,507.79   |
| 1    | 4,096  | 1,024 | 256.12     | 37.49           | 38,611.20    |
| 1    | 4,096  | 4,096 | 149,465.92 | 62.42           | 405,063.84   |
| 1    | 16,384 | 1,024 | 184,989.11 | 83.54           | 270,451.62   |
| 1    | 16,384 | 4,096 | 886,632.77 | 66.56           | 1,159,183.46 |
| 8    | 1,024  | 1,024 | 97.03      | 45.08           | 46,211.48    |
| 8    | 1,024  | 4,096 | 79,500.29  | 47.01           | 272,015.17   |
| 8    | 4,096  | 1,024 | 37,427.76  | 69.53           | 108,560.18   |
| 8    | 4,096  | 4,096 | 249,231.20 | 58.84           | 490,178.61   |
| 8    | 16,384 | 1,024 | 267,945.76 | 80.16           | 349,953.53   |
| 8    | 16,384 | 4,096 | 969,633.78 | 66.15           | 1,240,512.04 |
| 64   | 1,024  | 1,024 | 2,912.75   | 50.80           | 54,881.10    |
| 64   | 1,024  | 4,096 | 92,838.08  | 45.82           | 280,452.00   |
| 64   | 4,096  | 1,024 | 47,840.21  | 69.53           | 118,972.63   |
| 64   | 4,096  | 4,096 | 259,643.66 | 58.84           | 500,591.06   |
| 64   | 16,384 | 1,024 | 278,358.22 | 80.16           | 360,365.98   |
| 64   | 16,384 | 4,096 | 980,046.24 | 66.15           | 1,250,924.49 |

- 与非PD分离下场景的加速比（部分结果）

  | IL   | OL   | RR=1       | RR=8       | RR=64      |
  | ---- | ---- | ---------- | ---------- | ---------- |
  | 1K   | 1K   | 1.000x     | 1.000x     | 1.000x     |
  | 1K   | 4K   | 1.000x     | 1.000x     | 1.000x     |
  | 4K   | 1K   | 1.001x     | 1.012x     | 1.011x     |
  | 4K   | 4K   | 1.000x     | 1.002x     | 1.002x     |
  | 16K  | 1K   | **1.154x** | **1.130x** | **1.126x** |
  | 16K  | 4K   | **1.036x** | **1.036x** | **1.035x** |

  - 短输入基本没有收益。 Prefill 很短，PD 资源隔离无法抵消额外 KV 传输和调度成本。
  - IL=16K、OL=1K 收益最大，达到 12.6%–15.4%。 长 prefill、短 decode 是 PD 分离最容易受益的工作负载，prefill 不再阻塞 decode。
  - IL=16K、OL=4K 收益约 3.5%–3.6%。 Decode 本身占据大量时间，因此隔离 prefill 带来的收益被长 decode 稀释。
  - PD 分离引入的 1–19 ms KV 传输开销，相对于长输入场景中数百秒的排队时间很小，因此不会抵消资源隔离收益。



## PD-Agg 与 PD DisAgg 对比