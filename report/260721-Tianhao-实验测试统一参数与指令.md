## 改动

1. client端指令：为了保证输入输出长度都一致，减少无关因素

   ```bash
   --random-range-ratio 1 \
   ```

2. client端指令：增加测试压力

   ```bash
   for rr in 1 8 64; do
     for il in 1024 4096 8192 16384; do
       for ol in 1024 4096; do
   ```

3. server端指令：chunk大小，此设置可以让ISL（1024，4096，16384）在不同情况下chunk数量为（1，1，4）

   ```bash
   --chunked-prefill-size 4096 \
   
   # 这俩应该不用设置
   # --context-length 32768 \
   # --max-prefill-tokens 32768 \
   ```

4. Json配置文件：`max_running_per_replica` ，目前设置为64，【是否应该设置的很大以避免此项影响PD行为？】【没事的。这个值主要是每个replica里面能放的最大的running request数量，这个跟KV cache有关系。这也是我们PD分离调度逻辑的一部分所以我们也要验证，所以这里不用改大。】

5. Json配置文件: platform字段，【服务器的内存和硬盘传输速度是否有参考？】这个我记得我们当时测了一下CPU我们的RT X6000 Pro是Numa架构，两个GPU之间通过CPU来连接但是CPU有两个罗马区域所以其实是要跨IB来连接的。采用PCIE和IB连接，所以实际最大理论带宽是双向128GB/s，但实际考虑到损耗还有可能会单向传输的问题所以我们这里写64GB/s。disk值没有实际测过。

   ```bash
   "platform": {
       "disk_read_bandwidth_gb": 4,
       "disk_write_bandwidth_gb": 4,
       "memory_read_bandwidth_gb": 64,
       "memory_write_bandwidth_gb": 64
     }
   ```



## 目前的指令与配置文件

### server

```bash
# server
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
  --disable-radix-cache

```

### client

```bash
# client
export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
export MODEL_PATH="Qwen/Qwen3-8B"
export HISIM_PORT=30003

mkdir -p "$HOME/project/cases_same_seed_flush_cache_32kv_GBps"
mkdir -p "$HISIM_OUTPUT_DIR"

for rr in 1 8 64; do
  for il in 1024 4096 8192 16384; do
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

### Json

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
    "device_name": "rtx_pro_6000_server"
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
    "enabled": true,
    "backend": "single_process",
    "decode_queue_mode": "single_replica",
    "prefill": {
      "device_name": "rtx_pro_6000_server",
      "tp_size": 1,
      "ep_size": 1,
      "dp_size": 1,
      "pp_size": 1,
      "replicas": 1,
      "max_running_per_replica": 64,
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
      "max_running_per_replica": 64,
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


### PD合并情况下的参数配置

TP=2
```json
{
  "platform": {
    "accelerator": { "name": "rtx_pro_6000_server" },
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
    "tp_size": 2,
    "ep_size": 1,
    "dp_size": 1,
    "data_type": "FP16",
    "kv_cache_data_type": "FP16",
    "backend_name": "sglang",
    "backend_version": "0.5.10"
  }
}
```

DP=2（未测试，不知道是否能通过）
```json
{
  "platform": {
    "accelerator": { "name": "rtx_pro_6000_server" },
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
    "dp_size": 2,
    "data_type": "FP16",
    "kv_cache_data_type": "FP16",
    "backend_name": "sglang",
    "backend_version": "0.5.10"
  }
}
```