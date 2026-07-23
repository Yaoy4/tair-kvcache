# HiSim + AIC 实验设置

## 1. 实验目标

- 本实验用于验证 **HiSim + AIConfigurator** 对真实 SGLang LLM inference 性能的预测精度。
- HiSim 不直接运行真实 GPU workload，而是通过仿真方式预测以下指标：
  - **TTFT**：Time to First Token
  - **TPOT**：Time per Output Token
  - **Throughput**：总输出 token 吞吐量
- AIConfigurator 负责提供底层 **kernel-level latency prediction**。
- HiSim 负责模拟 SGLang 的 **token-level scheduling behavior**。

## 2. 被测模型

- 模型：Qwen-8B
- 精度：**FP16**
- KV-cache 数据类型：**FP16**
- Tensor Parallelism：`tp_size = 1`
- Expert Parallelism：`ep_size = 1`
- 实验仅验证该单一 dense model，没有扩展到 MoE、大模型或多 GPU 配置。

## 3. 真实对照基准

- Ground truth 来自真实 **SGLang benchmark**。
- 真实 benchmark 运行在 **NVIDIA RTX PRO 6000 Server 单卡上。
- HiSim + AIC 的预测结果与真实 SGLang 结果逐 cell 对比。
- 对比粒度与 workload 参数保持一致，包括 ISL、OSL、concurrency、request count 和输出长度控制方式。

## 4. 硬件建模条件

- - GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition
 - 架构：Blackwell / SM 12.0
 - SM 数量：188
 - 显存：97,887 MiB GDDR7
 - Memory bus：512-bit
 - Memory clock：12,481 MHz
 - Memory bandwidth：约 1,598 GB/s
 - Max boost clock：2,430 MHz
 - Application clock：N/A（已 deprecated）
 - TDP：600W
 - FP16 Tensor Core 性能：约 500 TFLOPS
 - Driver：610.43.02
 - CUDA：13.3
 - HiSim 配置中： - memory_read_bandwidth_gb = 1598
 - memory_write_bandwidth_gb = 1598

## 5. HiSim 软件环境

- HiSim repository：`gca.arch.tair-kvcache`
- Branch：`support-sglang-0.5.10`
- Key commit：`e55c92b`
- 该 commit 包含：
  - RTX PRO 6000 accelerator entry
  - per-forward overhead model
- HiSim server 端口：`30100`
- 使用配置文件：

```text
hisim-accuracy/hisim_config_rtx_pro_6000_fp16.json
```

## 6. AIConfigurator 设置

- AIConfigurator repository：`gca.arch.aiconfigurator`
- Branch：`tracelist`
- Key commits：
  - `dadf49e`：GEMM + attention performance data
  - `43a8856`：MoE performance data
- Predictor name：`aiconfigurator`
- Device name：`rtx_pro_6000_workstation`
- Database path：

```text
<workspace>/gca.arch.aiconfigurator/src/aiconfigurator/systems
```

- AIC database 使用真实 RTX PRO 6000 上采集得到的 **silicon-calibrated kernel latency data**。
- 该 database 是 HiSim + AIC 的底层算子性能预测基础。

## 7. AIC Kernel Database 覆盖范围

- GEMM cases：**101K**
- Context attention cases：**17K**
- Generation attention cases：**9K**
- MoE cases：**4,720**
- 覆盖不同 batch size、sequence length 和 head dimension 下的 kernel latency。
- HiSim + AIC 使用这些 kernel latency 数据预测每次 forward pass 的 GPU kernel execution time。

## 8. HiSim Platform 配置

```json
{
  "platform": {
    "accelerator": {
      "name": "RTX_PRO_6000"
    },
    "num_device_per_node": 1,
    "memory_read_bandwidth_gb": 1792,
    "memory_write_bandwidth_gb": 1792
  }
}
```

## 9. HiSim Predictor 配置

```json
{
  "predictor": {
    "name": "aiconfigurator",
    "database_path": "<workspace>/gca.arch.aiconfigurator/src/aiconfigurator/systems",
    "device_name": "rtx_pro_6000_workstation",
    "prefill_scale_factor": 1.0,
    "decode_scale_factor": 1.0,
    "prefill_overhead_ms": 12.0,
    "decode_overhead_ms": 0.2
  }
}
```

- `prefill_scale_factor = 1.0`
- `decode_scale_factor = 1.0`
- `prefill_overhead_ms = 12.0`
- `decode_overhead_ms = 0.2`
- scale factor 保持为 1.0，表示不额外缩放 AIC 的 kernel prediction。
- overhead 用于补偿 AIC kernel model 未覆盖的 CPU-side 开销。

## 10. Per-forward Overhead 建模

- Prefill overhead：**12.0 ms**
- Decode overhead：**0.2 ms**
- 该 overhead 覆盖真实 inference 中除 GPU kernel execution 之外的额外开销，包括：
  - scheduler loop / batch formation
  - CUDA kernel launch / synchronization
  - KV-cache block management
  - token sampling
  - detokenization
- 文档中说明，加入 overhead model 后，HiSim 的 TTFT MAPE 从未加入 overhead 时的 **28.0%** 降低到约 **10.2%**。

## 11. HiSim Scheduler 配置

```json
{
  "scheduler": {
    "tp_size": 1,
    "ep_size": 1,
    "data_type": "FP16",
    "kv_cache_data_type": "FP16",
    "backend_name": "sglang",
    "backend_version": "0.5.9"
  }
}
```

- Backend：`sglang`
- Backend version：`0.5.9`
- `tp_size = 1`
- `ep_size = 1`
- `data_type = FP16`
- `kv_cache_data_type = FP16`
- 该配置用于使 HiSim 的 scheduler 行为与真实 SGLang benchmark 设置对齐。

## 12. 真实 SGLang Server 对齐配置

```json
{
  "model-path": "meta-llama/Llama-3.1-8B-Instruct",
  "dtype": "float16",
  "tp-size": "1",
  "mem-fraction-static": "0.88",
  "disable-radix-cache": true,
  "chunked-prefill-size": "8192",
  "max-running-requests": "64"
}
```

- HiSim + AIC 的仿真目标是对齐上述 SGLang serving 配置下的真实性能。
- 其中 `chunked-prefill-size = 8192` 是与 SGLang scheduler 行为相关的重要配置。
- `max-running-requests = 64` 决定最大运行请求数上限。

## 13. 实验参数网格

- 总计：**55 个 cells**

### 13.1 Base Sweep

- Cell 数量：**40**
- 参数设置：
  - `ISL ∈ {512, 1024, 2048, 4096}`
  - `OSL ∈ {512, 1024}`
  - `Concurrency ∈ {1, 2, 4, 8, 16}`
- Grid size：

```text
4 × 2 × 5 = 40 cells
```

### 13.2 ISL=8192 Expansion

- Cell 数量：**7**
- 参数设置：
  - `ISL = 8192`
  - `OSL = 1024`
  - `Concurrency ∈ {1, 2, 4, 8, 16, 32, 64}`

### 13.3 High-concurrency Expansion

- Cell 数量：**8**
- 参数设置：
  - `ISL ∈ {512, 1024, 2048, 4096}`
  - `OSL = 1024`
  - `Concurrency ∈ {32, 64}`

## 14. 每个 Cell 的 Workload 设置

- 请求数量：

```text
num_requests = concurrency × 10
```

- 示例：
  - `concurrency = 8` 时，请求数为 `80`
  - `concurrency = 64` 时，请求数为 `640`

- Request rate：

```text
request_rate = inf
```

- 含义：closed-loop、maximum throughput 模式。

- Random range ratio：

```text
random_range_ratio = 0.8
```

- 含义：实际输入长度从以下区间均匀采样：

```text
[0.8 × ISL, ISL]
```

- 输出长度控制：启用 `--ignore-eos`
- 作用：保证生成长度严格等于目标 OSL。

## 15. HiSim 运行方式

- 对每个参数 cell，HiSim 启动一个 simulation server。
- Benchmark runner 调用 HiSim server，并收集该 cell 的 simulated metrics。
- Base sweep 运行命令示例：

```bash
source hisim-venv/bin/activate

python3 hisim-accuracy/benchmarks/run_benchmark.py hisim \
    --run-tag <tag> \
    --isl 512 1024 2048 4096 \
    --osl 512 1024 \
    --conc 1 2 4 8 16 \
    --hisim-config hisim-accuracy/hisim_config_rtx_pro_6000_fp16.json
```

- ISL=8192 expansion 运行命令示例：

```bash
python3 hisim-accuracy/benchmarks/run_benchmark.py hisim \
    --run-tag ix_aligned_expansion \
    --isl 8192 \
    --osl 1024 \
    --conc 1 2 4 8 16 32 64 \
    --hisim-config hisim-accuracy/hisim_config_rtx_pro_6000_fp16.json
```

- High-concurrency expansion 运行命令示例：

```bash
python3 hisim-accuracy/benchmarks/run_benchmark.py hisim \
    --run-tag ix_aligned_highconc \
    --isl 512 1024 2048 4096 \
    --osl 1024 \
    --conc 32 64 \
    --hisim-config hisim-accuracy/hisim_config_rtx_pro_6000_fp16.json
```

## 16. HiSim 仿真流程

- Request 进入 queue。
- Scheduler 根据当前系统状态形成 batch。
- HiSim 调用 AIC 对当前 batch 预测 forward latency。
- Forward latency 由两部分组成：

```text
forward_latency = AIC_predicted_kernel_time + per_forward_overhead
```

- 对 prefill forward 使用：

```text
prefill_overhead_ms = 12.0
```

- 对 decode forward 使用：

```text
decode_overhead_ms = 0.2
```

- Simulated clock 按 forward latency 推进。
- 每个 request 的 generated token timeline 被更新。
- 最终从模拟出的 token stream 中计算：
  - TTFT
  - TPOT
  - Throughput

## 17. 评估指标

### 17.1 TTFT

- 全称：**Time to First Token**
- 定义：从 request submission 到 first generated token 的时间。
- 单位：ms

### 17.2 TPOT

- 全称：**Time per Output Token**
- 定义：first token 之后的平均 inter-token latency。
- 单位：ms

### 17.3 Throughput

- 定义：所有并发请求合计的 output tokens per second。
- 单位：tokens/s

### 17.4 MAPE

- 全称：**Mean Absolute Percentage Error**
- 定义：

```text
MAPE = mean(|actual - predicted| / actual × 100%)
```

## 18. Accuracy Target

- TTFT MAPE：`< 10%`
- TPOT MAPE：`< 10%`
- Throughput MAPE：`< 10%`

## 19. 实验控制变量

- HiSim + AIC 与真实 SGLang benchmark 使用相同模型。
- 使用相同目标硬件平台。
- 使用相同 workload grid。
- 使用相同 ISL、OSL、concurrency 设置。
- 使用相同 request count 规则。
- 使用相同 random-range-ratio。
- 使用相同输出长度控制方式。
- 使用 AIC 在目标 GPU 上采集得到的 silicon-calibrated kernel database。

## 20. 实验核心特点

- AIC 提供硬件校准后的 kernel latency prediction。
- HiSim 替代 analytical IFB macro scheduling model。
- HiSim 在 token 粒度模拟 SGLang scheduler。
- 仿真过程覆盖：
  - request lifecycle
  - queueing
  - batch formation
  - prefill phase
  - decode phase
  - forward latency
  - token generation timeline
- 该设置使预测误差主要来源于：
  - HiSim scheduler simulation 的建模精度
  - per-forward overhead model 的校准效果
- 因为 kernel database、模型、硬件和 workload 设置保持一致，所以实验主要评估 **HiSim token-level scheduling simulation + AIC kernel prediction** 对真实 SGLang inference 行为的建模能力。

## 21. 简要总结

该实验将真实 SGLang benchmark 作为 ground truth，在相同模型、相同硬件、相同 workload 和相同 kernel latency database 的条件下，使用 HiSim 对 SGLang 的 token-level scheduling 过程进行仿真，并结合 AIConfigurator 的 silicon-calibrated kernel prediction 计算每次 forward pass 的 latency。通过对 55 个参数 cells 的 TTFT、TPOT 和 throughput 进行 MAPE 对比，该实验验证 HiSim + AIC 是否能够准确预测真实 SGLang inference performance。
