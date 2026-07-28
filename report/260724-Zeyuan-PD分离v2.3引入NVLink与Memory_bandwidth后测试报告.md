# HiSim NVLink 与 Host Memory Bandwidth 测试报告

**模型：** Qwen/Qwen3-8B（FP16）  
**仿真后端：** SGLang 0.5.6post2 + HiSim + AIConfigurator  
**测试日期：** 2026-07-24 至 2026-07-27  
**数据规模：** 11 套配置、198 个 workload、每个 workload 200 条请求  
**状态：** 负载趋势总体合理；存在特殊请求污染、TP/Replica 扩展非单调；带宽参数已进入直接受控路径，但均未成为当前 workload 的关键瓶颈

---

## 1. 执行摘要

本报告评估新版 HiSim 在更新 Host Memory Bandwidth、KV 传输带宽和相关 bug 修复后的性能表现。测试覆盖两套 PD 合并拓扑、四套基于 TP 的 PD 分离拓扑、五套基于 Replica 的 PD 分离拓扑，以及三套同拓扑带宽敏感性对照，统一使用 `seed=1`。

核心结论如下：

1. **负载趋势总体合理。** RR 增大主要推高排队延迟；IL 增大显著增加 Prefill 成本并降低吞吐；OL 增大显著增加 E2E。多个长输入 workload 在 RR=8 时已经饱和，因此 RR 从 8 增至 64 后 TPOT 和吞吐保持不变，而 TTFT/E2E 继续增加。
2. **Decode 仍是更有效的扩容方向。** 在 6 设备 Replica 预算下，P2D4 相对 P4D2 的 18/18 组 E2E 均降低，16/18 组吞吐提高；相对变化简单平均为 TTFT 降低 11.8%、TPOT 降低 6.2%、E2E 降低 24.6%、吞吐提高 31.4%。同样结论也出现在 TP 拓扑中。
3. **增加 Prefill Replica 的边际收益接近零。** Replica P2D2 扩至 P4D2 后，平均 E2E 仅降低 0.4%、吞吐仅提高 0.4%；P3D1 相对 P1D1 的平均 E2E 反而增加 7.8%、吞吐下降 5.3%。这说明当前长输入场景没有随 Prefill Replica 数量获得预期扩展。
4. **P8D8 提升总吞吐，但长输入 TPOT 反向恶化。** 相对 Replica P2D4，P8D8 在 18/18 组提高吞吐，平均提高 40.1%，但 `IL=16384` 的 TPOT 增加 37.2% 至 61.3%，其中 `OL=1024` 的 E2E 反而增加约 11% 至 14%。
5. **TP 与 Replica 扩展均存在非单调性。** TP4 并未全面优于 TP2；相同设备数下 Replica P2D2/P2D4 的 E2E 简单平均分别比对应 TP 拓扑高约 17%/21%。拓扑类型不能只按总设备数等价替换。
6. **Host Memory Bandwidth 已生效，但当前负载几乎不敏感。** TP4 PD 合并中，读写带宽由 480 降至 64 GB/s 后，累计 L2 load 从 79.14 ms 增至 359.46 ms，但 121,113 次 iteration 中仅 27 次发生 load，且全部被上一轮 374--454 ms 的 forward 覆盖，E2E 与吞吐 18/18 组不变。Replica-P2D2 PD 分离中没有发生 L2 load，核心指标与 480 GB/s 基线逐项完全相同。
7. **KV transfer 带宽已生效，但传输不是本轮瓶颈。** Replica-P2D2 的 `bw_gbps` 从 811 降至 128 后，典型 KV transfer 从 0.208--2.998 ms 增至 1.211--18.889 ms，倍率为 5.82--6.30 倍；但相对数万至百万毫秒的 Prefill/Decode queue 仍很小，最终 E2E 和吞吐的平均相对变化均低于 0.005%。
8. **结果文件完整但 workload 受污染。** 252 个 workload 均有结果且 `completed=200`；三套 PD 合并数据完全合规，十一套 PD 分离数据均有多数 workload 混入一条 `input_length=1`、`output_length=1`、`created_time=0` 的特殊请求。


## 2. 测试目标与验收口径

本轮测试用于验证更新后的 Host Memory Bandwidth 和 KV 传输带宽参数是否进入仿真路径，并检查不同 PD 拓扑下的 TTFT、TPOT、E2E 和吞吐是否符合负载、排队和容量规律。

| 检查项 | 通过标准 |
| --- | --- |
| 矩阵完整性 | 每套配置包含 18 个 RR/IL/OL 组合 |
| 文件完整性 | 每组包含顶层汇总及 `request.jsonl`、`iteration.jsonl`、`metrics.json` |
| 运行完成度 | 每组 `completed=200`，核心指标非空 |
| Workload 一致性 | 200 条请求均符合目标 IL/OL；内部请求不得进入业务指标 |
| 指标一致性 | E2E 与 TTFT、TPOT 聚合关系自洽，顶层与原始汇总一致 |
| 趋势合理性 | 增加负载不应降低排队压力；扩容收益若非单调，需要可验证解释 |

## 3. 测试环境与软件版本

| 项目 | 值 |
| --- | --- |
| HiSim commit | `49b144b9fd48a22d4730d1d0180a228f87b86f21` |
| AIConfigurator commit | `e0735ccca08c790b37c511a20f484a62a4127118` |
| 模型 | `Qwen/Qwen3-8B` |
| 后端 | SGLang 0.5.10 |
| 数据类型 | FP16；KV Cache FP16 |
| Host CPU | 2 × Intel Xeon Platinum 8468 |
| Host Memory Bandwidth | 读 480 GB/s，写 480 GB/s（仿真假设） |
| KV transfer | 配置值 811 GB/s；固定延迟 20 μs |
| Cache | SGLang Radix Cache + HiCache L2，`write_through`，`kernel` I/O backend |
| 随机种子 | 1 |

> 当前核心 JSON 文件在工作区存在未提交修改，各套实验通过修改同一个配置文件依次运行。

## 4. 测试设计

### 4.1 配置矩阵

| 配置 | 扩展方式 | Prefill | Decode | 总设备数 |
| --- | --- | ---: | ---: | ---: |
| PD 合并 TP2 | 合并 | 共享 TP2 | 共享 TP2 | 2 |
| PD 合并 TP4 | 合并 | 共享 TP4 | 共享 TP4 | 4 |
| PD 分离 P1D1 | 基线 | TP1 × 1 replica | TP1 × 1 replica | 2 |
| PD 分离 TP-P2D2 | TP | TP2 × 1 replica | TP2 × 1 replica | 4 |
| PD 分离 TP-P4D2 | TP | TP4 × 1 replica | TP2 × 1 replica | 6 |
| PD 分离 TP-P2D4 | TP | TP2 × 1 replica | TP4 × 1 replica | 6 |
| PD 分离 Replica-P3D1 | Replica | TP1 × 3 replicas | TP1 × 1 replica | 4 |
| PD 分离 Replica-P2D2 | Replica | TP1 × 2 replicas | TP1 × 2 replicas | 4 |
| PD 分离 Replica-P4D2 | Replica | TP1 × 4 replicas | TP1 × 2 replicas | 6 |
| PD 分离 Replica-P2D4 | Replica | TP1 × 2 replicas | TP1 × 4 replicas | 6 |
| PD 分离 Replica-P8D8 | Replica | TP1 × 8 replicas | TP1 × 8 replicas | 16 |

分离模式下，实际角色拓扑由 `disagg.prefill/decode` 内的 `tp_size` 与 `replicas` 共同决定。顶层 `scheduler.tp_size=1` 不代表 P/D 两侧的实际拓扑。Decode 多副本实验使用 `decode_queue_mode=per_replica_queue`；P3D1 的 Decode 仅有 1 个副本，沿用单队列不影响拓扑含义。

三套敏感性实验保持对应拓扑不变，仅调整带宽参数：

| 对照实验 | 拓扑 | 基线 | 变量配置 |
| --- | --- | --- | --- |
| PD 合并 Host BW | TP4、DP1 | read/write=480 GB/s | read/write=64 GB/s |
| PD 分离 Host BW | Replica-P2D2 | read/write=480 GB/s | read/write=64 GB/s |
| PD 分离 KV BW | Replica-P2D2 | `bw_gbps=811` | `bw_gbps=128` |

### 4.2 工作负载矩阵

每套配置运行 $3 \times 3 \times 2 = 18$ 个 workload，共 252 个 workload：

| 参数 | 取值 |
| --- | --- |
| Request Rate (RR) | 1、8、64 request/s |
| Input Length (IL) | 1024、4096、16384 token |
| Output Length (OL) | 1024、4096 token |
| 请求数 | 每组 200 |
| 长度随机范围 | `random_range_ratio=1`，目标为固定长度 |
| Cache 处理 | 每组使用 `--flush-cache` |

### 4.3 指标定义

| 指标 | 定义 |
| --- | --- |
| TTFT | 请求提交到首个输出 token 的平均延迟，单位 ms |
| TPOT | 首 token 之后每个输出 token 的平均耗时，单位 ms/token |
| E2E | 请求提交到输出完成的平均端到端延迟，单位 ms |
| Output throughput | 全部请求每秒生成的输出 token 数，单位 token/s |
| KV transfer | Prefill 到 Decode 的 KV Cache 传输时间，单位 ms |

对于固定 OL，汇总指标应近似满足：

$$E2E = TTFT + (OL - 1) \times TPOT$$

14 套结果均通过该一致性检查；PD 分离结果的最大相对误差低于 0.051%，PD 合并结果的轻微偏差来自请求级长度和均值聚合方式。

## 5. 互联与 Host Memory Bandwidth 假设

### 5.1 NVLink 与 KV 传输带宽

- RTX PRO 6000 本不支持 NVLink 技术，采用其他GPU的数据
- NVLink的速度由两个因素决定：
  - NVLink 代际
  - GPU-GPU之间的实际链路数

#### 参考表格

| GPU / 架构代表     | NVLink 代际 | 每 GPU 最大链路数 | 单链路双向聚合带宽 | 每 GPU 峰值双向聚合带宽 |
| ------------------ | ----------- | ----------------- | ------------------ | ----------------------- |
| P100 SXM2 / Pascal | 第一代      | 4                 | 约 40 GB/s         | 约 160 GB/s             |
| V100 SXM2 / Volta  | 第二代      | 6                 | 约 50 GB/s         | 300 GB/s                |
| A100 SXM / Ampere  | 第三代      | 12                | 约 50 GB/s         | 600 GB/s                |
| H100 SXM / Hopper  | 第四代      | 18                | 约 50 GB/s         | 900 GB/s                |
| B200 / Blackwell   | 第五代      | 18                | 约 100 GB/s        | 1.8 TB/s                |
| Rubin 平台         | 第六代      | 36                | 约 100 GB/s        | 3.6 TB/s                |

#### 实际仿真选择

- 没有找到与本拓扑严格对应的官方实测数据，根据下面的外部资料暂定 **811 GB/s** 作为敏感性测试假设。

  > refer：[GB200 NVL72 多节点互联方案性能对比 - 知乎](https://zhuanlan.zhihu.com/p/2002069773324948622)


### 5.2 Host Memory Bandwidth

- **服务器情况：**

  - 2 颗 Intel Xeon Platinum 8468
  - 每颗 CPU 支持 8 个 DDR5 内存通道
  - 最高支持 DDR5-4800

- **理论带宽**

  - 单颗CPU： `307.2 GB/s`
  - 双路整机：`307.2 × 2 = 614.4 GB/s`

  > 单通道：4800 MT/s × 8 Byte = 38.4 GB/s
  >
  > 单颗 CPU：38.4 × 8 通道 = 307.2 GB/s
  >
  > 双路整机：307.2 × 2 = 614.4 GB/s

- 本轮采用 480 GB/s，约为双路理论峰值 614.4 GB/s 的 78.1%。该值属于工程假设，并非本机 STREAM 或同类工具的实测值。

- 参数设置

  > 开启 L2 必须开启 L1

  |            | 服务端设置                                             | 含义                                    |
  | ---------- | ------------------------------------------------------ | --------------------------------------- |
  | `no_cache` | `--disable-radix-cache --disable-chunked-prefix-cache` | 禁用 GPU 前缀缓存                       |
  | `L1`       | 不添加上述禁用参数，也不开启 HiCache                   | 使用默认 Radix Cache，只在 GPU HBM 命中 |
  | `L2`       | `--enable-hierarchical-cache` 加 HiCache 参数          | GPU HBM + Host DRAM 两级缓存            |

- L2 完整server端指令（修改自 `260721-实验测试统一参数与指令.md` )

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
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel
  ```

  > hicache-ratio：HiCache 池 相对于 KV cache 池大小。也可以设置 hicache-size 直接指定大小。
  >
  > hicache-write-policy：
  >
  > | 策略                      | 行为                                   |
  > | ------------------------- | -------------------------------------- |
  > | `write_through`           | 立即写入下一层                         |
  > | `write_through_selective` | 达到访问次数阈值后才下沉，主要缓存热点 |
  > | `write_back`              | 上层缓存被淘汰时才写入下一层，I/O 最少 |
  >
  > hicache-io-backend：
  >
  > | 值       | 含义                                            |
  > | -------- | ----------------------------------------------- |
  > | `direct` | 使用标准 CUDA 内存拷贝路径                      |
  > | `kernel` | 使用 SGLang 的 GPU-assisted KV Cache I/O kernel |





## 6. 实验结果

- Hisim commit：`49b144b9fd48a22d4730d1d0180a228f87b86f21`
- AIC：`e0735ccca08c790b37c511a20f484a62a4127118`

- 核心配置文件：`tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  {
    "platform": {
      "accelerator": {
        "name": "rtx_pro_6000_server"
      },
      "disk_read_bandwidth_gb": 4,
      "disk_write_bandwidth_gb": 4,
      "memory_read_bandwidth_gb": 480,
      "memory_write_bandwidth_gb": 480
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
        "bw_gbps": 811,
        "latency_us": 20
      }
    }
  }
  ```

### 6.1 分析口径与数据完整性

- 14 套配置均覆盖 `RR={1,8,64}`、`IL={1024,4096,16384}`、`OL={1024,4096}`，即每套 18 个 workload；每组均有 200 条请求且 `completed=200`，TTFT、TPOT、E2E 和吞吐字段完整。
- 三套 PD 合并实验的 18 组请求长度均符合目标。
- **异常：**十一套 PD 分离实验中均存在 `input_length=1`、`output_length=1`、`created_time=0` 的特殊请求。除 TP-P2D4 有 16 组受影响外，其余十套 PD 分离配置均有 17 组受影响，共影响 186 个 workload。受影响组实际由 199 条目标请求和 1 条 `1x1` 请求组成；OL=1024 时总输出为 203,777，OL=4096 时为 815,105。
- 下表均沿用原始汇总文件的 mean 指标，未剔除该特殊请求。由于异常请求只占 0.5%，大部分趋势仍可参考，但 PD 分离结果不宜直接作为最终基线，尤其是低 RR 下的延迟和吞吐比较。
- 所有配置仅有 seed=1，当前结论描述确定性模拟中的趋势，不代表跨 seed 的统计置信度。

| 配置 | Workload | 完成请求 | 长度完全合规组 | 特殊 `1x1` 请求影响组 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| PD 合并 TP2 | 18/18 | 200/组 | 18/18 | 0 | 通过 |
| PD 合并 TP4 | 18/18 | 200/组 | 18/18 | 0 | 通过 |
| PD 合并 TP4、Host BW64 | 18/18 | 200/组 | 18/18 | 0 | 通过 |
| PD 分离 P1D1 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 TP-P2D2 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 TP-P4D2 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 TP-P2D4 | 18/18 | 200/组 | 2/18 | 16 | 有条件通过 |
| PD 分离 Replica-P3D1 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 Replica-P2D2 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 Replica-P4D2 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 Replica-P2D4 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 Replica-P8D8 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 Replica-P2D2、Host BW64 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |
| PD 分离 Replica-P2D2、KV BW128 | 18/18 | 200/组 | 1/18 | 17 | 有条件通过 |



### 6.2 PD 合并

#### TP=2、DP=1

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": false
      
  "scheduler": {
      "tp_size": 2
  ```

- server

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel
  ```

- client

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  
  mkdir -p "$HOME/project/cases_same_seed"
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
          --output-file "$HOME/project/cases_same_seed/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

- 结果：

  - seed=1

  | RR   | IL    | OL   | TTFT (ms)  | TPOT (ms/token) | E2E (ms)   | Output throughput (token/s) |
  | ---- | ----- | ---- | ---------- | --------------- | ---------- | --------------------------- |
  | 1    | 1024  | 1024 | 87.64      | 9.488           | 9,793.42   | 1,014.72                    |
  | 1    | 1024  | 4096 | 95.16      | 27.148          | 111,265.20 | 2,770.17                    |
  | 1    | 4096  | 1024 | 342.30     | 17.340          | 18,083.52  | 1,004.35                    |
  | 1    | 4096  | 4096 | 356.45     | 65.442          | 268,351.13 | 1,874.22                    |
  | 1    | 16384 | 1024 | 105,079.03 | 116.057         | 224,167.71 | 430.42                      |
  | 1    | 16384 | 4096 | 371,060.69 | 83.322          | 712,545.55 | 649.39                      |
  | 8    | 1024  | 1024 | 164.61     | 37.045          | 38,061.81  | 3,733.66                    |
  | 8    | 1024  | 4096 | 164.61     | 50.694          | 207,756.78 | 3,595.35                    |
  | 8    | 4096  | 1024 | 15,059.47  | 92.624          | 109,903.70 | 1,681.39                    |
  | 8    | 4096  | 4096 | 57,616.39  | 71.598          | 350,888.20 | 1,783.59                    |
  | 8    | 16384 | 1024 | 188,009.84 | 116.412         | 307,462.02 | 430.44                      |
  | 8    | 16384 | 4096 | 454,360.32 | 83.322          | 795,845.18 | 649.39                      |
  | 64   | 1024  | 1024 | 5,041.42   | 41.118          | 47,115.19  | 4,213.76                    |
  | 64   | 1024  | 4096 | 5,041.42   | 52.394          | 219,607.04 | 3,705.20                    |
  | 64   | 4096  | 1024 | 25,471.93  | 92.624          | 120,316.16 | 1,681.39                    |
  | 64   | 4096  | 4096 | 68,028.84  | 71.598          | 361,300.65 | 1,783.59                    |
  | 64   | 16384 | 1024 | 198,422.30 | 116.412         | 317,874.48 | 430.44                      |
  | 64   | 16384 | 4096 | 464,772.77 | 83.322          | 806,257.64 | 649.39                      |

- **分析：**

  - 随 RR 增大，TTFT 和 E2E 单调上升；IL 增大时 TTFT/E2E 上升且吞吐下降，符合计算量与排队压力增加的规律。RR=8 到 64 时，IL>=4096 的 TPOT 和吞吐完全相同，说明系统在 RR=8 已进入饱和区，继续提高到 RR=64 主要增加排队时间。

  - OL 从 1024 增至 4096 后 E2E 均显著增加。TPOT 的方向依赖负载：在 RR>=8 且 IL>=4096，或 IL=16384 的组合中 TPOT 多数降低，可由长解码阶段 continuous batching 利用率提高解释；但 RR=1、IL=4096 时 TPOT 明显上升，因此不能概括为所有 `IL>=4096` 均降低。

#### TP=4、DP=1

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": false
      
  "scheduler": {
      "tp_size": 4
  ```

- server

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel
  ```

- client

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  export HISIM_SEED=1
  
  mkdir -p "$HOME/project/cases_same_seed"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=$HISIM_SEED"
  
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
          --seed "$HISIM_SEED" \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

- 结果：

  - seed=1

  | RR   | IL    | OL   | TTFT (ms)  | TPOT (ms/token) | E2E (ms)   | Output throughput (token/s) |
  | ---- | ----- | ---- | ---------- | --------------- | ---------- | --------------------------- |
  | 1    | 1024  | 1024 | 118.97     | 8.470           | 8,784.11   | 1,020.92                    |
  | 1    | 1024  | 4096 | 122.92     | 18.951          | 77,725.38  | 3,185.33                    |
  | 1    | 4096  | 1024 | 510.44     | 15.484          | 16,353.79  | 1,014.33                    |
  | 1    | 4096  | 4096 | 516.63     | 45.318          | 186,103.06 | 2,400.47                    |
  | 1    | 16384 | 1024 | 87,962.08  | 168.297         | 260,642.96 | 462.59                      |
  | 1    | 16384 | 4096 | 183,807.98 | 92.558          | 563,117.89 | 961.15                      |
  | 8    | 1024  | 1024 | 313.16     | 37.982          | 39,170.99  | 3,824.92                    |
  | 8    | 1024  | 4096 | 313.16     | 41.535          | 170,400.23 | 4,409.99                    |
  | 8    | 4096  | 1024 | 25,530.59  | 84.833          | 112,397.20 | 1,647.66                    |
  | 8    | 4096  | 4096 | 25,530.59  | 66.986          | 299,901.72 | 2,627.31                    |
  | 8    | 16384 | 1024 | 171,261.71 | 168.297         | 343,942.59 | 462.59                      |
  | 8    | 16384 | 4096 | 267,107.61 | 92.558          | 646,417.51 | 961.15                      |
  | 64   | 1024  | 1024 | 7,844.56   | 40.258          | 49,037.88  | 4,053.41                    |
  | 64   | 1024  | 4096 | 7,844.56   | 42.366          | 181,342.10 | 4,480.68                    |
  | 64   | 4096  | 1024 | 35,943.04  | 84.833          | 122,809.65 | 1,647.66                    |
  | 64   | 4096  | 4096 | 35,943.04  | 66.986          | 310,314.17 | 2,627.31                    |
  | 64   | 16384 | 1024 | 181,674.16 | 168.297         | 354,355.04 | 462.59                      |
  | 64   | 16384 | 4096 | 277,520.06 | 92.558          | 656,829.97 | 961.15                      |

- **分析：**

  - RR、IL、OL 的总体变化方向与 TP2 一致：负载越高，TTFT/E2E 越大；长输入降低吞吐；RR=8 后多个长输入组合已经饱和。
  - **扩展性异常：**TP4 相对 TP2 仅在 8/18 组改善 TTFT、11/18 组改善 TPOT 和 E2E；吞吐在 15/18 组提高。低负载短输入下 TP4 的 TTFT 反而增加约 29%~49%，可能来自 TP 通信开销。

  - **需重点复核：**`IL=16384, OL=1024` 时 TP4 的 TPOT 为 168.297 ms/token，而 TP2 约为 116 ms/token，TP 扩容后明显回退；`RR=8/64, IL=4096, OL=1024` 的吞吐也下降约 2%，`RR=64, IL=1024, OL=1024` 下降约 3.8%。这说明当前 predictor/调度模型中的 TP 扩展并非单调。

### 6.3 PD 分离

#### P1D1、TP=1、DP=1

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": true,
      
  "disagg": {
      "prefill": {
          "tp_size": 1
  
  "disagg": {
      "decode": {
          "tp_size": 1
  ```

- server

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel
  ```

- client

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  export HISIM_SEED=1
  
  mkdir -p "$HOME/project/cases_same_seed"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=$HISIM_SEED"
  
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
          --seed "$HISIM_SEED" \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

- 结果：

  - seed=1

  | RR   | IL    | OL   | TTFT (ms)  | TPOT (ms/token) | E2E (ms)     | Output throughput (token/s) |
  | ---- | ----- | ---- | ---------- | --------------- | ------------ | --------------------------- |
  | 1    | 1024  | 1024 | 63.81      | 15.003          | 15,412.35    | 991.78                      |
  | 1    | 1024  | 4096 | 25,811.31  | 41.990          | 197,759.14   | 1,895.43                    |
  | 1    | 4096  | 1024 | 253.05     | 37.439          | 38,553.24    | 926.04                      |
  | 1    | 4096  | 4096 | 175,116.86 | 57.816          | 411,875.11   | 1,014.28                    |
  | 1    | 16384 | 1024 | 187,517.58 | 80.866          | 270,243.77   | 322.20                      |
  | 1    | 16384 | 4096 | 890,253.26 | 65.787          | 1,159,649.93 | 366.14                      |
  | 8    | 1024  | 1024 | 93.95      | 45.040          | 46,169.89    | 3,072.04                    |
  | 8    | 1024  | 4096 | 84,272.13  | 45.824          | 271,921.16   | 2,113.01                    |
  | 8    | 4096  | 1024 | 37,532.49  | 69.389          | 108,517.56   | 1,256.95                    |
  | 8    | 4096  | 4096 | 255,799.66 | 57.218          | 490,108.48   | 1,015.22                    |
  | 8    | 16384 | 1024 | 270,577.23 | 77.069          | 349,418.69   | 322.36                      |
  | 8    | 16384 | 4096 | 975,183.37 | 64.912          | 1,240,996.84 | 357.73                      |
  | 64   | 1024  | 1024 | 2,902.41   | 50.766          | 54,836.53    | 3,554.52                    |
  | 64   | 1024  | 4096 | 93,571.87  | 45.749          | 280,915.63   | 2,145.74                    |
  | 64   | 4096  | 1024 | 47,944.94  | 69.389          | 118,930.01   | 1,256.95                    |
  | 64   | 4096  | 4096 | 266,212.11 | 57.218          | 500,520.94   | 1,015.22                    |
  | 64   | 16384 | 1024 | 280,989.68 | 77.069          | 359,831.14   | 322.36                      |
  | 64   | 16384 | 4096 | 985,595.83 | 64.912          | 1,251,409.30 | 357.73                      |

- **分析：**RR 增大时 TTFT/E2E 单调增加；IL 增大时 TTFT/E2E 增加、吞吐下降，符合排队规律。RR=8 到 64 时，IL>=4096 的 TPOT 与吞吐基本不再增长，说明 P1D1 已饱和。
- P1D1 是 PD 分离基线，长输入、长输出下 TTFT 最高可接近 986 秒，E2E 超过 1,251 秒。KV transfer 仅为毫秒量级，不是主要瓶颈，主要成本来自 Prefill/Decode 排队。
  - **数据异常：**除 `RR=1, IL=1024, OL=1024` 外，其余 17 组各混入一条 `1x1` 特殊请求。

#### P2D2、P_TP=2、D_TP=2

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": true,
      
  "disagg": {
      "prefill": {
          "tp_size": 2
  
  "disagg": {
      "decode": {
          "tp_size": 2
  ```

- server

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel
  ```

- client

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  export HISIM_SEED=1
  
  mkdir -p "$HOME/project/cases_same_seed"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=$HISIM_SEED"
  
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
          --seed "$HISIM_SEED" \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

- 结果：

  - seed=1

  | RR   | IL    | OL   | TTFT (ms)  | TPOT (ms/token) | E2E (ms)   | Output throughput (token/s) |
  | ---- | ----- | ---- | ---------- | --------------- | ---------- | --------------------------- |
  | 1    | 1024  | 1024 | 87.22      | 9.487           | 9,792.82   | 1,014.72                    |
  | 1    | 1024  | 4096 | 90.00      | 27.143          | 111,238.96 | 2,747.28                    |
  | 1    | 4096  | 1024 | 365.80     | 17.311          | 18,075.10  | 994.48                      |
  | 1    | 4096  | 4096 | 19,375.86  | 57.636          | 255,396.19 | 1,789.59                    |
  | 1    | 16384 | 1024 | 91,161.70  | 99.817          | 193,274.21 | 454.73                      |
  | 1    | 16384 | 4096 | 373,889.55 | 70.056          | 660,767.08 | 680.73                      |
  | 8    | 1024  | 1024 | 158.62     | 37.042          | 38,052.57  | 3,648.80                    |
  | 8    | 1024  | 4096 | 158.62     | 50.686          | 207,717.20 | 3,562.27                    |
  | 8    | 4096  | 1024 | 15,076.28  | 92.251          | 109,449.51 | 1,665.46                    |
  | 8    | 4096  | 4096 | 88,459.82  | 58.696          | 328,818.78 | 1,854.78                    |
  | 8    | 16384 | 1024 | 174,818.26 | 99.576          | 276,684.62 | 454.59                      |
  | 8    | 16384 | 4096 | 457,189.18 | 70.056          | 744,066.71 | 680.73                      |
  | 64   | 1024  | 1024 | 5,117.96   | 41.068          | 47,131.00  | 4,106.46                    |
  | 64   | 1024  | 4096 | 5,117.96   | 52.374          | 219,590.99 | 3,670.26                    |
  | 64   | 4096  | 1024 | 25,488.74  | 92.251          | 119,861.97 | 1,665.46                    |
  | 64   | 4096  | 4096 | 98,872.28  | 58.696          | 339,231.23 | 1,854.78                    |
  | 64   | 16384 | 1024 | 185,230.72 | 99.576          | 287,097.07 | 454.59                      |
  | 64   | 16384 | 4096 | 467,601.63 | 70.056          | 754,479.17 | 680.73                      |

- **分析：**相对 P1D1，P2D2 在 14/18 组改善 TTFT、16/18 组改善 E2E，并在 18/18 组提高吞吐；按每个 workload 的相对变化简单平均，E2E 下降约 28%，吞吐提高约 51%。双侧扩容总体有效。

- **TPOT 并非全面改善：**P2D2 仅在 6/18 组低于 P1D1，说明 TP2 的通信/预测开销会抵消部分单 token 收益；但 TTFT、E2E 与吞吐的总体容量收益仍明显。

- **数据异常：**除 `RR=1, IL=1024, OL=1024` 外，其余 17 组各混入一条 `1x1` 特殊请求。磁盘上的实际目录名为 `cases_same_seed_1-P2D2-Ptp2-Dtp2-dp1-PDdisagg`。



#### P4D2、P_TP=4、D_TP=2

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": true,
      
  "disagg": {
      "prefill": {
          "tp_size": 4
  
  "disagg": {
      "decode": {
          "tp_size": 2
  ```

- server

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel
  ```

- client

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  export HISIM_SEED=1
  
  mkdir -p "$HOME/project/cases_same_seed"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=$HISIM_SEED"
  
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
          --seed "$HISIM_SEED" \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

- 结果：

  - seed=1

  | RR   | IL    | OL   | TTFT (ms)  | TPOT (ms/token) | E2E (ms)   | Output throughput (token/s) |
  | ---- | ----- | ---- | ---------- | --------------- | ---------- | --------------------------- |
  | 1    | 1024  | 1024 | 119.68     | 9.807           | 10,152.16  | 1,014.51                    |
  | 1    | 1024  | 4096 | 122.44     | 28.238          | 115,755.89 | 2,720.48                    |
  | 1    | 4096  | 1024 | 563.23     | 23.268          | 24,366.37  | 984.30                      |
  | 1    | 4096  | 4096 | 27,094.31  | 60.208          | 273,647.11 | 1,721.71                    |
  | 1    | 16384 | 1024 | 133,950.14 | 112.696         | 249,238.60 | 382.51                      |
  | 1    | 16384 | 4096 | 416,033.41 | 72.897          | 714,546.20 | 635.62                      |
  | 8    | 1024  | 1024 | 322.45     | 41.630          | 42,909.68  | 3,483.73                    |
  | 8    | 1024  | 4096 | 322.45     | 52.150          | 213,875.37 | 3,524.85                    |
  | 8    | 4096  | 1024 | 25,844.90  | 102.355         | 130,554.15 | 1,420.45                    |
  | 8    | 4096  | 4096 | 99,218.55  | 60.398          | 346,548.27 | 1,765.08                    |
  | 8    | 16384 | 1024 | 216,962.13 | 112.186         | 331,728.25 | 382.61                      |
  | 8    | 16384 | 4096 | 499,333.04 | 72.897          | 797,845.82 | 635.62                      |
  | 64   | 1024  | 1024 | 7,940.15   | 43.647          | 52,591.18  | 3,699.41                    |
  | 64   | 1024  | 4096 | 7,940.15   | 53.019          | 225,051.17 | 3,582.19                    |
  | 64   | 4096  | 1024 | 36,257.35  | 102.355         | 140,966.60 | 1,420.45                    |
  | 64   | 4096  | 4096 | 109,631.00 | 60.398          | 356,960.73 | 1,765.08                    |
  | 64   | 16384 | 1024 | 227,374.59 | 112.186         | 342,140.71 | 382.61                      |
  | 64   | 16384 | 4096 | 509,745.49 | 72.897          | 808,258.28 | 635.62                      |

- **分析：**单看 RR/IL/OL 变化，排队和饱和规律正常；但横向对比 P2D2 后出现明显反常。
- **高优先级异常：**P_TP 从 2 增至 4、Decode 保持 TP2 后，P4D2 在 18/18 个相同 workload 中 TTFT、TPOT、E2E 全部变差，吞吐也全部下降。相对 P2D2，按 workload 相对变化简单平均，TTFT 增加约 41%、TPOT 增加约 8%、E2E 增加约 12%、吞吐下降约 7%。
- 该结果不能用普通负载波动解释，因为模拟使用相同 seed 且回退覆盖全部 18 组。建议优先检查 AIC 中 RTX Pro 6000 Server 的 TP4 Prefill 预测曲线、TP 通信开销和 Prefill scheduler 的并行度映射。
- **数据异常：**除 `RR=1, IL=1024, OL=1024` 外，其余 17 组各混入一条 `1x1` 特殊请求。

#### P2D4、P_TP=2、D_TP=4

- json：`pd_disagg_rtx6000_sglang_0_5_10.json`

  ```json
  "disagg": {
      "enabled": true,
      
  "disagg": {
      "prefill": {
          "tp_size": 2
  
  "disagg": {
      "decode": {
          "tp_size": 4
  ```

- server

  ```bash
  # server
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --chunked-prefill-size 4096 \
    --skip-server-warmup \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel
  ```

- client

  ```bash
  # client
  export HF_ENDPOINT=https://hf-mirror.com
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30004
  export HISIM_SEED=1
  
  mkdir -p "$HOME/project/cases_same_seed"
  mkdir -p "$HISIM_OUTPUT_DIR"
  
  for rr in 1 8 64; do
    for il in 1024 4096 16384; do
      for ol in 1024 4096; do
        echo "RUN rr=$rr il=$il ol=$ol seed=$HISIM_SEED"
  
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
          --seed "$HISIM_SEED" \
          --flush-cache \
          --output-file "$HOME/project/cases_same_seed/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

- 结果：

  - seed=1

  | RR   | IL    | OL   | TTFT (ms)  | TPOT (ms/token) | E2E (ms)   | Output throughput (token/s) |
  | ---- | ----- | ---- | ---------- | --------------- | ---------- | --------------------------- |
  | 1    | 1024  | 1024 | 86.42      | 8.182           | 8,456.35   | 1,021.07                    |
  | 1    | 1024  | 4096 | 90.37      | 18.141          | 74,377.34  | 3,209.20                    |
  | 1    | 4096  | 1024 | 364.68     | 12.319          | 12,967.51  | 1,005.68                    |
  | 1    | 4096  | 4096 | 375.30     | 39.466          | 161,986.59 | 2,488.21                    |
  | 1    | 16384 | 1024 | 58,260.62  | 71.654          | 131,563.01 | 574.43                      |
  | 1    | 16384 | 4096 | 219,489.12 | 46.103          | 408,281.06 | 1,023.36                    |
  | 8    | 1024  | 1024 | 160.94     | 33.234          | 34,159.11  | 3,948.39                    |
  | 8    | 1024  | 4096 | 160.94     | 40.115          | 164,431.69 | 4,437.14                    |
  | 8    | 4096  | 1024 | 15,076.28  | 74.415          | 91,203.22  | 1,957.36                    |
  | 8    | 4096  | 4096 | 56,526.99  | 44.463          | 238,603.39 | 2,530.02                    |
  | 8    | 16384 | 1024 | 141,761.69 | 71.436          | 214,840.29 | 574.18                      |
  | 8    | 16384 | 4096 | 302,568.75 | 46.157          | 491,580.69 | 1,023.36                    |
  | 64   | 1024  | 1024 | 5,117.96   | 37.602          | 43,584.86  | 4,422.50                    |
  | 64   | 1024  | 4096 | 5,117.96   | 41.698          | 175,871.51 | 4,569.90                    |
  | 64   | 4096  | 1024 | 25,488.74  | 74.415          | 101,615.67 | 1,957.36                    |
  | 64   | 4096  | 4096 | 66,939.45  | 44.463          | 249,015.84 | 2,530.02                    |
  | 64   | 16384 | 1024 | 152,174.14 | 71.436          | 225,252.74 | 574.18                      |
  | 64   | 16384 | 4096 | 312,981.20 | 46.157          | 501,993.14 | 1,023.36                    |

- **分析：**相对 P2D2，仅将 Decode 从 TP2 扩至 TP4 后，18/18 组 TPOT 和 E2E 均改善，18/18 组吞吐均提高，11/18 组 TTFT 也改善。按 workload 相对变化简单平均，TTFT 下降约 19%、TPOT 下降约 25%、E2E 下降约 24%、吞吐提高约 26%，Decode 扩容效果稳定。
- 在相同 6 卡预算下，P2D4 相对 P4D2 的 18/18 组 TTFT、TPOT、E2E 都更低，吞吐全部更高；简单平均改善约为 TTFT 42%、TPOT 30%、E2E 32%、吞吐 36%。当前 workload 明显更适合把额外资源分配给 Decode。
- **数据异常：**`RR=1, IL=1024` 的两个 OL 组合长度完整，其余 16 组各混入一条 `1x1` 特殊请求。

### 6.4 PD 分离：Replica 扩展

Replica 实验固定 P/D 两侧 `tp_size=1`，通过 `replicas` 增加独立实例。Decode 副本数大于 1 时使用 `decode_queue_mode=per_replica_queue`。P3D1 的 Decode 仅有一个副本，因此单队列配置不会改变其行为。

#### Replica-P3D1

| RR | IL | OL | TTFT (ms) | TPOT (ms/token) | E2E (ms) | Output throughput (token/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 1024 | 61.90 | 14.988 | 15,394.68 | 991.79 |
| 1 | 1024 | 4096 | 25,715.84 | 41.944 | 197,477.19 | 1,897.03 |
| 1 | 4096 | 1024 | 241.75 | 38.027 | 39,143.69 | 925.79 |
| 1 | 4096 | 4096 | 175,861.48 | 57.941 | 413,131.20 | 1,012.24 |
| 1 | 16384 | 1024 | 306,004.14 | 92.861 | 401,000.94 | 235.42 |
| 1 | 16384 | 4096 | 1,010,610.26 | 68.308 | 1,290,331.19 | 324.49 |
| 8 | 1024 | 1024 | 67.46 | 43.655 | 44,726.25 | 3,108.05 |
| 8 | 1024 | 4096 | 83,567.83 | 45.557 | 270,124.82 | 2,122.62 |
| 8 | 4096 | 1024 | 35,204.68 | 67.869 | 104,634.93 | 1,295.76 |
| 8 | 4096 | 4096 | 254,384.30 | 57.072 | 488,093.14 | 1,018.76 |
| 8 | 16384 | 1024 | 389,303.76 | 92.861 | 484,300.57 | 235.42 |
| 8 | 16384 | 4096 | 1,093,909.89 | 68.308 | 1,373,630.82 | 324.49 |
| 64 | 1024 | 1024 | 1,002.25 | 49.062 | 51,192.43 | 3,795.80 |
| 64 | 1024 | 4096 | 91,676.08 | 45.539 | 278,157.19 | 2,166.69 |
| 64 | 4096 | 1024 | 45,617.13 | 67.869 | 115,047.38 | 1,295.76 |
| 64 | 4096 | 4096 | 264,796.75 | 57.072 | 498,505.59 | 1,018.76 |
| 64 | 16384 | 1024 | 399,716.22 | 92.861 | 494,713.02 | 235.42 |
| 64 | 16384 | 4096 | 1,104,322.35 | 68.308 | 1,384,043.27 | 324.49 |

相对 P1D1，P3D1 在 11/18 组降低 TTFT，但按 workload 相对变化简单平均，TTFT、TPOT、E2E 分别增加 3.9%、3.3%、7.8%，吞吐下降 5.3%。回退集中在长输入，最大 Prefill queue 达 1,103 秒。仅增加 Prefill replicas 没有缓解该瓶颈，属于明显的反向扩展。

#### Replica-P2D2

| RR | IL | OL | TTFT (ms) | TPOT (ms/token) | E2E (ms) | Output throughput (token/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 1024 | 61.65 | 13.532 | 13,904.80 | 994.67 |
| 1 | 1024 | 4096 | 61.35 | 28.138 | 115,287.96 | 2,710.74 |
| 1 | 4096 | 1024 | 235.49 | 21.556 | 22,287.08 | 971.40 |
| 1 | 4096 | 4096 | 24,865.15 | 51.975 | 237,700.47 | 1,821.74 |
| 1 | 16384 | 1024 | 244,859.48 | 104.409 | 351,656.20 | 255.98 |
| 1 | 16384 | 4096 | 778,311.98 | 71.023 | 1,069,142.77 | 370.71 |
| 8 | 1024 | 1024 | 70.30 | 29.521 | 30,270.02 | 4,039.50 |
| 8 | 1024 | 4096 | 70.30 | 44.010 | 180,290.33 | 3,974.00 |
| 8 | 4096 | 1024 | 4,445.73 | 76.841 | 83,023.22 | 2,083.95 |
| 8 | 4096 | 4096 | 89,939.46 | 58.971 | 331,407.16 | 1,451.24 |
| 8 | 16384 | 1024 | 328,159.11 | 104.409 | 434,955.83 | 255.98 |
| 8 | 16384 | 4096 | 861,611.61 | 71.023 | 1,152,442.40 | 370.71 |
| 64 | 1024 | 1024 | 1,636.52 | 32.264 | 34,642.43 | 5,439.66 |
| 64 | 1024 | 4096 | 1,636.52 | 45.614 | 188,424.93 | 4,218.48 |
| 64 | 4096 | 1024 | 14,858.18 | 76.841 | 93,435.68 | 2,083.95 |
| 64 | 4096 | 4096 | 100,351.91 | 58.971 | 341,819.62 | 1,451.24 |
| 64 | 16384 | 1024 | 338,571.56 | 104.409 | 445,368.29 | 255.98 |
| 64 | 16384 | 4096 | 872,024.06 | 71.023 | 1,162,854.85 | 370.71 |

相对 P1D1，Replica-P2D2 在 15/18 组改善 TTFT、15/18 组改善 E2E 和吞吐；平均相对变化为 TTFT 降低 39.5%、E2E 降低 18.1%、吞吐提高 31.2%，双侧 Replica 扩容总体有效。但 `IL=16384, OL=1024` 的三档 RR 均回退，吞吐下降约 20.6%，说明扩容收益并不覆盖全部长输入场景。

#### Replica-P4D2

| RR | IL | OL | TTFT (ms) | TPOT (ms/token) | E2E (ms) | Output throughput (token/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 1024 | 61.65 | 13.532 | 13,904.80 | 994.67 |
| 1 | 1024 | 4096 | 61.35 | 28.138 | 115,287.96 | 2,710.74 |
| 1 | 4096 | 1024 | 234.35 | 21.607 | 22,337.95 | 973.62 |
| 1 | 4096 | 4096 | 24,974.07 | 52.038 | 238,069.73 | 1,820.13 |
| 1 | 16384 | 1024 | 244,871.12 | 104.415 | 351,674.02 | 255.96 |
| 1 | 16384 | 4096 | 778,323.63 | 71.024 | 1,069,158.98 | 370.70 |
| 8 | 1024 | 1024 | 64.99 | 29.298 | 30,036.80 | 4,049.54 |
| 8 | 1024 | 4096 | 64.99 | 43.940 | 180,000.92 | 3,974.61 |
| 8 | 4096 | 1024 | 4,444.88 | 76.827 | 83,007.98 | 2,084.27 |
| 8 | 4096 | 4096 | 89,953.54 | 58.968 | 331,409.36 | 1,451.08 |
| 8 | 16384 | 1024 | 328,170.75 | 104.415 | 434,973.65 | 255.96 |
| 8 | 16384 | 4096 | 861,623.25 | 71.024 | 1,152,458.61 | 370.70 |
| 64 | 1024 | 1024 | 678.27 | 31.379 | 32,778.64 | 5,724.46 |
| 64 | 1024 | 4096 | 678.27 | 45.393 | 186,561.14 | 4,259.57 |
| 64 | 4096 | 1024 | 14,857.34 | 76.827 | 93,420.44 | 2,084.27 |
| 64 | 4096 | 4096 | 100,366.00 | 58.968 | 341,821.81 | 1,451.08 |
| 64 | 16384 | 1024 | 338,583.20 | 104.415 | 445,386.10 | 255.96 |
| 64 | 16384 | 4096 | 872,035.71 | 71.024 | 1,162,871.06 | 370.70 |

Replica-P4D2 与 Replica-P2D2 几乎重合：平均 E2E 仅降低 0.4%、吞吐仅提高 0.4%，改善主要集中在 `RR=64, IL=1024`。对于 `IL>=4096`，额外两个 Prefill replicas 基本没有收益，Prefill queue 最大值仍约 871 秒。这不是理想的 Prefill 水平扩展结果。

#### Replica-P2D4

| RR | IL | OL | TTFT (ms) | TPOT (ms/token) | E2E (ms) | Output throughput (token/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 1024 | 61.09 | 12.710 | 13,062.93 | 997.41 |
| 1 | 1024 | 4096 | 58.01 | 17.754 | 72,760.02 | 3,145.80 |
| 1 | 4096 | 1024 | 233.30 | 16.473 | 17,084.68 | 982.89 |
| 1 | 4096 | 4096 | 236.88 | 33.324 | 136,698.05 | 2,628.19 |
| 1 | 16384 | 1024 | 152,317.01 | 149.710 | 305,398.04 | 314.49 |
| 1 | 16384 | 4096 | 408,116.49 | 79.200 | 732,409.15 | 493.04 |
| 8 | 1024 | 1024 | 70.41 | 22.378 | 22,962.71 | 4,739.87 |
| 8 | 1024 | 4096 | 70.41 | 29.445 | 120,647.34 | 5,630.54 |
| 8 | 4096 | 1024 | 4,445.73 | 62.040 | 67,895.65 | 2,083.95 |
| 8 | 4096 | 4096 | 4,445.73 | 52.033 | 217,490.54 | 3,005.36 |
| 8 | 16384 | 1024 | 235,616.64 | 149.710 | 388,697.67 | 314.49 |
| 8 | 16384 | 4096 | 491,416.12 | 79.200 | 815,708.78 | 493.04 |
| 64 | 1024 | 1024 | 1,636.52 | 24.217 | 26,410.51 | 6,916.40 |
| 64 | 1024 | 4096 | 1,636.52 | 30.639 | 127,101.67 | 6,087.21 |
| 64 | 4096 | 1024 | 14,858.18 | 62.040 | 78,308.11 | 2,083.95 |
| 64 | 4096 | 4096 | 14,858.18 | 52.033 | 227,903.00 | 3,005.36 |
| 64 | 16384 | 1024 | 246,029.09 | 149.710 | 399,110.12 | 314.49 |
| 64 | 16384 | 4096 | 501,828.57 | 79.200 | 826,121.23 | 493.04 |

相对 Replica-P2D2，仅增加 Decode replicas 后，18/18 组 E2E 改善，16/18 组吞吐提高；平均 TTFT、TPOT、E2E 分别降低 28.5%、6.4%、24.9%，吞吐提高 31.9%。相同 6 设备预算下，Replica-P2D4 的 18/18 组 E2E 均优于 Replica-P4D2，表明当前 workload 更需要 Decode 容量。

#### Replica-P8D8

| RR | IL | OL | TTFT (ms) | TPOT (ms/token) | E2E (ms) | Output throughput (token/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 1024 | 60.99 | 12.328 | 12,672.70 | 997.86 |
| 1 | 1024 | 4096 | 56.80 | 14.342 | 58,785.91 | 3,267.28 |
| 1 | 4096 | 1024 | 233.22 | 14.742 | 15,314.04 | 983.69 |
| 1 | 4096 | 4096 | 237.20 | 22.274 | 91,450.82 | 3,023.74 |
| 1 | 16384 | 1024 | 101,503.93 | 241.462 | 348,343.23 | 440.79 |
| 1 | 16384 | 4096 | 101,503.93 | 108.670 | 546,417.26 | 1,239.00 |
| 8 | 1024 | 1024 | 63.15 | 18.171 | 18,652.27 | 5,170.68 |
| 8 | 1024 | 4096 | 63.28 | 20.321 | 83,277.76 | 7,637.96 |
| 8 | 4096 | 1024 | 4,444.88 | 57.551 | 63,306.91 | 2,084.27 |
| 8 | 4096 | 4096 | 4,444.88 | 39.476 | 166,079.80 | 3,005.53 |
| 8 | 16384 | 1024 | 184,803.56 | 241.462 | 431,642.86 | 440.79 |
| 8 | 16384 | 4096 | 184,803.56 | 108.670 | 629,716.89 | 1,239.00 |
| 64 | 1024 | 1024 | 100.01 | 16.823 | 17,309.95 | 9,895.35 |
| 64 | 1024 | 4096 | 100.01 | 20.618 | 84,529.70 | 8,530.67 |
| 64 | 4096 | 1024 | 14,857.34 | 57.551 | 73,719.36 | 2,084.27 |
| 64 | 4096 | 4096 | 14,857.34 | 39.476 | 176,492.26 | 3,005.53 |
| 64 | 16384 | 1024 | 195,216.01 | 241.462 | 442,055.31 | 440.79 |
| 64 | 16384 | 4096 | 195,216.01 | 108.670 | 640,129.34 | 1,239.00 |

相对 Replica-P2D4，P8D8 在 18/18 组提高吞吐，平均提高 40.1%，TTFT 和 E2E 分别平均降低 26.9% 和 15.4%。但 `IL=16384` 的 TPOT 增加 37.2% 至 61.3%，`OL=1024` 的 E2E 在三档 RR 下反而增加约 11% 至 14%。最大 Prefill queue 降至 194 秒，而最大 Decode queue 升至 186 秒，显示瓶颈从 Prefill 向 Decode 迁移，并伴随单副本 batch 变小或负载分配效率下降。

### 6.5 带宽敏感性实验

#### Host Memory Bandwidth：PD 合并 TP4

该实验将 `memory_read_bandwidth_gb` 和 `memory_write_bandwidth_gb` 同时从 480 降至 64，拓扑保持 TP4、DP1、PD 合并不变。

| RR | IL | OL | TTFT (ms) | TPOT (ms/token) | E2E (ms) | Output throughput (token/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 1024 | 119.603 | 8.470 | 8,784.115 | 1,020.916 |
| 1 | 1024 | 4096 | 123.563 | 18.950 | 77,725.384 | 3,185.330 |
| 1 | 4096 | 1024 | 512.761 | 15.482 | 16,353.793 | 1,014.326 |
| 1 | 4096 | 4096 | 518.943 | 45.318 | 186,103.058 | 2,400.474 |
| 1 | 16384 | 1024 | 87,962.174 | 168.297 | 260,642.962 | 462.591 |
| 1 | 16384 | 4096 | 183,808.079 | 92.558 | 563,117.886 | 961.146 |
| 8 | 1024 | 1024 | 314.569 | 37.981 | 39,170.987 | 3,824.922 |
| 8 | 1024 | 4096 | 314.569 | 41.534 | 170,400.227 | 4,409.989 |
| 8 | 4096 | 1024 | 25,533.003 | 84.831 | 112,397.196 | 1,647.665 |
| 8 | 4096 | 4096 | 25,533.003 | 66.985 | 299,901.717 | 2,627.311 |
| 8 | 16384 | 1024 | 171,261.801 | 168.297 | 343,942.589 | 462.591 |
| 8 | 16384 | 4096 | 267,107.707 | 92.558 | 646,417.513 | 961.146 |
| 64 | 1024 | 1024 | 7,846.947 | 40.255 | 49,037.885 | 4,053.409 |
| 64 | 1024 | 4096 | 7,846.947 | 42.365 | 181,342.102 | 4,480.675 |
| 64 | 4096 | 1024 | 35,945.456 | 84.831 | 122,809.650 | 1,647.665 |
| 64 | 4096 | 4096 | 35,945.456 | 66.985 | 310,314.171 | 2,627.311 |
| 64 | 16384 | 1024 | 181,674.255 | 168.297 | 354,355.043 | 462.591 |
| 64 | 16384 | 4096 | 277,520.160 | 92.558 | 656,829.967 | 961.146 |

相对 480 GB/s 基线，E2E 和吞吐 18/18 组不变；TTFT 平均增加 0.164%，绝对增加 0.096--2.416 ms，TPOT 平均降低 0.0024%。这组微小的 TTFT/TPOT 反向变化在 E2E 中完全抵消，不代表 Decode 因带宽降低而加速。

| L2 直接指标 | 480 GB/s | 64 GB/s |
| --- | ---: | ---: |
| 总 iteration | 121,113 | 121,113 |
| 非零 L2 load iteration | 27 | 27 |
| 累计 L2 load | 79.14 ms | 359.46 ms |
| 最大单次 L2 load | 3.53 ms | 15.56 ms |
| 累计 L2 backup | 10.124 s | 25.244 s |
| 累计模型 forward | 6,230.93 s | 6,230.93 s |
| 重叠后暴露的 L2 load | 0 ms | 0 ms |

参数已经进入 L2 I/O 模型，但 H2D load 只占 0.0223% 的 iteration，并且只出现在 `IL=16384` 的 6 个 workload。发生 load 时，上一轮 forward 为 374--454 ms，显著大于 64 GB/s 下最大 15.56 ms 的 load，因此全部被 overlap schedule 隐藏。与此同时，本实验也降低了 write bandwidth，累计 D2H backup 增加 15.12 s；该变量混杂使毫秒级 TTFT 差异不能单独归因于 read bandwidth。

#### Host Memory Bandwidth：PD 分离 Replica-P2D2

Replica-P2D2 将 Host read/write 从 480 同时降至 64 GB/s 后，TTFT、TPOT、E2E 和吞吐在 18/18 个 workload 中逐项完全相同，整体均值仍为 TTFT 203,431.68 ms、TPOT 59.141 ms/token、E2E 349,384.11 ms、吞吐 1,840.03 token/s。

直接指标显示两组均有 197,501 次 iteration 和 3,209 次 L2 backup，但 **L2 load 为 0 次**。累计 backup 从 19.779 s 增至 80.072 s，模型 forward 均为 11,948.42 s。也就是说，write-through 写入发生且 write bandwidth 参数改变了直接受控时长，但当前 PD 分离请求计时没有体现这部分差异；read bandwidth 则因为没有 Host L2→GPU load-back 而没有可作用对象。

#### KV Transfer Bandwidth：PD 分离 Replica-P2D2

该实验仅将 `disagg.kv_transfer.bw_gbps` 从 811 降至 128，Replica-P2D2 拓扑和 Host bandwidth 保持不变。

| IL | 811 GB/s 典型 KV transfer | 128 GB/s 典型 KV transfer | 放大倍率 |
| ---: | ---: | ---: | ---: |
| 1024 | 0.208--0.400 ms | 1.211--2.436 ms | 5.82--6.09 倍 |
| 4096 | 0.765 ms | 4.737 ms | 6.20 倍 |
| 16384 | 2.998 ms | 18.889 ms | 6.30 倍 |

理论带宽比为 $811/128=6.336$。实测倍率略低，是因为传输模型还包含固定 20 μs 延迟；结果证明参数已经进入 Prefill→Decode KV transfer 路径。

| 指标 | 811 GB/s | 128 GB/s | 逐 workload 平均相对变化 |
| --- | ---: | ---: | ---: |
| TTFT (ms) | 203,431.68 | 203,438.73 | +0.42685% |
| TPOT (ms/token) | 59.141 | 59.138 | -0.00787% |
| E2E (ms) | 349,384.11 | 349,388.11 | -0.00425% |
| Output throughput (token/s) | 1,840.033 | 1,840.035 | -0.00115% |

TTFT 的逐 workload 变化范围为 +0.0018% 至 +2.1302%，E2E 为 -0.1447% 至 +0.0397%。低于 0.1% 的局部反向变化来自 batching、队列边界和均值聚合，不应解释为降低带宽带来性能收益。即使在 128 GB/s 下，最大约 18.9 ms 的传输仍远小于高负载下数万至百万毫秒的队列时间，因此它不是当前瓶颈。



## 7. 跨配置分析

### 7.1 十四套配置整体均值

以下为每套配置对 18 个 workload 的简单算术平均，仅用于整体观察，不应替代同一 workload 的逐项比较：

| 配置         | 平均 TTFT (ms) | 平均 TPOT (ms/token) | 平均 E2E (ms) | 平均输出吞吐 (token/s) |
| ------------ | -------------: | -------------------: | ------------: | ---------------------: |
| PD 合并 TP2  |     108,843.07 |               68.220 |    279,255.30 |               1,782.27 |
| PD 合并 TP4  |      72,770.29 |               74.254 |    244,424.79 |               2,067.29 |
| PD 合并 TP4、Host BW64 |      72,771.61 |               74.253 |    244,424.79 |               2,067.29 |
| PD 分离 P1D1 |     254,427.31 |               56.859 |    398,170.54 |               1,239.21 |
| PD 分离 TP-P2D2 |     111,570.01 |               61.321 |    262,262.51 |               1,776.69 |
| PD 分离 TP-P4D2 |     128,820.86 |               66.291 |    287,618.70 |               1,675.38 |
| PD 分离 TP-P2D4 |      75,707.86 |               45.636 |    184,988.00 |               2,159.43 |
| PD 分离 Replica-P3D1 |     293,448.00 |               59.450 |    441,313.80 |               1,240.49 |
| PD 分离 Replica-P2D2 |     203,431.68 |               59.141 |    349,384.11 |               1,840.03 |
| PD 分离 Replica-P2D2、Host BW64 |     203,431.68 |               59.141 |    349,384.11 |               1,840.03 |
| PD 分离 Replica-P2D2、KV BW128 |     203,438.73 |               59.138 |    349,388.11 |               1,840.04 |
| PD 分离 Replica-P4D2 |     203,335.96 |               59.069 |    349,175.55 |               1,858.78 |
| PD 分离 Replica-P2D4 |     115,440.83 |               61.212 |    255,320.57 |               2,429.42 |
| PD 分离 Replica-P8D8 |      55,698.12 |               76.892 |    216,660.91 |               3,040.34 |

> 这些数值是 18 个异质 workload 的简单算术平均。长输入高延迟组合会主导平均值，因此该表只用于总体观察；配置优劣应以相同 workload 的逐项比较为主。

### 7.2 总体判断

1. **负载规律基本正确。**固定配置下，RR 增大会增加排队延迟；IL 增大会增加 Prefill 成本并降低吞吐；OL 增大会显著增加 E2E。RR=8 到 64 时多个 workload 的 TPOT/吞吐完全相同，表示模型进入容量饱和区，而不是 RR=64 没有产生额外压力。
2. **Decode 是当前 workload 的主要扩容方向。**在 TP 与 Replica 两类 6 设备拓扑中，P2D4 都明显优于 P4D2。Replica-P2D4 相对 Replica-P4D2 的 18/18 组 E2E 更低，平均吞吐提高 31.4%。
3. **Prefill Replica 扩展几乎失效。**Replica-P2D2 扩至 Replica-P4D2 后，大多数长输入结果完全或近似重合；P3D1 相对 P1D1 还出现整体回退。额外 Prefill replicas 没有按预期降低 Prefill queue，需要检查副本 admission、affinity、虚拟时钟和 AIC batch 输入。
4. **P8D8 体现吞吐与单请求延迟的权衡。**P8D8 吞吐最高，特别是 `RR=64, IL=1024` 达 8,531 至 9,895 token/s；但长输入 TPOT 显著高于 Replica-P2D4，`IL=16384, OL=1024` 的 E2E 反而恶化。不能仅按总吞吐认定扩容全面成功。
5. **TP 与 Replica 拓扑不等价。**相同 P/D 设备数下，两类拓扑的 batch 形成、通信和排队机制不同。Replica-P2D2/P2D4 的 E2E 简单平均分别比对应 TP 拓扑高约 17%/21%，因此容量规划必须显式标注扩展方式。
6. **Host Memory Bandwidth 不是当前关键路径瓶颈。**TP4 PD 合并的 Host BW64 数据证明 L2 I/O 时长会随参数变化，但 H2D load 极少且被 forward 完全覆盖；Replica-P2D2 更是没有发生 L2 load。当前测试只能说明 workload 对 Host bandwidth 不敏感，不能说明 64 GB/s 与 480 GB/s 等效。
7. **KV 传输不是当前瓶颈。**`bw_gbps` 从 811 降至 128 后，KV transfer 按理论方向放大约 6 倍，但最大仍只有约 18.9 ms，而 Replica 配置的最大 Prefill queue 为 194 至 1,103 秒、最大 Decode queue 为 27 至 186 秒。最终 E2E 与吞吐变化低于 0.005%，说明本组数据主要反映计算与排队能力。

## 8. 异常与风险分级

### 8.1 高优先级

1. **TP-P4D2 扩展全面回退。** 相对 TP-P2D2，增加 Prefill TP 后 18/18 个 workload 的 TTFT、TPOT、E2E 全部变差，吞吐全部下降。建议检查 RTX PRO 6000 Server 的 TP4 Prefill AIC 曲线、TP 通信模型、拓扑映射和 predictor 输入。
2. **Prefill Replica 扩展不生效或反向扩展。** Replica-P2D2 扩至 Replica-P4D2 的平均 E2E/吞吐仅变化约 0.4%；P3D1 相对 P1D1 的平均 E2E 增加 7.8%。该现象覆盖确定性同 seed 数据，不能归因于普通随机波动。
3. **P8D8 长输入 TPOT 反向恶化。** 相对 Replica-P2D4，`IL=16384` 的 TPOT 增加 37.2% 至 61.3%，并使 `OL=1024` 的 E2E 恶化约 11% 至 14%。应检查 Decode replica-local batching、负载均衡和队列时间归属。
4. **特殊 `1x1` 请求污染聚合。** 十一套 PD 分离数据共有 186 个 workload 受影响。请求固定表现为 `input_length=1`、`output_length=1`、`created_time=0`，更像探测、占位或残留请求。它占每组请求数的 0.5%，但对低 RR 下的 mean TTFT/E2E 和吞吐可能产生不成比例的影响。

### 8.2 中优先级

1. **PD 合并 TP4 的 TP 扩展非单调。** TP4 相比 TP2 的吞吐多数提高，但部分短输入和 `IL=16384, OL=1024` 组合回退，需要核验 TP4 Decode 预测曲线及通信开销。
2. **同设备数 TP/Replica 结果不等价。** 这可能来自合理的 TP 通信与 Replica 独立队列差异，但当前差距较大，需用 iteration trace 和 AIC 单步预测拆分验证。
3. **PD 分离的 L2 backup 未反映到最终指标。** Replica-P2D2 的累计 backup 从 19.779 s 增至 80.072 s，但 18/18 组核心指标完全相同。需要核对 PD 请求时钟与 HiCache backup 记账边界，避免 I/O 已计算但未进入请求生命周期。
4. **单 seed。** 本轮只能证明确定性仿真在 seed=1 下的行为，不能提供跨 seed 的置信区间。

### 8.3 解释性现象

1. **RR=8 与 RR=64 的 TPOT/吞吐相同。** 这符合确定性容量模型进入饱和区的表现，不应直接判为数据重复。
2. **长 OL 下 TPOT 有时降低。** 该现象与更稳定的 continuous batching 相容，但需要 iteration 级 batch-size 统计验证。
3. **降低带宽后局部指标略有改善。** KV BW128 中低于 0.1% 的 TPOT、E2E 或吞吐反向变化由 batching、队列边界和聚合口径主导，不构成“低带宽更快”的证据。

## 9. 结论与配置建议

1. 对本轮 Qwen3-8B、固定长度、大输出 workload，优先增加 Decode 资源；在已测试的 6 设备预算中，无论 TP 还是 Replica 扩展，都优先选择 P2D4 而不是 P4D2。
2. 811 GB/s 只应作为假设性互联带宽，不代表 RTX PRO 6000 的真实能力。
3. 480 GB/s Host Memory Bandwidth 应通过本机 NUMA 感知的 STREAM 实测校准；后续必须固定 write、只扫 read，或固定 read、只扫 write，避免本轮读写同时变化的因果混杂。
4. Host bandwidth 验证必须构造“写入 Host L2→GPU L1 淘汰→再次命中并 load-back”的请求序列，并同时报告 `l2_load_latency`、`l2_backup_latency` 和 overlap 后暴露时间。仅开启 L2 和 write-through 不保证 read bandwidth 会影响 E2E。
5. KV transfer 带宽评估应在缩短队列或提高传输占比的 workload 上进行；当前 128 GB/s 已经证明参数生效，但不能用于推断更低带宽下的拐点。

## 10. 后续验证计划

| 优先级 | 验证项 | 预期产物 |
| --- | --- | --- |
| P0 | 定位并排除 `created_time=0` 的 `1x1` 请求 | 修复代码、单元测试、重新生成十一套 PD 分离指标 |
| P0 | 核对 `bw_gbps` 单位 | 字段迁移或换算修复、兼容性测试、配置文档 |
| P0 | 核对 PD 分离 L2 backup 记账 | 请求时钟与 iteration I/O 时长对齐测试 |
| P0 | 复核 P4D2 全面回退 | TP2/TP4 Prefill 单点 AIC 曲线、iteration trace、拓扑映射检查 |
| P0 | 复核 Prefill Replica 扩展失效 | P2D2/P4D2 的 replica admission、affinity、busy-until 与 batch trace |
| P0 | 复核 P8D8 长输入 TPOT 回退 | Decode 每副本 batch-size、队列时间、利用率与路由分布 |
| P1 | 增加多 seed | 至少 3 个 seed，报告 mean、标准差和置信区间 |
| P1 | Host BW 单变量测试 | 固定 write 的 read sweep、固定 read 的 write sweep、强制 L2 load-back workload |
| P1 | KV BW 拐点测试 | 控制队列后的 811/128/64/32 GB/s sweep 与传输占比曲线 |


## 11. 已知限制

1. 本报告基于仿真结果，没有对应真实 GPU 多卡 ground truth，不能计算 MAPE 或证明绝对精度。
2. RTX PRO 6000 不支持 NVLink，811 GB/s 是替代拓扑假设；结论不代表该 GPU 的真实多卡互联性能。
3. Host Memory Bandwidth 480 GB/s 未经本机实测校准。
4. 仅测试 Qwen3-8B、FP16 和当前 AIC 数据库，不能直接外推到 MoE、FP8 KV Cache 或其他模型。
5. 仅使用 seed=1，且十一套 PD 分离 mean 指标受特殊请求影响。
6. 本轮启用了 L2 HiCache，但没有 no-cache/L1 对照组，因此无法从本轮数据中单独量化 480 GB/s 参数带来的收益。

## 12. 数据与制品位置

| 制品 | 路径或标识 |
| --- | --- |
| 本报告 | `260724-NVLink与Memory_bandwidth测试报告.md` |
| 核心配置 | `tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json` |
| PD 合并 TP2 | `cases_same_seed_1-tp2-dp1-PDagg` |
| PD 合并 TP4 | `cases_same_seed_1-tp4-dp1-PDagg` |
| PD 合并 TP4、Host BW64 | `cases_same_seed_1-tp4-dp-1-MemBW64-PDagg` |
| PD 分离 P1D1 | `cases_same_seed_1-P1D1-tp1-dp1-PDdisagg` |
| PD 分离 TP-P2D2 | `cases_same_seed_1-P2D2-Ptp2-Dtp2-dp1-PDdisagg` |
| PD 分离 TP-P4D2 | `cases_same_seed_1-P4D2-Ptp4-Dtp2-dp1-PDdisagg` |
| PD 分离 TP-P2D4 | `cases_same_seed_1-P2D4-Ptp2-Dtp4-dp1-PDdisagg` |
| PD 分离 Replica-P3D1 | `cases_same_seed_1-P3D1-tp1-dp1-Preplica3-PDdisagg-SingleReplica` |
| PD 分离 Replica-P2D2 | `cases_same_seed_1-P2D2-tp1-dp1-Preplica2-Dreplica2-PDdisagg` |
| PD 分离 Replica-P2D2、Host BW64 | `cases_same_seed_1-P2D2-tp1-dp1-Preplica2-Dreplica2-MemBW64-PDdisagg` |
| PD 分离 Replica-P2D2、KV BW128 | `cases_same_seed_1-P2D2-tp1-dp1-Preplica2-Dreplica2-KVBW128-PDdisagg` |
| PD 分离 Replica-P4D2 | `cases_same_seed_1-P4D2-tp1-dp1-Preplica4-Dreplica2-PDdisagg` |
| PD 分离 Replica-P2D4 | `cases_same_seed_1-P2D4-tp1-dp1-Preplica2-Dreplica4-PDdisagg` |
| PD 分离 Replica-P8D8 | `cases_same_seed_1-P8D8-tp1-dp1-Preplica8-Dreplica8-PDdisagg` |