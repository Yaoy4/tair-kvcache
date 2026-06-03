
---

## 一、Hisim 与 AIConfigurator 的整体逻辑

### 1.1 Hisim 是什么

Hisim（**Hi**gh-performance **Si**mulation）是一个**基于 CPU 的 LLM 推理仿真系统**。它的核心目标是：

> 在不消耗真实 GPU 资源的前提下，通过回放真实生产 trace 或合成 workload，快速预测不同模型、硬件、推理框架配置下的关键性能指标（TTFT、TPOT、ITL、吞吐量等）。

Hisim **不是**一个独立的推理引擎，而是"寄生"在 SGLang 上的仿真层：

```
┌──────────────────────────────────────────────────────────────┐
│  SGLang 推理框架（正常代码路径）                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │Tokenizer │ → │Scheduler │ → │ModelRunner│ → │Detokenizer│ │
│  │Manager   │   │          │   │           │   │           │ │
│  └──────────┘   └────┬─────┘   └─────┬────┘   └──────────┘ │
│                      │  ← Hook        │  ← Hook              │
│                      ▼                ▼                       │
│               ┌─────────────────────────────────┐            │
│               │         Hisim 仿真层              │            │
│               │  ① 拦截 Scheduler.run_batch       │            │
│               │  ② 调用 AIC 预测延迟               │            │
│               │  ③ 推进全局时钟                    │            │
│               │  ④ 记录 per-request 统计           │            │
│               └─────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 AIConfigurator（AIC）是什么

AIC（[ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator)）是一个**算子级延迟建模框架**：

- 内置各 GPU 型号（H100、H20、B60、RTX PRO 6000 等）的 **算子 profiling 数据库**（`PerfDatabase`）
- 通过 `InferenceSession.run_static()` 接收一次推理的 `RuntimeConfig`（batch_size、isl、osl、prefix），返回各算子的延迟分解
- 支持 `static_ctx`（prefill）和 `static_gen`（decode）两种模式
- 支持量化类型映射（FP16/BF16/FP8/INT8/FP4 → GEMM/FMHA/KVCache/MoE QuantMode）

AIC 对 Hisim 来说是一个**黑盒时间预测器**，提供的接口是：

```
RuntimeConfig(batch_size, isl, prefix, osl) → latency_dict {op_name: ms}
```

### 1.3 整体工作流程

```
用户请求（真实 trace / 合成） 
        │
        ▼
bench_serving.py（HTTP 客户端，模拟并发请求）
        │  POST /generate
        ▼
SGLang HTTP Server
        │
        ▼ [TokenizerManager]（Hisim Hook：记录 server_created_time）
        │
        ▼ [Scheduler]（Hisim Hook 全面接管）
        │  ① wrapped_recv_requests：按时间戳排序进入 FUTURE_QUEUE
        │  ② wrapped_get_new_batch_prefill：从队列取 batch，记录 queue_end
        │  ③ wrapped_run_batch：
        │      - 收集每个请求的 (input_length, past_kv_length)
        │      - 构造 HisimScheduleBatch
        │      - 调用 AIC 预测延迟 → predicted_latency
        │      - 推进 StateManager.global_clock
        │  ④ wrapped_process_batch_result：
        │      - 记录每个 token 的生成延迟（ITL）
        │      - PD 分离模式：注入 KV transfer 延迟
        │      - 汇总 iteration_stats
        │
        ▼ [ModelRunner]（Hisim Hook：forward 直接返回假 logits，不做实际计算）
        │
        ▼ [profile endpoint]（wrapped_profile）
           - 计算 TTFT / TPOT / ITL / 吞吐量
           - 写出 metrics.json / request.jsonl / iteration.jsonl
```

---

## 二、PD 分离场景下 Hisim 需要做哪些修改

### 2.1 背景：什么是 PD 分离（Prefill-Decode Disaggregation）

传统 LLM 推理中，prefill（处理输入 token）和 decode（逐 token 生成）在同一 GPU 上进行。PD 分离将二者拆到不同节点：

- **P 节点**：专门做 prefill，处理完成后将 KV Cache 通过网络传输到 D 节点
- **D 节点**：专门做 decode，接收到 KV Cache 后开始逐 token 生成

从仿真角度看，需要在 prefill 完成后额外注入一段 **KV Cache 网络传输延迟**。

### 2.2 已完成的修改（当前实现）

#### 2.2.1 配置项（`SchedulerConfig`）

```python
# src/hisim/simulation/types.py
@dataclass
class SchedulerConfig:
    ...
    pd_disagg_enabled: bool = False
    pd_kv_transfer_bandwidth_gb: Optional[float] = None  # P→D 传输带宽，单位 GB/s
```

JSON 配置文件中对应：

```json
{
  "scheduler": {
    "pd_disagg_enabled": true,
    "pd_kv_transfer_bandwidth_gb": 100.0
  }
}
```

#### 2.2.2 KV Transfer 延迟计算与注入

在 `wrapped_process_batch_result`（`C_SchedulerHook`）中，每次 prefill batch 完成后：

```python
# src/hisim/simulation/sglang/sglang_hook.py：wrapped_process_batch_result
if (
    sched_config.pd_disagg_enabled
    and C_SchedulerHook.HISIM_BATCH.is_prefill()
):
    # 只统计最终完成 prefill 的 token（排除 chunked prefill 中间块）
    final_prefill_tokens = sum(
        hreq.input_length
        for sreq, hreq in zip(batch.reqs, hisim_reqs)
        if sreq.is_chunked == 0 and hreq.input_length > 1
    )
    
    kv_bytes_per_token = ConfigManager.get_kv_cache_bytes()
    kv_transfer_dur = final_prefill_tokens * kv_bytes_per_token / (
        sched_config.pd_kv_transfer_bandwidth_gb * 1e9
    )
    # 在统计 request stats 之后推进全局时钟（保证 TTFT = 纯 prefill 时间）
    StateManager.step_global_clock(kv_transfer_dur)
```

**关键设计决策**：KV transfer 延迟的时钟推进发生在 `request.gen_token_latencies[0]` 记录完成**之后**，因此：

| 指标 | 语义 |
|------|------|
| `TTFT` | = 纯 prefill 时间（不含 KV 传输） |
| `ITL[0]`（第一个 decode token 延迟）| = KV 传输时间 + 第一个 decode 时间 |
| `mean_kv_transfer_ms` | 每次 prefill batch 的平均 KV 传输延迟 |
| `total_kv_transfer_ms` | 所有 prefill batch 的 KV 传输时间总和 |

#### 2.2.3 Chunked Prefill 的正确处理

若开启 chunked prefill，一个长请求可能分多个 iteration 完成 prefill。只有 `is_chunked == 0`（最后一块）才触发 KV transfer，中间块不触发，避免重复计算。

#### 2.2.4 新增 KV Transfer 指标输出

在 `wrapped_profile` 中：

```python
kv_transfer_latencies = [
    item.get("kv_transfer_latency", 0.0)
    for item in C_SchedulerHook.ITERATION_STATS
    if item.get("kv_transfer_latency", 0.0) > 0.0
]
metrics["mean_kv_transfer_ms"] = float(np.mean(kv_transfer_latencies)) * 1000
metrics["total_kv_transfer_ms"] = float(np.sum(kv_transfer_latencies)) * 1000
```

最终写入 `metrics.json`。

### 2.3 尚未实现 / 可以继续扩展的点

| 扩展项 | 说明 |
|--------|------|
| P/D 分别建模 | 当前 P 和 D 共用同一份 AIC 数据库。真实场景下 P 节点和 D 节点可能配置不同的 GPU |
| 多请求并发 KV 传输 | 当前是串行计算每次 prefill batch 的 KV 传输。真实场景下网络带宽可能被多个请求共享 |
| KV Transfer 与 Decode 重叠 | 真实系统中 KV 传输可以和下一个 prefill 重叠；当前模型是串行 |
| P 节点队列独立建模 | P 节点和 D 节点各有自己的队列和调度，当前仿真只有一个统一调度器 |

---

## 三、Hisim 对 AIC 的互联 与 真实 Workload 推理逻辑（详细）

### 3.1 Hook 机制：Hisim 如何"寄生"在 SGLang 上

Hisim 使用**动态猴子补丁（monkey patching）**拦截 SGLang 的关键类，不修改 SGLang 源码。

#### 3.1.1 Hook 注册（`launch_server.py`）

```python
hisim_hook.install_class_hooks([
    sglang_hook.C_SchedulerHook,       # 核心：调度器完整接管
    sglang_hook.C_ModelRunnerHook,     # 绕过真实模型计算
    sglang_hook.C_TokenizerManagerHook,# 注入 server_created_time
    sglang_hook.C_StorageBackendFactory,# 替换 HiCache 存储后端为 Mock
    sglang_hook.C_HiCacheController,   # 仿真 L2 缓存 prefetch/backup 延迟
    sglang_hook.C_HiRadixCacheHook,    # 初始化替换 HiCache 组件
])
```

#### 3.1.2 Hook 的工作方式

每个 Hook 类定义：
- `HOOK_CLASS_NAME`：目标类名（如 `"Scheduler"`）
- `HOOK_MODULE_NAME`：目标模块路径（如 `"sglang.srt.managers.scheduler"`）
- `hook(cls, target)`：接收目标类，用 `setattr` 替换其方法

SGLang 在**运行时动态导入**这些类时，Hisim 已经注入了补丁，因此所有实例都是被 Hook 后的版本。

### 3.2 Hisim 对 AIC 的互联

#### 3.2.1 初始化链路

```
launch_server.py
    └── C_SchedulerHook.wrapped_init(self, ...)
            ├── ConfigManager.get_model_info(hf_config)     → ModelInfo（从 HuggingFace config.json 解析）
            ├── ConfigManager.get_accelerator_info()         → AcceleratorInfo（从 hisim config 读取 GPU 型号）
            ├── ConfigManager.get_scheduler_config(...)      → SchedulerConfig（tp/ep/dp/dtype/pd_disagg 等）
            └── ConfigManager.get_inference_time_predictor(model, hw, config)
                    └── AIConfiguratorTimePredictor.__init__(model, hw, config, ...)
                            ├── _load_perf_database(system=hw.name, backend="sglang", version=...)
                            │       └── aiconfigurator.sdk.perf_database.get_database(...)
                            │           → 加载对应 GPU 的算子 profiling 数据库
                            ├── get_perf_model(config, model)
                            │       ├── 构建 ModelConfig（量化模式、并行度）
                            │       └── models.get_model(model_name, model_config, backend_name)
                            │           → 返回 AIC 内部的 perf model（含 context_ops / generation_ops）
                            └── InferenceSession(model, backend, database)
                                → AIC 的推理会话，后续调用 run_static() 预测延迟
```

#### 3.2.2 数据类型映射

Hisim 将自己的 `DataType` 枚举映射到 AIC 的量化模式枚举：

```
DataType.FP16 / BF16  →  GEMMQuantMode.float16 / bfloat16
DataType.FP8          →  GEMMQuantMode.fp8
DataType.INT8         →  GEMMQuantMode.int8_wo
DataType.FP4          →  GEMMQuantMode.nvfp4
DataType.INT4         →  GEMMQuantMode.int4_wo
```

KV Cache 和 FMHA 也有独立的量化模式映射。

#### 3.2.3 预测调用链路（每个 batch）

```
C_SchedulerHook.wrapped_run_batch(batch)
    ├── 遍历 batch.reqs，构造 HisimScheduleBatch
    │       每个 FakeRequest 包含：
    │         - input_length: extend_input_len（prefill）或 1（decode）
    │         - past_kv_length: len(prefix_indices) + len(output_ids)
    │
    └── AIConfiguratorTimePredictor.predict_infer_time(hisim_batch)
            │
            ├── [if decode] ──────────────────────────────────────────────────
            │   isl = mean(past_kv_length)     # 平均历史 KV 长度作为 isl
            │   osl = 2                        # AIC 约定：decode 用 osl=2
            │   
            │   [可选] XGBoost 修正：
            │   如果加载了 xgb_bucket_models：
            │       根据 batch_size 选择对应的分桶模型
            │       预测 aic_gen_attn_ms / measured_gen_attn_ms 的比值
            │       gen_attn_scale = 1 / pred_ratio（用于校正 AIC 对 attention 延迟的高估）
            │   
            │   RuntimeConfig(batch_size, isl=isl, osl=2, gen_seq_imbalance_correction_scale=...)
            │   → session.run_static(config, mode="static_gen")
            │   → latency_dict = summary.get_generation_latency_dict()
            │
            └── [if prefill] ─────────────────────────────────────────────────
                mean_past = mean(past_kv_length)
                mean_input = mean(input_length)
                isl = int(mean_past + mean_input)   # 平均总序列长度
                prefix = int(mean_past)              # 平均已有 KV 长度
                osl = 1                             # AIC 约定：prefill 用 osl=1
                
                seq_imbalance_correction_scale = ctx_attn_flops_ratio_with_avg(batch.reqs)
                # 计算各请求 attention FLOPS 相对于"全部用平均值计算"的比值
                # 用于修正 batch 内各请求长度不均匀时的 attention 时间预测偏差
                
                RuntimeConfig(batch_size, isl, prefix, osl=1, seq_imbalance_correction_scale=...)
                → session.run_static(config, mode="static_ctx")
                → latency_dict = summary.get_context_latency_dict()
            
            infer_time = sum(latency_dict.values())   # 各算子延迟相加（ms）
            
            [if decode]:  infer_time *= decode_scale_factor
            [if prefill]: infer_time *= prefill_scale_factor
            
            infer_time += _estimate_tp_comm_time_ms(batch)
            # TP 通信开销 = 每层 2 次 ring-allreduce
            # 区分节点内（NVLink）和节点间（InfiniBand/RoCE/Ethernet）带宽
            
            return infer_time / 1e3   # 转换为秒
```

#### 3.2.4 TP 通信开销估算

当 `tp_size > 1` 时，AIC 的单 GPU 延迟需要叠加 TP 通信开销：

```python
def _estimate_tp_comm_time_ms(self, batch):
    # 每个 transformer 层 2 次 ring allreduce（attention 后 + FFN 后）
    collective_count = 2 * num_layers
    payload_bytes = tokens * hidden_size * dtype_bytes
    
    # ring-allreduce 时间公式：
    # transfer_ms = payload * 2*(N-1)/N / bandwidth * 1e3
    # launch_ms   = 2*(N-1) * latency_us / 1e3
    total_ms = collective_count * (transfer_ms + launch_ms)
```

支持的 interconnect 类型及默认参数：

| 模式 | 默认延迟(us) | 带宽效率 |
|------|------------|---------|
| none | 0 | 100% |
| nvlink | 1 | 100% |
| pcie | 3 | 90% |
| ib / infiniband | 5 | 90% |
| roce | 7 | 85% |
| ethernet | 10 | 80% |

---

### 3.3 真实 Workload 推理逻辑（完整流程）

#### 3.3.1 数据准备：真实 trace 格式

真实 trace 文件（`.jsonl`），每行一个请求：

```json
{
  "rid": "21e5xx",
  "timestamp": 732.31,
  "output_length": 1024,
  "input_length": 1024,
  "input_ids": [925, 3911, ...],
  "output_ids": [244, 129, ...],
  "queue_end": 7522.45,
  "final_prefix_cache_len": 0
}
```

`HisimCollectionDataset._load_dataset()` 会：
1. 读取所有行
2. 找到最小 `timestamp`，将所有时间戳归零（对齐到 0）
3. 每个请求转换为 `GenericRequest`，带 `custom_params={"created_time": ...}`

#### 3.3.2 请求发送：bench_serving.py 的模拟并发

`bench_serving.py` 是改造自 SGLang 的 bench 脚本，支持 `--bench-mode simulation`：

```
对每个请求：
  - 携带 custom_params: {"simulation": {"enabled": true, "created_time": ts, "total_request": N}}
  - HTTP POST /generate（异步，不等待结果）
  
所有请求发送完后等待全部响应 → 触发 /profile（输出 metrics）
```

#### 3.3.3 OFFLINE 仿真模式的时间线管理

Hisim 有两种仿真模式：

| 模式 | 说明 |
|------|------|
| `BLOCKING` | 推理延迟通过 `time.sleep()` 实现，用真实墙钟时间驱动，适合实时调试 |
| `OFFLINE` | 全部请求收齐后，用虚拟全局时钟（`StateManager.global_clock`）驱动，速度极快 |

**OFFLINE 模式详细流程**：

```
阶段 1：收集请求
────────────────────
wrapped_recv_requests()：
    接收所有 HTTP 请求 → 提取 created_time → 放入 FUTURE_QUEUE（最小堆）
    当收到的请求数 == total_request 时，设置 OFFLINE_RECV_ALL_REQUEST = True
    触发 heapq.heapify(FUTURE_QUEUE)，按 created_time 排序

阶段 2：仿真推进（每次 event_loop 迭代）
─────────────────────────────────────────
① wrapped_recv_requests()：
    current_ts = StateManager.get_global_clock()
    将 FUTURE_QUEUE 中 enqueue_time ≤ current_ts 的请求出队 → 加入 SGLang waiting_queue

② wrapped_get_new_batch_prefill()：
    原始 SGLang 逻辑组 batch（FCFS 或其他调度策略）
    记录 req_stats.queue_end = global_clock（等待队列出队时刻）
    
    特殊情况：
    - 若 running_batch 为空 且 waiting_queue 非空：时钟小步推进 5ms（等待 prefetch 完成）
    - 若 running_batch 为空 且 FUTURE_QUEUE 非空：时钟跳跃到下一个请求的 created_time

③ wrapped_run_batch(batch)：
    构造 HisimScheduleBatch，调用 AIC 预测 predicted_latency（秒）
    StateManager.set_current_inference_dur(predicted_latency)
    
    [注意：此处不推进时钟，时钟推进在 process_batch_result 中统一完成]

④ wrapped_process_batch_result(batch)：
    hicache_l2_load_dur  = pop L2 cache 加载延迟（本轮）
    hicache_l2_backup_dur = pop L2 cache 备份延迟（本轮）
    current_inference_dur = AIC 预测的推理延迟
    
    [非 overlap schedule 模式（默认）]：
    StateManager.step_global_clock(
        hicache_l2_load_dur + current_inference_dur + hicache_l2_backup_dur
    )
    request_response_time = global_clock
    
    [overlap schedule 模式]：
    # 假设 L2 load 与上一轮推理重叠
    StateManager.step_global_clock(
        max(hicache_l2_load_dur - last_inference_dur, 0)
    )
    StateManager.step_global_clock(current_inference_dur)
    request_response_time = global_clock + hicache_l2_backup_dur
    
    for req in batch.reqs:
        req_stats.gen_token_latencies.append(
            request_response_time - req_stats.last_event_time
        )
        req_stats.last_event_time = request_response_time
    
    [if PD disagg enabled and is_prefill]:
        kv_transfer_dur = final_prefill_tokens * kv_bytes_per_token / bandwidth
        StateManager.step_global_clock(kv_transfer_dur)  # 在 TTFT 记录后推进
```

#### 3.3.4 全局时钟与请求统计的对应关系

```
时间轴（global_clock）：

  t=0    t=q0   t=q1       t=p0              t=p0+kv     t=p0+kv+d0    t=p0+kv+d0+d1
  │      │      │           │                │            │             │
  │      │─────►│           │────[prefill]───►│──[kv tx]──►│──[decode0]──►│──[decode1]──►...
  │      │      │           │                │            │             │
  queue  │      queue_end   │                │            │             │
  start  │      = global_   TTFT = p0 - q0   │            ITL[0]        ITL[1]
         │      clock       (记录在此刻)       │      = kv_tx + d0       = d1
         │                                   │
         created_time                        step_clock(kv_transfer_dur)在此执行
```

**gen_token_latencies 数组的含义**：
- `[0]`：TTFT = prefill 完成时刻 - 请求被调度时刻（`queue_end`）
- `[1]`：第一个 decode token 延迟（PD disagg 模式下包含 KV 传输时间）
- `[2..N]`：后续 decode token 延迟（纯 decode 时间）

#### 3.3.5 HiCache L2 延迟的仿真

当开启 HiCache（hierarchical KV cache，GPU HBM → Host DRAM → Disk）时：

**L2 Prefetch（Disk → Host DRAM）**：
```
C_HiCacheController.handle_prefetch_operation()：
    remain_dur = current_inference_dur（当前推理时间预算）
    
    while remain_dur > 0 and prefetch_queue 非空：
        取 prefetch 操作
        查询 storage 命中数（storage_hit_count）
        计算可在 remain_dur 内完成的 token 数：
            completed_tokens = min(storage_hit_count, remain_dur * disk_bandwidth / kv_cache_bytes)
        
        if 未完成（分块预取）：
            记录 chunked_prefetch_operation，下次继续
        else：
            更新 req_stats.prefetch_complete_tokens
            remain_dur -= prefetch_dur
    
    实际延迟 = StateManager.inc_hicache_l2_load_dur(prefetch_dur)
    → 在 process_batch_result 的时钟推进中叠加
```

**L2 Backup（Host DRAM → Disk）**：
- 当前实现为即时完成（追踪 backup 操作但不计算精确延迟）
- 通过 `StateManager.inc_hicache_l2_backup_dur()` 记录

#### 3.3.6 KV Cache 容量估算

若未指定 `max_total_tokens`，Hisim 通过 AIC 估算 KV Cache 容量：

```python
def estimate_kv_cache_pool_capacity(model, device, scheduler_config):
    perf_model = get_perf_model(scheduler_config, model)
    weights = sum(op.get_weights() for op in perf_model.context_ops) / pp_size
    
    rest_memory = (mem_fraction_static * device.hbm_capacity_gb - 1.4GB) - weights
    kv_cache_space_per_token = kv_cache_cell_elems * data_type.bytes
    
    return int(rest_memory / kv_cache_space_per_token)
```

其中 KV cache 每个 token 占用（单 GPU）：

```python
# 标准 MHA：
kv_cache_cell_elems = num_kv_heads/tp_size * head_dim * num_layers/pp_size * 2 (K+V)

# MLA（如 DeepSeek）：
kv_cache_cell_elems = (kv_lora_rank + qk_rope_head_dim) * num_layers/pp_size
```

---

### 3.4 完整数据流图

```
用户准备 trace.jsonl
        │
        ▼
bench_serving.py
  ├── HisimCollectionDataset.load() → GenericRequest 列表（含 input_ids, created_time）
  ├── 异步并发发送 N 个 HTTP /generate 请求
  │     每个请求携带：
  │       prompt_token_ids = input_ids
  │       custom_params = {
  │           "simulation": {
  │               "enabled": true,
  │               "created_time": ts,
  │               "total_request": N
  │           }
  │       }
  └── 等待所有响应 → POST /profile
        ↓
SGLang HTTP Server（被 Hisim Hook 接管）
        │
        ▼ TokenizerManager（Hook）
          注入 server_created_time = time.time()
        │
        ▼ Scheduler（Hook）── 核心仿真循环 ──────────────────────────────────────
        │                                                                        │
        │  FUTURE_QUEUE（最小堆，按 created_time 排序）                           │
        │  ┌─────────────────────────────────────────────────────────────────┐  │
        │  │ (ts0, salt, req0), (ts1, salt, req1), ...                       │  │
        │  └───────────────────────────┬─────────────────────────────────────┘  │
        │                              │ 按 global_clock 出队                     │
        │                              ▼                                          │
        │  SGLang waiting_queue → Scheduler.get_new_batch_prefill()              │
        │                              │ 按调度策略（FCFS）组 batch               │
        │                              ▼                                          │
        │  ┌──────────────────────────────────────────────────────────────────┐  │
        │  │  Prefill Batch：多个请求，各 input_length > 1                     │  │
        │  │  Decode Batch： 多个请求，各 input_length = 1                     │  │
        │  └──────────────────┬───────────────────────────────────────────────┘  │
        │                     │                                                   │
        │                     ▼ wrapped_run_batch()                               │
        │              ┌─────────────────────┐                                   │
        │              │ HisimScheduleBatch   │                                   │
        │              │ [FakeRequest(il, kv), │                                   │
        │              │  FakeRequest(il, kv), │                                   │
        │              │  ...]                 │                                   │
        │              └──────────┬────────────┘                                  │
        │                         │                                                │
        │                         ▼ AIC 预测                                       │
        │              ┌──────────────────────────────────────────────────┐       │
        │              │ AIConfiguratorTimePredictor.predict_infer_time() │       │
        │              │                                                   │       │
        │              │ if decode:                                        │       │
        │              │   isl = mean(past_kv_length)                     │       │
        │              │   RuntimeConfig(bs, isl, osl=2)                  │       │
        │              │   → session.run_static(mode="static_gen")        │       │
        │              │   → Σ latency_dict.values()                      │       │
        │              │                                                   │       │
        │              │ if prefill:                                       │       │
        │              │   isl = mean(past_kv + input)                    │       │
        │              │   prefix = mean(past_kv)                         │       │
        │              │   RuntimeConfig(bs, isl, prefix, osl=1)          │       │
        │              │   → session.run_static(mode="static_ctx")        │       │
        │              │   → Σ latency_dict.values()                      │       │
        │              │                                                   │       │
        │              │ + _estimate_tp_comm_time_ms()                    │       │
        │              │ × scale_factor                                   │       │
        │              │ → predicted_latency (秒)                         │       │
        │              └──────────────────────────────────────────────────┘       │
        │                         │                                                │
        │                         ▼ wrapped_process_batch_result()                 │
        │              全局时钟推进：                                               │
        │              global_clock += l2_load + inference + l2_backup            │
        │              记录每个请求的 gen_token_latencies                           │
        │              [if PD disagg + prefill]:                                   │
        │                global_clock += kv_transfer_dur                           │
        │                                                                          │
        └──────────────────────── 循环直到所有请求完成 ───────────────────────────┘
        │
        ▼ POST /profile（wrapped_profile）
          calc_metrics(all_request_stats)
          写出：
            metrics.json      （汇总统计：TTFT/TPOT/ITL/吞吐量/KV传输延迟）
            request.jsonl     （每个请求的详细统计）
            iteration.jsonl   （每次迭代的延迟分解）
```

---

## 四、总结

| 组件 | 职责 |
|------|------|
| **AIC** | 提供基于硬件 profiling 数据的算子级延迟数据库，给定 batch_size/isl/osl/prefix 返回延迟分解 |
| **Hisim Hook 层** | 通过猴子补丁拦截 SGLang 的 Scheduler/ModelRunner，将真实 GPU 计算替换为 AIC 查表预测 |
| **StateManager** | 维护全局虚拟时钟，驱动 OFFLINE 模式下的时间线推进 |
| **ConfigManager** | 解析 JSON 配置，初始化 ModelInfo / AcceleratorInfo / SchedulerConfig，创建 AIC 预测器 |
| **PD disagg** | 在 prefill batch 完成后，基于 KV 字节数和网络带宽，将 KV 传输延迟注入时钟（但不记入 TTFT）|
| **HiCache** | 仿真 L2 缓存 prefetch 延迟（按磁盘带宽和 idle 时间窗口计算），叠加到推理迭代时间中 |
