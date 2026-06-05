
# Hisim × AIConfigurator 解析

> 目录：
> - **第一章**：Hisim 解析
> - **第二章**：AIConfigurator（AIC）内部机制解析
> - **第三章**：PD 分离场景下 Hisim 需要做哪些修改
> - **第四章**：Hisim 对 AIC 的互联 + Hisim 对真实 workload 的推理逻辑
> - **第五章**：常见疑问 FAQ
> - **第六章**：总结
>
> 该总结主要包括：Hisim 的实现级细节（Hook 注入原理、配置解析链路、指标计算公式、KVCM 存储后端集成、AIC 内部 SOL 建模等），并提供术语表、数值算例与 FAQ。

---

## 0. 术语表

| 术语                 | 全称 / 含义                              | 在 Hisim 中的具体取值                                         |
| ------------------ | ------------------------------------ | ------------------------------------------------------ |
| **TTFT**           | Time To First Token，首 token 延迟       | `gen_token_latencies[0]`，= prefill 完成时刻 − 请求被调度时刻      |
| **TPOT**           | Time Per Output Token，平均每输出 token 时间 | `mean(gen_token_latencies[1:])`                        |
| **ITL**            | Inter-Token Latency，token 间延迟        | `gen_token_latencies[1:]` 的每一项                         |
| **isl**            | Input Sequence Length，输入序列长度         | prefill: `mean(past_kv+input)`；decode: `mean(past_kv)` |
| **osl**            | Output Sequence Length，输出序列长度        | AIC 约定：prefill 传 `osl=1`，decode 传 `osl=2`（见 §4.2.5）    |
| **prefix**         | 已缓存（复用）的 KV 前缀长度                     | prefill: `mean(past_kv_length)`                        |
| **past_kv_length** | 该请求当前已有的 KV 长度                       | `len(prefix_indices) + len(output_ids)`                |
| **prefill**        | 预填充阶段，一次性处理整段输入                      | batch 中存在 `input_length > 1` 的请求                       |
| **decode**         | 解码阶段，每步生成 1 个 token                  | batch 中所有请求 `input_length == 1`                        |
| **global_clock**   | Hisim 维护的虚拟全局时钟（秒）                   | `StateManager._global_clock`                           |
| **AIC**            | AIConfigurator，算子级延迟数据库              | 通过 `InferenceSession.run_static()` 查询                  |
| **HiCache**        | 分层 KV 缓存（HBM → DRAM → Disk）          | L1=HBM，L2=DRAM，L3=Disk                                 |
| **PD 分离**          | Prefill-Decode Disaggregation        | P 节点做 prefill，D 节点做 decode                             |
| **E2E**            | End-to-End，端到端延迟                     | `sum(gen_token_latencies)`                             |
| **queue_dur**      | 排队时长                                 | `queue_end − queue_start`                              |
| **SOL**            | Speed of Light，理论硬件极限延迟             | `max(sol_math, sol_mem)`（算力/访存取大，见 §2.4）              |
| **roofline**       | 算力 vs 访存瓶颈模型                         | SOL 公式的物理本质                                            |
| **TP/EP/PP/DP**    | 张量/专家/流水/数据并行度                       | `SchedulerConfig.tp_size / ep_size / pp_size / dp_size`|
| **KVCM**           | KV Cache Manager，KV 缓存管理器            | `MockHiCacheStorage` 可对接的真实存储后端                       |
| **前缀缓存**           | radix prefix cache，相同前缀复用 KV         | 影响 `past_kv_length` 与 prefill 工作量                      |
| **chunked prefill**| 长输入切块、跨多迭代完成 prefill                 | `req.is_chunked`（0=最后一块，1=未完）                          |
| **OOM**            | Out Of Memory，显存超限                    | `summary.check_oom()`，触发时返回负延迟                        |
| **BLOCKING/OFFLINE**| 两种仿真模式（真实墙钟 / 虚拟时钟）                 | `HISIM_SIMULATION_MODE`（默认 OFFLINE）                    |

---

## 0.1 常用函数速查

下表列出贯穿全文的核心函数，按照调用顺序排列，给出计算公式与参数说明，便于对照代码与后续章节。（执行类函数无数值公式，以“—”标注。）

| 序号  | 函数（所在模块）                                                                                                          | 计算公式                                                                                                                                                                                       | 参数说明                                                                                                                                                                                                                                                                                                                                       | 核心功能                                                                                                              |
| --- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| 1   | `install_class_hooks(hooks)` / `install_module_hooks(hooks)`<br>`hook/class_hook_entry.py`、`module_hook_entry.py` | —（执行类，覆盖 `builtins.__build_class__`）                                                                                                                                                       | `hooks`=待注册的 Hook 类列表；按“类名+模块名”匹配后在类定义时注入补丁                                                                                                                                                                                                                                                                                                | 在不修改 SGLang 源码的前提下注入仿真逻辑，实现零侵入接管                                                                                  |
| 2   | `get_model_info(hf_config)` / `get_scheduler_config(...)`<br>`simulation/manager/config.py`                       | —（解析配置）                                                                                                                                                                                    | `hf_config`=HF `config.json`（归一化层数/头数等字段）；`server_args`/JSON=tp/ep/dp/dtype/pd_disagg 等调度参数                                                                                                                                                                                                                                                | 解析模型与运行配置，转换为 Hisim 内部统一的参数表示                                                                                     |
| 3   | `get_inference_time_predictor(model, hw, sched)`<br>`simulation/manager/config.py`                                | —（构建并缓存预测器）                                                                                                                                                                                | `model`=ModelInfo；`hw`=平台/加速器信息（`hw.name` 被 `predictor.device_name` 覆盖）；`sched`=SchedulerConfig                                                                                                                                                                                                                                            | 初始化并缓存 AIC 延迟预测器，供后续逐 batch 调用                                                                                    |
| 4   | `AIConfiguratorTimePredictor.__init__(model, hw, config, …)`<br>`time_predictor/aiconfigurator.py`                | —（构建预测器实例）                                                                                                                                                                                 | `model`=ModelInfo；`hw`=AcceleratorInfo（`hw.name` 决定加载哪个 AIC 系统数据库）；`config`=SchedulerConfig；`database_mode`=SILICON/HYBRID/EMPIRICAL/SOL/SOL_FULL（PerfDatabase 数据来源）；`xgb_model_path`=可选 XGBoost 校正模型目录；`prefill_scale_factor`/`decode_scale_factor`=prefill/decode 标定系数；构造时把 `database._nearest_1d_point_helper` 覆写为放宽 `inner_only`（允许外插） | AIC 延迟预测器主类：加载 `PerfDatabase`、构建 `InferenceSession`、加载 XGBoost 桶模型，对外统一提供 `predict_infer_time` 等方法（5/6/10 号均为其方法） |
| 5   | `estimate_kv_cache_pool_capacity(model, device, sched)`<br>`simulation/utils.py`                                  | `rest_memory=(mem_fraction_static×HBM_bytes)−framework_reserved−weights`<br>`kv_cache_space_per_token = kv_cache_cell_elems × kv_dtype_bytes`<br>`capacity = ⌊rest_memory / per_token⌋`    | `weights`=单卡权重(`Σ op.get_weights()/pp`)；`1.4`=框架预留 GB；`mem_frac`=`mem_fraction_static` 静态显存占比；`hbm_gb`=显存容量；容量决定最大并发请求数                                                                                                                                                                                                                    | 估算单卡可容纳的 KV Cache 容量，决定最大并发请求数                                                                                    |
| 6   | `predict_infer_time(batch)`<br>`time_predictor/aiconfigurator.py`                                                 | `infer_ms = Σ latency_dict.values() × scale + tp_comm_ms`<br>返回 `infer_ms / 1000`（秒）；OOM 时返回 `-abs(infer_ms)/1000`                                                                         | `batch`=当前批（含 `reqs`、`batch_size`、`is_decode()`）；`scale`=`decode_scale_factor` 或 `prefill_scale_factor`（标定系数）；tp_comm_ms：TP 是当前集成里最明确会漏掉的通信，所以需要给tp的延迟打上补丁                                                                                                                                                                                 | 预测当前 batch 一次前向的推理延迟，是仿真的核心调用                                                                                     |
| 7   | `ctx_attn_flops_ratio_with_avg(reqs)`<br>`time_predictor/aiconfigurator.py`                                       | `scale = actual_flops / avg_flops`<br>`actual_flops = Σ_req (2·past+input)·input/2`<br>`avg_flops = (2·mean_past+mean_input)·mean_input/2 × N`                                             | `reqs`=批内请求列表；`past`=`past_kv_length` 已有 KV 长度；`input`=`input_length` 本次输入长；`N`=请求数；单请求时返回 `1.0`，`<0.4` 视为不可靠不校正                                                                                                                                                                                                                           | 校正批内请求长度不均衡导致的 attention 估算偏差，提升预测精度                                                                              |
| 8   | `InferenceSession.run_static(cfg, mode)`<br>AIC `inference_session.py`                                            | `latency_dict = Σ_op op.run(...)`<br>单算子 SOL：`sol = max(sol_math, sol_mem)`<br>`sol_math = ops/(tc_flops×compute)×1000`<br>`sol_mem = mem_bytes/mem_bw×1000`                               | `cfg`=RuntimeConfig（`batch_size` 批大小、`isl` 序列长、`osl` 输出长、`prefix` 缓存前缀）；`mode`：static_ctx=prefill、static_gen=decode、static=全链路；`tc_flops`/`mem_bw` 来自 SystemSpec；`compute`=量化算力倍率；延迟单位 ms                                                                                                                                                  | 查询算子延迟表并结合 SOL 公式，计算单次前向各算子耗时                                                                                     |
| 9   | `get_context_latency_dict()` / `get_generation_latency_dict()`<br>AIC `inference_summary.py`                      | `infer = Σ latency_dict.values()`                                                                                                                                                          | 无入参；分别返回 prefill / decode 阶段的 `{算子名: 延迟ms}`；Hisim 对所有 value 求和（串行假设）                                                                                                                                                                                                                                                                       | 提取各算子延迟字典并求和，得到单次前向总延迟                                                                                            |
| 10  | `check_oom()` / `set_memory_and_check_oom(...)`<br>AIC `inference_summary.py`                                     | `is_oom = total_gib ≥ mem_capacity / 2³⁰`<br>`kv_budget = (cap−non_kv)×free_frac×(1−reserved)×(1−tolerance)`                                                                               | `total_gib`=权重+激活+KV 占用(GiB)；`mem_capacity`=显存容量(B)；`free_frac`/`reserved`/`tolerance`=KV 预算的可用比例/预留/容差                                                                                                                                                                                                                                    | 判定当前配置是否超出显存容量，超限时返回告警信号                                                                                          |
| 11  | `_estimate_tp_comm_time_ms(batch)`<br>`time_predictor/aiconfigurator.py`                                          | `payload = tokens × hidden_size × dtype_bytes`<br>`transfer_ms = payload × 2(N−1)/N / bw × 1e3`<br>`launch_ms = 2(N−1) × lat_us / 1e3`<br>`total = 2·num_layers × (transfer_ms+launch_ms)` | `tokens`=decode 取 `batch_size`、prefill 取 `num_context_tokens`；`N`=`tp_size`；`bw`/`lat_us`=互联带宽/延迟（NVLink/IB 等档位）；`tp_size≤1` 返回 0                                                                                                                                                                                                          | 估算张量并行下的卡间通信开销，并叠加到推理延迟中                                                                                          |
| 12  | `get_kv_cache_bytes()`<br>`simulation/manager/config.py`                                                          | `kv_bytes = cell_elems × data_type.bytes`<br>标准 MHA：`cell_elems = max(n_kv//tp,1) × head_dim × (n_layers//pp) × 2`                                                                         | `tp`=`tp_size`、`pp`=`pp_size`；`n_kv`=KV 头数；`2`=K+V；`data_type.bytes`=每元素字节（FP16=2）；用于 PD KV 传输时长计算                                                                                                                                                                                                                                         | 计算每 token 的 KV Cache 字节数，用于估算 P→D 传输时长                                                                            |
| 13  | `step_global_clock(dur)` / `inc_iteration()`<br>`simulation/manager/state.py`                                     | `global_clock += dur`<br>`iteration += 1`                                                                                                                                                  | `dur`=本步推进的时长（秒），可为推理/L2 加载/KV 传输延迟；`inc_iteration` 无参，每个 batch 调一次                                                                                                                                                                                                                                                                        | 将本步耗时累加至虚拟全局时钟，推进仿真时间线                                                                                            |
| 14  | `calc_metrics(requests)`<br>`simulation/utils.py`                                                                 | `ttft = gen_lat[0]`，`tpot = mean(gen_lat[1:])`，`e2e = Σ gen_lat`<br>`throughput = num_req / total_dur_s`<br>`total_dur_s = max(last_event_time)`<br>`reuse_ratio = Σreused / Σinput`       | `requests`=RequestStats 列表；`gen_lat`=`gen_token_latencies`；各指标再算 mean/median/std/p90/p95/p99 并 ×1000 转 ms；warmup 请求剔除                                                                                                                                                                                                                      | 汇总各请求延迟，输出 TTFT、TPOT、吞吐等性能指标                                                                                      |

---

## 一、Hisim 深度解析

### 1.1 Hisim 是什么

Hisim（**Hi**gh-performance **Si**mulation）是一个**基于 CPU 的 LLM 推理仿真系统**。核心目标：

> 在不消耗真实 GPU 资源的前提下，通过回放真实生产 trace 或合成 workload，快速、低成本、高保真地预测不同模型 / 硬件 / 推理引擎 / 配置组合下的关键性能指标（TTFT、TPOT、ITL、吞吐量、E2E 延迟等）。

**关键定位澄清**：Hisim **不是**独立推理引擎，也**不重写** SGLang。它通过运行时注入（见 §4.1），让 SGLang 走完整的调度 / 内存管理 / 缓存逻辑，**唯独把"真正的 GPU 矩阵计算"这一步替换为 AIC 的延迟查表**。因此 Hisim 的调度行为、batching 行为、缓存命中行为与真实 SGLang 完全一致，只有"算得多快"是预测出来的。

```
┌──────────────────────────────────────────────────────────────┐
│  SGLang 推理框架（真实代码路径，未被改写）                       │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌───────────┐ │
│  │Tokenizer │ → │Scheduler │ → │ModelRunner│ → │Detokenizer│ │
│  │Manager   │   │（核心接管）│   │（计算被替换）│   │           │ │
│  └────┬─────┘   └────┬─────┘   └─────┬─────┘   └───────────┘ │
│       │ Hook         │ Hook          │ Hook                   │
│       ▼              ▼               ▼                         │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                   Hisim 仿真层                            │   │
│  │  ① 拦截 Scheduler.run_batch → 收集 batch 形状           │   │
│  │  ② 调用 AIC 预测该 batch 的推理延迟                       │   │
│  │  ③ 把延迟"加"到全局虚拟时钟上（而非真的等这么久）          │   │
│  │  ④ 按虚拟时钟记录每个请求的 per-token 延迟               │   │
│  │  ⑤ ModelRunner.forward 直接返回假的 logits（全 1）       │   │
│  └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 整体工作流程（端到端）

```
用户请求（真实 trace.jsonl / 合成 random）
        │
        ▼
bench_serving.py（HTTP 压测客户端，模拟并发，--bench-mode simulation）
        │  POST /generate（异步，不阻塞等待结果）
        ▼
SGLang HTTP Server
        │
        ▼ [TokenizerManager]（Hook：把 server_created_time 写进 custom_params）
        │
        ▼ [Scheduler]（Hook：全面接管调度循环）
        │   ① recv_requests        → 请求按时间戳入 FUTURE_QUEUE（最小堆）
        │   ② get_new_batch_prefill → 按虚拟时钟出队、组 batch、记录 queue_end
        │   ③ run_batch            → 构造 HisimScheduleBatch → 调 AIC 预测延迟
        │   ④ process_batch_result → 推进 global_clock、记录 ITL、（PD）注入 KV 传输
        │
        ▼ [ModelRunner]（Hook：forward 返回空 logits，sample 返回全 1，不做真实计算）
        │
        ▼ [/profile endpoint]（Hook：wrapped_profile）
            calc_metrics → 写出 metrics.json / request.jsonl / iteration.jsonl
```

---

## 二、AIConfigurator（AIC）内部机制深度解析

> 本章基于 aiconfigurator 源码，说明 AIC 内部如何用算子数据库 + SOL 模型 + 插值，把 `RuntimeConfig` 转换为 `latency_dict`。这对理解 Hisim 预测精度的来源、以及 Hisim 选择 `aiconfigurator` 作为 predictor 的原因至关重要。
>
> 核心 SDK 位于 `src/aiconfigurator/sdk/`。

### 2.0 AIC 的定位（原始项目视角）

AIC 本身是 NVIDIA 为 **Dynamo 分离式服务（disaggregated serving）** 设计的**部署配置优化器**：给定模型、GPU 数量、GPU 型号和 SLA（TTFT/TPOT 目标），它在巨大的配置空间（多少 prefill worker、多少 decode worker、各自并行度）里搜索，输出"在给定延迟下吞吐最优"的部署配置。

> 论文：《AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving》(arXiv:2601.06288)

**Hisim 只用了 AIC 的"底层延迟建模能力"**（`InferenceSession.run_static`），而没有用它上层的配置搜索（`DisaggInferenceSession`、`find_best_*`）。这点很关键：**Hisim 把 AIC 当"算子延迟计算器"，自己负责调度仿真**。

### 2.1 AIC SDK 模块全景

```
src/aiconfigurator/sdk/
├── perf_database.py       ← 算子 profiling 数据库（PerfDatabase），CSV 加载 + 查询 + 插值
├── inference_session.py   ← InferenceSession.run_static / DisaggInferenceSession
├── inference_summary.py   ← InferenceSummary：延迟/能耗/内存/OOM 结果容器
├── config.py              ← ModelConfig / RuntimeConfig（Hisim 构造的入参）
├── common.py              ← 各种枚举：DatabaseMode / *QuantMode / PerfDataFilename / BackendName
├── interpolation.py       ← 1D 最近邻/线性插值、2D 双线性插值
├── system_spec.py         ← SystemSpec：从 YAML 读硬件规格（flops/带宽/拓扑）
├── operations/            ← 算子类：GEMM / Attention / MoE / 通信 / MLA / Mamba …
├── backends/              ← TRTLLM / SGLANG / VLLM 后端：把算子串成一次前向
├── models/                ← 模型族：LLAMA / MOE / DeepSeek / Qwen3VL …（定义 context_ops/generation_ops）
└── systems/*.yaml         ← 各 GPU 系统规格（h100_sxm / h200_sxm / b200_sxm / rtx_pro_6000_server …）
```

### 2.2 调用层次：从 `run_static` 到 `latency_dict`

```
InferenceSession.run_static(runtime_config, mode)        # inference_session.py:54
   └─ backend.run_static(model, database, runtime_config, mode, stride=32, latency_correction_scale=1.0)
        # BaseBackend.run_static（backends/base_backend.py）
        ├─ 按 mode 选择阶段：
        │    static_ctx → 只算 context（prefill）阶段
        │    static_gen → 只算 generation（decode）阶段
        │    static     → encoder + context + generation 全链路
        ├─ 遍历 model.context_ops / model.generation_ops（有序算子列表）
        │    每个 op.run(database, ...) → 该算子延迟（ms）
        │    累加进 context_latency_dict / generation_latency_dict（按 op 名归类）
        │    同时记录来源标签 context_source_dict（silicon / empirical / sol）
        ├─ set_memory_and_check_oom(...) → 判断是否超显存
        └─ 返回 InferenceSummary（含 latency_dict / energy_dict / memory / oom 标志 / summary df）
```

Hisim 取的就是：

```python
summary = session.run_static(cfg, mode="static_ctx")   # 或 static_gen
latency_dict = summary.get_context_latency_dict()       # {op_name: latency_ms}
infer_time = sum(latency_dict.values())                  # Hisim 把所有算子延迟相加
```

> **说明**：`latency_dict` 的 key 是**算子类别**（如 `gemm`、`context_attention`、`custom_allreduce`），value 是该类算子在这一次前向中的**总延迟（ms）**。Hisim 直接对所有 value 求和，即"一次前向 = 各算子延迟之和"（串行假设，不建模算子间重叠）。

### 2.3 PerfDatabase：算子 profiling 数据库

#### 2.3.1 数据来源与组织

`PerfDatabase`（`perf_database.py`）从一组 CSV/TXT 文件加载各算子在不同 shape 下的**实测延迟**。文件名定义在 `common.PerfDataFilename`：

| 文件 | 算子 |
|------|------|
| `gemm_perf.txt` | 稠密 GEMM（矩阵乘）|
| `context_attention_perf.txt` | prefill 阶段 attention |
| `generation_attention_perf.txt` | decode 阶段 attention |
| `moe_perf.txt` | MoE 专家层 |
| `custom_allreduce_perf.txt` | 自定义 AllReduce（TP 通信）|
| `nccl_perf.txt` | NCCL 集合通信 |
| `context_mla_perf.txt` / `generation_mla_perf.txt` | MLA（DeepSeek 多头潜在注意力）|
| `mla_bmm_perf.txt` | MLA 批量矩阵乘 |
| `mamba2_perf.txt` / `gdn_perf.txt` | Mamba2 / GatedDeltaNet（混合架构）|
| `wideep_*` / `dsv4_*` / `dsa_*` | WideEP / DeepSeek-V4 / DSA 模块级 profiling |

> 这些数据来自在**真实目标机器**上对各算子做的 micro-benchmark。Hisim README 里说的"aiconfigurator 数据包"（从 LatencyPrism/hisim 下载、解压到 `download_path`）就是这些 CSV。

加载后绑定为 `self._gemm_data`、`self._context_attention_data` 等 `LoadedOpData` 包装。查询通过 `query_gemm` / `query_context_attention` / `query_generation_attention` / `query_moe` / `query_custom_allreduce` / `query_nccl` / `query_p2p` 等方法（均委托给对应 op 类）。

#### 2.3.2 DatabaseMode：5 种延迟来源

`common.DatabaseMode`（Hisim 通过 `predictor.database_mode` 配置，默认 `SILICON`）：

| 模式 | 值 | 含义 |
|------|----|------|
| `SILICON` | 0 | **默认**。用真实硅片实测数据（最准）|
| `HYBRID` | 1 | 有实测就用实测，否则回退到 **SOL + 经验因子** |
| `EMPIRICAL` | 2 | 全部用 **SOL + 经验因子**（不查实测表）|
| `SOL` | 3 | 只返回 **SOL（Speed-Of-Light，理论极限）** 延迟 |
| `SOL_FULL` | 4 | 返回 SOL 延迟 + 细节（`sol_time / sol_math / sol_mem`）|

**SOL（Speed of Light）= 理论硬件极限延迟**，取"算力受限"和"访存受限"二者的较大值（见 §2.4）。实测数据通常 = SOL × 某经验效率因子（如 MoE 经验值 = `sol / 0.4`，即只能达到 40% 的 SOL 效率）。

> Hisim 默认用 `SILICON`，因此精度最高（README 报告误差 < 5%）；当目标 GPU 没有实测数据时，可切到 `HYBRID`/`EMPIRICAL` 让 AIC 用解析公式估算。

#### 2.3.3 插值与外插

- `nearest_1d_point_helper(x, values, inner_only=True)`（`interpolation.py`）：返回 x 的左右包夹采样点。`inner_only=True` 时只在采样范围**内**插值；`inner_only=False` 时允许从边缘邻居**外插**。
- `interp_1d`：1D 线性插值（支持标量或 `{latency, power}` 叶子）。
- `bilinear_interpolation`：2D 双线性插值（例如同时按 batch 和 seq_len 两维插值）。

> **回顾 §4.2.6**：Hisim 用 `wrapped_nearest_1d_point_helper` 强制 `inner_only=False`，**让 AIC 在 trace 出现数据库未覆盖的超大 batch / 超长序列时也能外插出预测值**，避免仿真中断。代价是外插区间精度下降。

#### 2.3.4 `_correct_data` 与 system_spec 兼容

`PerfDatabase._correct_data()` 会对 GEMM 和 GenerationAttention 重新套用 SOL 钳制（`_correct_sol`）。Hisim 在加载数据库前用 `_install_aic_system_spec_compatibility()` 包装它，先调 `_backfill_aic_system_spec` 补 `float16_tc_flops`（从 `bfloat16_tc_flops` 拷贝），以兼容新旧数据库字段差异。

### 2.4 算子延迟模型（以 SOL 公式为例）

AIC 每个算子都有"**实测表查询**"和"**SOL 解析公式**"两条路径。SOL 公式揭示了延迟的物理本质——**算力 vs 访存的 roofline**。

#### 2.4.1 Attention（`operations/attention.py`）

context attention 的 SOL（已核对源码 `attention.py:215-226`）：

```python
full_s = s + prefix
if w > 0 and full_s > w:                      # 滑动窗口注意力
    ops = 2 * b * (full_s - prefix) * w * n * h * 2
else:                                          # 标准全注意力
    ops = 2 * b * (full_s*full_s - prefix*prefix) * n * h * 2 / 2
mem_bytes = 2*b*(n*(full_s-prefix)*h + n*(full_s-prefix)*h) \
          + kvcache_quant_mode.value.memory * b * (2 * n_kv * full_s * h)
sol_math = ops / system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
sol_mem  = mem_bytes / system_spec["gpu"]["mem_bw"] * 1000
sol_time = max(sol_math, sol_mem)             # roofline：取算力/访存瓶颈的较大者
```

其中：
- `b`=batch, `s`=本次输入长度, `prefix`=已缓存前缀, `n`=注意力头数, `n_kv`=KV 头数, `h`=head_dim, `w`=滑窗大小；
- `sol_math` = 计算量 / 算力（受 `fmha_quant_mode.compute` 加速，FP8 compute=2 表示 FP8 算力是 BF16 的 2 倍）；
- `sol_mem` = 访存字节 / 显存带宽（受 `kvcache_quant_mode.memory` 影响，FP8 KV 占 1 字节）；
- **prefix 越大（缓存命中越多），`full_s²−prefix²` 减小 → ops 下降 → attention 越快**。这正是 Hisim 把 `past_kv_length` 作为 prefix 传给 AIC 的原因——前缀缓存命中会直接降低 prefill attention 延迟。

#### 2.4.2 GEMM（`operations/gemm.py`）

表查询 + SOL 钳制 + 外插。量化模式通过 `GEMM._get_quant_tc_flops()` 影响有效算力：`GEMMQuantMode` 的 `compute` 因子（见 §2.5）决定 tensor core FLOPS 倍率（如 `nvfp4` compute=4，算力是 BF16 的 4 倍）。

#### 2.4.3 MoE（`operations/moe.py`）

```python
total_tokens = num_tokens * topk                # 每 token 路由到 topk 个专家
ops = total_tokens * hidden_size * inter_size * num_gemms * 2 // moe_ep_size // moe_tp_size
sol_time = max(ops/(bfloat16_tc_flops*compute)*1000, mem_bytes/mem_bw*1000)
# 经验回退：empirical = sol / 0.4   （实测 MoE 仅达 ~40% SOL 效率）
```

#### 2.4.4 通信（`operations/communication.py`）

- **CustomAllReduce** SOL：`2 * size * 2 / tp_size * (tp_size-1) / p2p_bw * 1000`（ring-allreduce 经典公式）。若 `tp_size > num_gpus_per_node`，按跨节点带宽缩放。
- **P2P**：纯解析，由拓扑带宽/延迟得出（无实测表）。

> **对比 Hisim 的 TP 通信估算（§4.2.8）**：Hisim 自己用 `_estimate_tp_comm_time_ms` 又算了一遍 TP 通信，叠加在 AIC 返回值之上。这是因为 Hisim 在 server 启动时**不让 SGLang 真正开 TP**（用单进程模拟），所以 AIC 返回的是"单卡算子延迟"，TP 通信需 Hisim 额外补。**潜在重复风险**：若 AIC 的 `context_ops` 里也含 `CustomAllReduce`，可能与 Hisim 的补偿叠加——使用时需注意 `scheduler.tp_size` 的语义边界。

### 2.5 量化模式枚举（QuantMapping：memory + compute 因子）

`common.py` 中每个 QuantMode 是一个 `QuantMapping(memory, compute, name)`：
- `memory` = 每元素字节数因子（影响访存量 `sol_mem`）；
- `compute` = tensor core 算力倍率（影响算力 `sol_math`，倍率越大算得越快）。

| 枚举 | 成员（memory, compute）|
|------|------------------------|
| **GEMMQuantMode** | `bfloat16`(2,1) `int8_wo`(1,1) `int4_wo`(0.5,1) `fp8`(1,2) `fp8_static`(1,2) `sq`(1,2) `fp8_block`(1,2) `fp8_ootb`(1,2) `nvfp4`(9/16,4) |
| **MoEQuantMode** | `bfloat16`(2,1) `fp8`(1,2) `int4_wo`(0.5,1) `fp8_block`(1,2) `w4afp8`(0.5,2) `nvfp4`(9/16,4) `w4a16_mxfp4`(0.5,1) `w4a8_mxfp4_mxfp8`(0.5,2) |
| **FMHAQuantMode** | `bfloat16`(2,1) `fp8`(1,2) `fp8_block`(1,2) |
| **KVCacheQuantMode** | `bfloat16`(2,0) `int8`(1,0) `fp8`(1,0) |
| **CommQuantMode** | `half`(2,0) `int8`(1,0) `fp8`(1,0) |

> 这解释了 §4.2.2 里 Hisim 的 DataType→QuantMode 映射为何重要：选错量化模式会直接改变 `sol_math`/`sol_mem`，进而影响预测延迟。例如把 FP8 误当 BF16，compute 从 2 掉到 1，预测延迟会翻倍。

### 2.6 模型定义：context_ops / generation_ops

`models.get_model(model_path, model_config, backend_name)`（`models/__init__.py`）根据 HF config 的架构映射到模型族，并构建三个**有序算子列表**（`models/base.py`）：

- `encoder_ops`：（多模态）ViT 编码器算子；
- `context_ops`：prefill 前向的算子骨架；
- `generation_ops`：decode 前向的算子骨架。

模型族示例：

| 族 | 类 | 典型算子组成 |
|----|----|-------------|
| 标准 Dense | `LLAMAModel` / `GPTModel` | GEMM(QKV/O/FFN) + ContextAttention/GenerationAttention + CustomAllReduce + P2P |
| MoE | `MOEModel` / `SGLangEPMOEModel` | + MoE / MoEDispatch（替换 FFN）|
| MLA | `DeepSeekModel` / `DeepSeekV32Model` / `DeepSeekV4Model` | ContextMLA/GenerationMLA + MLABmm |
| 混合 | `NemotronHModel` / `HybridMoEModel` | Mamba2 / GDN + MoE + Attention 交错 |
| 多模态 | `Qwen3VLModel` / `Qwen3VLMoEModel` | encoder_ops(ViT) + LLM backbone |

> `backends/` 决定算子如何组装（TRTLLM/SGLANG/VLLM 在 kernel 融合、通信策略上有差异）。**Hisim 固定用 `backend_name="sglang"`**（`SchedulerConfig.backend_name`），保证与被仿真的 SGLang 框架一致。

### 2.7 InferenceSummary 与 OOM 判定

`InferenceSummary`（`inference_summary.py`）是 `run_static` 的返回容器，含：runtime config、内存字典、encoder/context/generation 的延迟字典与能耗字典（W·ms = mJ）、每算子来源标签、OOM 标志、功率均值、汇总 DataFrame。

关键方法：
- `get_context_latency_dict()` / `get_generation_latency_dict()` → Hisim 取延迟用；
- `set_memory_and_check_oom(...)`：`is_oom = total_gib >= mem_capacity / 2^30`，还可按 KV cache 预算判定：
  `kv_budget = (cap − non_kv) * free_frac * (1−reserved) * (1−tolerance)`；
- `check_oom()` → Hisim 在 `predict_infer_time` 里调用，OOM 时返回**负延迟**作为信号（§4.2.4、FAQ Q6）。

### 2.8 SystemSpec：硬件规格（YAML）

`SystemSpec`（`system_spec.py`，继承 `dict`）从 `systems/<name>.yaml` 读取。可用系统：`a100_pcie/a100_sxm/a30/b60/b200_sxm/b300_sxm/gb200/gb300/h100_pcie/h100_sxm/h200_sxm/l4/l40s/rtx_pro_6000_server`。

关键字段（以 `h100_sxm.yaml` 为例）：

```yaml
gpu:
  mem_bw: 3.35e12          # 显存带宽 B/s（= 3350 GB/s，即 H100 的 HBM 带宽）
  mem_capacity: 80 GiB
  bfloat16_tc_flops: 9.89e14   # BF16 tensor core 算力（= 989 TFLOPS）
  fp8_tc_flops: 1.978e15       # FP8 算力（= 1978 TFLOPS，2× BF16）
node:
  num_gpus_per_node: 8
  intra_node_bw: 4.5e11    # NVLink 节点内带宽（450 GB/s）
  inter_node_bw: 5e10      # 节点间带宽（50 GB/s）
  p2p_latency: 10us
```

`get_p2p_bandwidth(num_gpus)` 三级拓扑选择：
- `≤ num_gpus_per_node` → `intra_node_bw`（NVLink）
- `≤ num_gpus_per_rack` → `inter_node_bw`（机架内 NVSwitch）
- 否则 → `inter_rack_bw`（跨机架 IB，缺省回退 `inter_node_bw`）

> **对应 Hisim**：§4.2.1 提到的 `predictor.device_name`（如 `h100_sxm`）正是这里的 YAML 文件名；SOL 公式里的 `bfloat16_tc_flops`、`mem_bw` 都来自该文件。这就是为什么 device_name 决定预测精度。

### 2.9 AIC 原生 PD 分离 vs Hisim 的 PD 实现

AIC **自身有** `DisaggInferenceSession`（`inference_session.py:142`）原生建模 PD 分离：

- 用**独立的** prefill / decode 的 `InferenceSession`（数据库、并行度、精度可不同）；
- `set_latency_correction_scales(prefill, decode, encoder)`：对延迟做标定（`corrected = latency × scale`）；
- **rate-matching 退化因子**：`_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR`（默认 0.9，prefill 流水线气泡）、`_RATE_MATCHING_DECODE_DEGRADATION_FACTOR`（默认 0.92，decode batch 欠饱和）；
- prefill 走 `static_ctx`、decode 走 `static_gen`，再由 `_build_disagg_summary_dict` 合并；
- **但 AIC 不显式建模 KV cache 跨节点传输**——源码有 TODO：`# TODO, should consider kvcache model in future`（`inference_session.py:153`）。

#### 2.9.1 AIC 里面的 PD 分离具体是怎么实现的？

如果把 AIC 原生 PD 说得更直白，它做的不是“按请求真实回放一条 P→D 时间线”，而是“把一次推理拆成两个静态阶段分别估时，再把结果拼起来”：

1. **拆阶段**：prefill 和 decode 分别由各自的 `InferenceSession` 建模；
2. **分模式**：prefill 仍然调用 `static_ctx`，decode 仍然调用 `static_gen`；
3. **做标定**：两阶段各自允许乘延迟校正系数（latency correction scale）；
4. **做稳态退化修正**：用 rate-matching degradation factor 近似表示 prefill 流水线气泡、decode batch 欠饱和这类稳态损失；
5. **做汇总**：最后由 `_build_disagg_summary_dict` 把 prefill 和 decode 的结果合并成一个面向配置搜索的 summary。

所以，AIC 原生 PD 的重点是：**把 P 和 D 当成两条独立的阶段性性能路径来估算**。它更适合回答“这套 P/D worker 配置是否划算、吞吐是否合理”，而不是回答“某个具体请求在某一时刻为什么多等了几毫秒”。

#### 2.9.2 AIC缺少哪些部分？

这里最好分成两层来看。

**AIC 原生 PD 缺少的部分**：

- **缺少 P→D 的 KV Cache 传输模型**：这是最核心的缺口。AIC 原生会把 prefill 和 decode 分开估时，但不会显式计算“prefill 结束后，KV 要花多久从 P 节点传到 D 节点”；
- **缺少逐请求时间线**：它没有 `global_clock`、没有请求队列、没有 token 级事件推进，因此不能自然回答 TTFT、首个 decode ITL 是如何被传输拖慢的；
- **缺少动态排队/背压语义**：AIC 原生更偏静态配置分析，不直接建模 trace 回放里的等待队列、批次到达时序、批间相互影响；
- **缺少并发传输竞争与固定网络开销**：没有把多请求并发时的带宽分摊、协议固定开销、握手开销等因素单独拉出来建模。

**当前仓库接入层仍然缺少或简化的部分**：

- **当前并没有直接使用 AIC 的 `DisaggInferenceSession`**。本仓实际走的是 `AIConfiguratorTimePredictor.predict_infer_time`：prefill batch 用 `static_ctx`，decode batch 用 `static_gen`，然后再由 Hisim 的 Scheduler Hook 在 `wrapped_process_batch_result` 中补上一段 KV 传输延迟；
- **P/D 仍没有真正独立成两套调度系统**。当前仿真只有一个统一 Scheduler，P 和 D 没有各自独立的队列、资源池和调度策略；
- **KV 传输目前按简单公式注入**：`tokens × kv_bytes / bandwidth`，属于一阶近似；
- **传输与后续计算没有做细粒度重叠建模**：现在是直接推进虚拟时钟，而不是把网络、prefill、decode 放进一个更细的并行事件模型；
- **还有一个实现级精度注意项**：当前 `ConfigManager.get_kv_cache_bytes()` 用的是 `scheduler_config.data_type.bytes`，而不是 `scheduler_config.kv_cache_data_type.bytes`。如果权重 dtype 与 KV dtype 不同，这里的 PD 传输时长会有系统性偏差。

换句话说，**AIC 原生 PD 负责“阶段内算得多快”，Hisim 当前这层补丁负责“阶段间 KV 传得多慢，以及这段延迟在时间线上落到哪里”**。这也是为什么本仓的 PD 逻辑最终落在 `wrapped_process_batch_result`，而不是只靠 predictor 内部一次性完成。

**对比结论（关键洞察）**：

| 维度      | AIC 原生 DisaggInferenceSession | Hisim 的 PD 实现（§3）           |
| ------- | ----------------------------- | --------------------------- |
| 目的      | **静态配置搜索**（找最优 worker 数/并行度）  | **动态 trace 回放仿真**（逐请求时间线）   |
| KV 传输延迟 | ❌ 未建模（TODO）                   | ✅ 显式注入 `tokens×kv_bytes/带宽` |
| 时间线     | 无（吞吐/SLA 的稳态分析）               | 有（global_clock 逐迭代推进）       |
| 多请求建模   | ❌ 不用                          | ✅ 自己在 Scheduler hook 里实现    |

> **这正是 Hisim 在 §3 里补充 PD 逻辑的根本原因**：AIC 只提供"单阶段算子延迟"，而 PD 分离最关键的"P→D KV 传输延迟"AIC 没建模。Hisim 在 `wrapped_process_batch_result` 里基于 `ConfigManager.get_kv_cache_bytes()` 和 `pd_kv_transfer_bandwidth_gb` 自行补上，并把它吸收进 ITL[0]。两者形成互补：**AIC 算"算得多快"，Hisim 算"传得多慢 + 何时发生"**。

---

## 三、PD 分离场景下 Hisim 需要做哪些修改

### 3.1 背景：什么是 PD 分离（Prefill-Decode Disaggregation）

传统（aggregated）推理中，prefill 与 decode 在同一张卡上交替执行。**PD 分离**把两阶段拆到不同节点：

- **P 节点（Prefill）**：只做预填充。算完后，需要把这段输入对应的 **KV Cache 通过网络发送到 D 节点**。
- **D 节点（Decode）**：接收 KV Cache 后，开始逐 token 生成。

**为什么要分离？** prefill 是算力密集（compute-bound），decode 是访存密集（memory-bound）。分开后可以各自独立扩缩容、用不同并行策略，提升整体吞吐与资源利用率。

**对仿真的影响**：相比 aggregated，PD 分离多了一段 **"P→D 的 KV Cache 网络传输延迟"**。这段延迟必须在仿真时钟中体现出来。

### 3.2 已完成的修改（当前实现）

#### 3.2.1 配置项（`SchedulerConfig` + JSON）

```python
# src/hisim/simulation/types.py
@dataclass
class SchedulerConfig:
    ...
    pd_disagg_enabled: bool = False                 # 总开关
    pd_num_prefill_instances: int = 1               # P（prefill）实例数
    pd_num_decode_instances: int = 1                # D（decode）实例数
    pd_kv_transfer_bandwidth_gb: Optional[float] = None  # P→D 带宽，GB/s，开关打开时必填
```

```python
# src/hisim/simulation/manager/config.py：get_scheduler_config()
pd_disagg_enabled=scheduler_config.get("pd_disagg_enabled", False),
pd_num_prefill_instances=scheduler_config.get("pd_num_prefill_instances", 1),
pd_num_decode_instances=scheduler_config.get("pd_num_decode_instances", 1),
pd_kv_transfer_bandwidth_gb=scheduler_config.get("pd_kv_transfer_bandwidth_gb"),
```

JSON 配置：

```json
{
  "scheduler": {
    "tp_size": 1,
    "pd_disagg_enabled": true,
    "pd_num_prefill_instances": 1,
    "pd_num_decode_instances": 1,
    "pd_kv_transfer_bandwidth_gb": 100.0
  }
}
```

#### 3.2.2 KV Transfer 延迟的计算与注入

在 `C_SchedulerHook.wrapped_process_batch_result` 中，**每个 prefill batch 结束后**计算 KV 传输延迟，并在记录请求统计**之前**推进虚拟时钟：

```python
# src/hisim/simulation/sglang/sglang_hook.py
kv_transfer_dur = 0.0
sched_config = ConfigManager.get_cached_scheduler_config()
is_pd_prefill = (
    sched_config is not None
    and sched_config.pd_disagg_enabled
    and C_SchedulerHook.HISIM_BATCH is not None
    and not C_SchedulerHook.HISIM_BATCH.is_empty()
    and C_SchedulerHook.HISIM_BATCH.is_prefill()          # 仅 prefill batch 触发
)
if is_pd_prefill:
    hisim_reqs = C_SchedulerHook.HISIM_BATCH.reqs
    if len(batch.reqs) == len(hisim_reqs):
        # 只统计"本轮完成最终 prefill"的请求的 token
        final_prefill_tokens = sum(
            hreq.input_length
            for sreq, hreq in zip(batch.reqs, hisim_reqs)
            if sreq.is_chunked == 0 and hreq.input_length > 1
        )
    else:
        final_prefill_tokens = C_SchedulerHook.HISIM_BATCH.num_context_tokens  # 回退

    if final_prefill_tokens > 0:
        if not sched_config.pd_kv_transfer_bandwidth_gb:
            logger.warning("pd_disagg_enabled=True 但未设带宽，跳过 KV 传输仿真")
        else:
            kv_bytes_per_token = ConfigManager.get_kv_cache_bytes()   # 每 token 的 KV 字节数
            kv_transfer_dur = final_prefill_tokens * kv_bytes_per_token / (
                sched_config.pd_kv_transfer_bandwidth_gb * 1e9         # GB/s → B/s
            )
            StateManager.step_global_clock(kv_transfer_dur)           # 先推进虚拟时钟
            request_response_time += kv_transfer_dur                  # 让首条延迟项含传输时间
```

**为什么 KV 传输要算进 TTFT**：

在 PD 分离的语境下，TTFT 是用户拿到**第一个 token** 的时间。这个 token 并不是在 P 实例上产生的——它要等到 **KV cache 从 P 实例迁移到 D 实例之后，由 D 实例 decode 出来**。因此：

```
TTFT = Prefill 时间 + KV Cache 传输时间 + D 实例首个 decode token 时间
```

实现上分两步完成这条语义：

1. **prefill batch 结束**：先把时钟推进 `prefill 计算时间 + KV 传输时间`，并据此写入 `gen_token_latencies[0]`（此时它 = prefill + 传输），同时给该请求打上 `pd_first_decode_pending` 标记；
2. **该请求首个 decode 完成**：把这次 decode 的时长**折叠进 `gen_token_latencies[0]`**（而不是单独记为一条 ITL），并清除标记。于是 `TTFT = prefill + 传输 + 首个 decode`，后续 decode 才正常记为 ITL。

> **健壮性说明**：采用"先记录、后折叠"而非"延迟到首个 decode 才记录"，是为了保证 `gen_token_latencies[0]` 永远存在。即使请求因 EOS / 提前停止而没有 decode（`output_length == 1`），TTFT 也已 = `prefill + 传输`，不会出现取 `[0]` 越界。

| 指标                           | 语义（PD 分离模式）                          |
| ---------------------------- | ------------------------------------ |
| `TTFT`（`gen_token_latencies[0]`） | = **prefill + KV 传输 + 首个 decode**（首 token 由 D 实例产出） |
| `ITL`（`gen_token_latencies[1:]`） | = **首 token 之后**每个 decode 的时延（不含已折叠进 TTFT 的首个 decode） |
| `mean_kv_transfer_ms`        | 各 prefill batch 的 KV 传输延迟均值（ms）       |
| `total_kv_transfer_ms`       | 所有 prefill batch 的 KV 传输延迟总和（ms）      |

> **副作用**：由于"prefill token + 首个 decode token"被合并成同一条 TTFT，PD 请求的 `gen_token_latencies` 比非 PD 少一项。但这不影响任何指标——`total_output` 用的是 `output_length`，端到端时延 `sum(gen_token_latencies)` 也保持不变（仅把首个 decode 从 ITL 归并到了 TTFT）。

#### 3.2.3 Chunked Prefill 的正确处理

开启 chunked prefill 时，一个长输入会被切成多个 chunk，跨多个 iteration 完成 prefill。SGLang 用 `req.is_chunked` 标记：

- `is_chunked == 1`：本 chunk 不是最后一块，KV 还在累积 → **不传输**；
- `is_chunked == 0`：本 chunk 是最后一块，整段 KV 已就绪 → **触发传输**。

代码里 `if sreq.is_chunked == 0 and hreq.input_length > 1` 保证只在最后一块、且确实是 prefill 请求（`input_length > 1`）时计入传输 token，**避免对同一请求重复计算 KV 传输**。

#### 3.2.4 KV Transfer 指标聚合输出

```python
# wrapped_profile：从每轮 iteration 统计里捞出有传输的项做聚合
kv_transfer_latencies = [
    item.get("kv_transfer_latency", 0.0)
    for item in C_SchedulerHook.ITERATION_STATS
    if item.get("kv_transfer_latency", 0.0) > 0.0
]
metrics["mean_kv_transfer_ms"]  = float(np.mean(kv_transfer_latencies)) * 1000 if kv_transfer_latencies else 0.0
metrics["total_kv_transfer_ms"] = float(np.sum(kv_transfer_latencies))  * 1000 if kv_transfer_latencies else 0.0
```

每轮的 `kv_transfer_latency` 还会写进 `iteration.jsonl`，便于逐迭代分析。

### 3.3 PD 分离数值算例

设：Qwen3-8B，FP16 KV，单卡（tp=1，pp=1）。

- 每 token KV 字节数（标准 MHA）：
  `kv_bytes = num_kv_heads × head_dim × num_layers × 2(K+V) × dtype_bytes`
  假设 `num_kv_heads=8, head_dim=128, num_layers=36, dtype=FP16(2B)`：
  `kv_bytes = 8 × 128 × 36 × 2 × 2 = 147,456 B ≈ 144 KB/token`
- prefill 一个 1024-token 的请求：`final_prefill_tokens = 1024`
- 带宽 `pd_kv_transfer_bandwidth_gb = 100 GB/s = 1e11 B/s`

则：

```
kv_transfer_dur = 1024 × 147456 / 1e11 ≈ 1.51e-3 s ≈ 1.51 ms
```

效果：

- `TTFT` 增加约 **1.51 ms**（= prefill + **KV 传输 1.51 ms** + 首个 decode）；
- 该 1.51 ms 不再出现在 `ITL` 里（首个 decode 已折叠进 TTFT）；
- `total_kv_transfer_ms` 累加所有请求的 1.51 ms。

### 3.4 尚未实现 / 可继续扩展的点

| 扩展项           | 现状                         | 真实场景差异                |
| ------------- | -------------------------- | --------------------- |
| P/D 异构建模      | P、D 共用同一份 AIC 数据库 / 同一 GPU | 实际 P、D 可用不同卡、不同 TP/EP |
| 并发 KV 传输的带宽竞争 | 每个 prefill batch 串行独占带宽    | 多请求并发时带宽被分摊           |
| 传输与计算重叠       | 串行注入，不与后续计算重叠              | 真实系统传输可与下一 prefill 重叠 |
| P/D 独立队列与调度   | 仅一个统一 Scheduler            | P、D 各自有队列与调度策略        |
| 传输延迟的固定开销     | 只算 `bytes/带宽`，无握手/排队固定项    | 真实有连接建立、RDMA 握手等固定延迟  |

---

## 四、Hisim 对 AIC 的互联 + 真实 Workload 推理逻辑

### 4.1 Hook 机制：Hisim 如何挂载到 SGLang 上

Hisim 不直接用 setattr 替换方法，而是 **覆盖 Python 内建的 `__build_class__`** 来让补丁自动生效。

#### 4.1.1 注册（`launch_server.py`）

```python
import hisim.hook as hisim_hook
from hisim.simulation.sglang import sgl_kernel_hook, sglang_hook

if not torch.cuda.is_available():            # 纯 CPU 仿真环境
    hisim_hook.install_module_hooks([sgl_kernel_hook.M_SGLangKernelLoadUtilHook])

hisim_hook.install_class_hooks([
    sglang_hook.C_SchedulerHook,        # 核心：调度循环完整接管
    sglang_hook.C_ModelRunnerHook,      # 旁路真实模型计算
    sglang_hook.C_TokenizerManagerHook, # 注入 server_created_time
    sglang_hook.C_StorageBackendFactory,# HiCache 存储后端替换为 Mock
    sglang_hook.C_HiCacheController,    # 仿真 L2 prefetch/backup 延迟
    sglang_hook.C_HiRadixCacheHook,     # 替换 HiCache 组件初始化
])
```

#### 4.1.2 补丁如何自动生效（`class_hook_entry.py`）

```python
# src/hisim/hook/class_hook_entry.py
_builtins_build_class_ = builtins.__build_class__   # 保存原始内建

def _custom_build_class_(func, name, *bases, **kwargs):
    for hook in CLASS_HOOKS:
        if name == hook.HOOK_CLASS_NAME:                 # 类名匹配（如 "Scheduler"）
            module_name = func.__globals__.get("__name__", "")
            if module_name == hook.HOOK_MODULE_NAME:     # 模块名匹配
                target_class = _builtins_build_class_(func, name, *bases, **kwargs)  # 先正常建类
                new_class = hook.hook(target_class)       # 再交给 Hook 改写
                return new_class or target_class
    return _builtins_build_class_(func, name, *bases, **kwargs)

def install_class_hooks(hooks):
    _register_hooks(CLASS_HOOKS, hooks)
    builtins.__build_class__ = _custom_build_class_       # 替换全局内建
```

**为什么这样设计**：Python 在执行每一个 `class Foo(...):` 定义时，底层都会调用 `builtins.__build_class__`。Hisim 把这个内建函数整体替换掉，于是 **任何模块在 import 阶段定义目标类时，都会被拦截**。关键收益：

- **无需提前知道 SGLang 何时 import 哪个类** —— 只要那个类被定义，就会被自动 Hook；
- **不修改 SGLang 源码**，零侵入；
- 通过"类名 + 模块名"双重匹配，避免误伤同名类。

> 注意：因为是覆盖内建，`install_class_hooks` 必须在 `launch_server` 真正 import 并构建 SGLang 类**之前**调用 —— 这正是为什么注册代码位于 `launch_server.py` 顶部、`if __name__ == "__main__"` 之外。

#### 4.1.3 每个 Hook 的职责一览

| Hook 类                    | 目标 SGLang 类             | 做了什么                                                                                                        |
| ------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| `C_SchedulerHook`         | `Scheduler`             | 接管 `__init__/recv_requests/get_new_batch_prefill/run_batch/process_batch_result/profile/event_loop_overlap` |
| `C_ModelRunnerHook`       | `ModelRunner`           | `initialize` 用 Mock pool、`forward` 返回空 logits、`sample` 返回全 1                                                |
| `C_TokenizerManagerHook`  | `TokenizerManager`      | `_send_one_request` 时把 `created_time` 塞进 `custom_params["simulation"]`                                      |
| `C_StorageBackendFactory` | `StorageBackendFactory` | `create_backend` 返回 `MockHiCacheStorage`                                                                    |
| `C_HiCacheController`     | `HiCacheController`     | 把异步 prefetch/backup 线程改为同步、按时钟预算切片执行                                                                        |
| `C_HiRadixCacheHook`      | `HiRadixCache`          | `__init__` 用 `MockTokenToKVPoolHost` 替换 host 内存池                                                            |

### 4.2 Hisim 对 AIC 的互联

#### 4.2.1 初始化链路（在 `Scheduler.__init__` 被 Hook 后触发）

```
C_SchedulerHook.wrapped_init(self, ...)
  ├─ 强制 disable_overlap_schedule = True（仿真简化，见 §4.3.5）
  ├─ original_init(...)                          # 跑完 SGLang 原始 Scheduler 初始化
  ├─ ConfigManager.get_model_info(hf_config)     → ModelInfo（解析 HF config.json，见 §4.2.7）
  ├─ ConfigManager.get_accelerator_info()        → AcceleratorInfo（按 config 的 GPU 名查注册表）
  ├─ ConfigManager.get_scheduler_config(...)     → SchedulerConfig（tp/ep/dp/dtype/pd_disagg…）
  ├─ ConfigManager.set_scheduler_config(...) / set_model_info(...)   # 缓存供后续查 KV 字节数
  └─ ConfigManager.get_inference_time_predictor(model, hw, sched)
       └─ AIConfiguratorTimePredictor.__init__(...)
            ├─ _load_perf_database(system=hw.name, backend="sglang", version=backend_version)
            │     └─ aiconfigurator.sdk.perf_database.get_database(...)   # 载入该 GPU 的算子库
            ├─ database._nearest_1d_point_helper 被包装（放开 inner_only，允许外插）
            ├─ get_perf_model(config, model)      # 构建 AIC 内部 perf model（含 context/generation ops）
            ├─ InferenceSession(model, backend, database)
            └─ 若配置了 xgb_model_path：_load_bucket_models(...)  # 载入 decode attention 校正模型
```

> **配置来源澄清**：`hw.name` 在 `get_inference_time_predictor` 中会被 `predictor.device_name` 覆盖（`hw.name = device_name`）。也就是说 **AIC 数据库实际用的是 `predictor.device_name`（如 `"h100_sxm"`、`"rtx_pro_6000_server"`），而非 `platform.accelerator.name`**。后者只用于查硬件规格（HBM 容量、带宽等）。这是一个容易混淆的点。

#### 4.2.2 数据类型映射（DataType → AIC QuantMode）

```
DataType.FP16 / BF16  →  GEMMQuantMode.float16/bfloat16   KVCacheQuantMode.float16   FMHAQuantMode.float16
DataType.FP8          →  GEMMQuantMode.fp8                KVCacheQuantMode.fp8       FMHAQuantMode.fp8
DataType.INT8         →  GEMMQuantMode.int8_wo            KVCacheQuantMode.int8
DataType.FP4          →  GEMMQuantMode.nvfp4 / MoE nvfp4
DataType.INT4         →  GEMMQuantMode.int4_wo
```

- 模型权重/激活用 `data_type` 决定 GEMM/MoE/Comm 模式；
- KV cache 用 `kv_cache_data_type` 决定 KVCache/FMHA 模式（二者可不同，例如权重 FP8 但 KV FP16）。
- **兼容性处理**：旧版 AIC SDK 可能只有 `float16` 没有 `bfloat16`，Hisim 在加载数据库前用 `_install_aic_legacy_dtype_aliases()` 给四个 QuantMode 枚举补上 `bfloat16` 别名（指向 `float16`），并用 `_backfill_aic_system_spec` 补 `float16_tc_flops` 字段，避免新数据库 + 旧 SDK 报错。

#### 4.2.3 预测调用链路（每个 batch 一次）

```
C_SchedulerHook.wrapped_run_batch(batch)
  ├─ 仅当 ret 是 GenerationBatchResult 才处理
  ├─ 构造 HisimScheduleBatch：
  │     若 batch.forward_mode.is_extend():        # prefill
  │         FakeRequest(input_length=req.extend_input_len,
  │                     past_kv_length=len(prefix_indices)+len(output_ids))
  │     若 batch.forward_mode.is_decode():        # decode
  │         FakeRequest(input_length=1,
  │                     past_kv_length=len(prefix_indices)+len(output_ids))
  │
  └─ predicted_latency = INFERENCE_PREDICTOR.predict_infer_time(hisim_batch)   # 秒
        ├─ StateManager.inc_iteration()
        ├─ [BLOCKING 模式] time.sleep(|predicted_latency|)，forward_latency=真实墙钟差
        ├─ [OFFLINE 模式]  forward_latency = predicted_latency（不 sleep）
        └─ StateManager.set_current_inference_dur(forward_latency)
```

> **说明：run_batch 不推进全局时钟**。它只把预测延迟存进 `current_inference_dur`。真正推进时钟在下一步 `process_batch_result` 统一进行（这样能把 L2 load/backup、KV 传输等一起合并计算）。

#### 4.2.4 `predict_infer_time` 内部（AIC 查表 + 校正）

```
predict_infer_time(batch):
  if batch.is_decode():
      isl = int(mean(req.past_kv_length))          # decode 把"历史 KV 平均长度"当 isl
      # —— 可选 XGBoost 校正 decode attention —— 
      if 有 xgb_bucket_models:
          按 batch_size 选分桶模型 → pred_ratio = aic_gen_attn / measured_gen_attn
          gen_attn_scale = 1 / pred_ratio          # 把 AIC 高估的 attention 拉回实测
          RuntimeConfig(batch_size, isl, osl=2, gen_seq_imbalance_correction_scale=gen_attn_scale)
      else:
          RuntimeConfig(batch_size, isl, osl=2)
      summary = session.run_static(cfg, mode="static_gen")
      latency_dict = summary.get_generation_latency_dict()
  else:  # prefill
      isl    = int(mean(past_kv) + mean(input))    # 总序列长
      prefix = int(mean(past_kv))                  # 已复用前缀长
      scale  = ctx_attn_flops_ratio_with_avg(reqs) # 长度不均衡校正（见下）
      if scale >= 0.4:
          RuntimeConfig(batch_size, isl, prefix, osl=1, seq_imbalance_correction_scale=scale)
      else:
          RuntimeConfig(batch_size, isl, prefix, osl=1)
      summary = session.run_static(cfg, mode="static_ctx")
      latency_dict = summary.get_context_latency_dict()

  infer_time = sum(latency_dict.values())          # 各算子延迟相加（ms）
  if is_oom: 警告并返回 -|infer_time|/1e3           # OOM 用负值传递信号
  infer_time *= (decode|prefill)_scale_factor       # 标定系数
  infer_time += _estimate_tp_comm_time_ms(batch)    # 叠加 TP 通信
  return infer_time / 1e3                            # ms → s
```

**`ctx_attn_flops_ratio_with_avg` 说明**：AIC 对 batch 内所有请求用"同一个平均长度"建模 attention，但若 batch 内请求长度差异很大，真实 attention FLOPS 与"用平均值算"会有偏差。该函数计算

```
实际总 attention FLOPS / 用平均长度估的 FLOPS
```

作为 `seq_imbalance_correction_scale` 传给 AIC，让其按比例修正 attention 部分。`< 0.4` 时认为校正不可靠，退回不校正。

#### 4.2.5 为什么 decode 用 osl=2、prefill 用 osl=1

这是 AIC 的接口约定：

- **prefill（`static_ctx`, osl=1）**：prefill 本质是处理 `isl` 个输入 token、产出"第 1 个" token，对应一次 context 前向。`osl=1` 表示"只看这一次 context 计算"。
- **decode（`static_gen`, osl=2）**：decode 要测"生成一个增量 token"的稳态成本。AIC 用 `osl=2` 表示"从已有 `isl` 长度再走 1~2 步生成"，从而取到稳定的单步 decode 延迟（osl=1 在 generation 模式下可能退化）。`isl` 取历史 KV 平均长度，代表当前序列已有多长。

#### 4.2.6 最近邻/插值放宽

```python
db_nearest_1d_point_helper = database._nearest_1d_point_helper
def wrapped_nearest_1d_point_helper(x, values, inner_only=False):
    return db_nearest_1d_point_helper(x, values, inner_only)  # 强制 inner_only=False
database._nearest_1d_point_helper = wrapped_nearest_1d_point_helper
```

AIC 默认 `inner_only=True` 时只在采样点**区间内**插值，超出范围会报错或夹断。Hisim 放开后允许**外插**，使得 trace 中出现数据库未覆盖的超大 batch / 超长序列时仍能给出预测（精度略降但不中断仿真）。

#### 4.2.7 ModelInfo 如何从 HF config 解析

`ModelInfo.from_config(hf_config)` 通过 `model_config_mapping_dict` 把不同模型家族的字段名归一化，例如：

```
num_hidden_layers ← n_layers / num_layers / n_layer
num_attention_heads ← n_head / num_heads / attention_heads
num_key_value_heads ← n_head_kv / num_kv_heads / multi_query_group_num
max_seq_len ← max_position_embeddings / seq_length / n_positions
# MoE / MLA 字段
n_routed_experts ← num_experts / num_local_experts
kv_lora_rank, qk_rope_head_dim …（DeepSeek MLA）
```

这让 Hisim 能统一支持 qwen/llama/chatglm（标准 MHA）、deepseek/kimi（MLA）、qwen3_moe（MoE）等不同架构。`get_perf_model` 再据 `model_type` 把这些字段填进 AIC 的 `SupportedModels`。

#### 4.2.8 TP 通信开销估算

```python
def _estimate_tp_comm_time_ms(self, batch):
    if platform_config is None or tp_size <= 1: return 0.0
    collective_count = 2 * num_layers                  # 每层 2 次 allreduce（attn后 + FFN后）
    tokens = batch.batch_size if decode else batch.num_context_tokens
    payload_bytes = tokens * hidden_size * dtype_bytes
    # ring-allreduce：每参与方收发 2(N-1)/N 份数据
    transfer_ms = payload_bytes * 2*(N-1)/N / bandwidth * 1e3
    launch_ms   = 2*(N-1) * latency_us / 1e3
    total = collective_count * (transfer_ms + launch_ms)
```

- 区分**节点内**（intra-node，通常 NVLink）和**节点间**（inter-node，IB/RoCE/Ethernet）链路；当 `tp_size` 跨节点时分两段叠加（先节点间、再节点内）。
- interconnect 默认参数：

| 模式 | 默认延迟(us) | 带宽效率 |
|------|------------|---------|
| none | 0 | 100% |
| nvlink | 1 | 100% |
| pcie | 3 | 90% |
| ib / infiniband | 5 | 90% |
| roce | 7 | 85% |
| ethernet | 10 | 80% |

- 带宽可由 `platform.interconnect_bandwidth_gb` 指定；未指定时回退到 `AcceleratorInfo` 的 `intra_node_bandwidth_gb` / `inter_node_bandwidth_gb`。

### 4.3 真实 Workload 推理逻辑（完整流程）

#### 4.3.1 数据准备：真实 trace 格式

`.jsonl`，每行一个请求：

```json
{"rid":"21e5xx","timestamp":732.31,"output_length":1024,"input_length":1024,
 "input_ids":[925,3911,...],"output_ids":[244,129,...],
 "queue_end":7522.45,"final_prefix_cache_len":0}
```

`HisimCollectionDataset._load_dataset()`：

1. 读取所有行；
2. 时间字段兼容：优先 `created_time`，否则用 `timestamp`；
3. 找全局最小时间戳，把所有请求时间**对齐到 0**（保持相对到达间隔）；
4. 每条转 `GenericRequest(token_ids=input_ids, input_length, output_length, custom_params={"created_time": ts-min})`。

> **说明**：`input_ids` 的具体取值会影响**前缀缓存命中**（相同前缀的请求会命中 radix cache），进而影响 `past_kv_length` 与 prefill 工作量。所以真实 token id 不是可有可无的——它决定了缓存复用行为。

#### 4.3.2 请求发送：bench_serving.py 的模拟并发

`--bench-mode simulation` 时，每个请求的 `sampling_params.custom_params` 携带：

```json
{"simulation": {"enabled": true,
                "created_time": <对齐后时间>,
                "queue_start": <可选，调试对齐用>,
                "total_request": <本次总请求数 N>}}
```

客户端异步并发把 N 个请求全部 POST 出去（不等结果），最后等所有响应、再 POST `/profile` 触发指标输出。`total_request` 让服务端知道"收齐多少请求后才能开始仿真"。

#### 4.3.3 两种仿真模式的本质区别

| | BLOCKING | OFFLINE（默认）|
|--|----------|----------------|
| 时间来源 | **真实墙钟**（`time.time()`）| **虚拟时钟**（`StateManager.global_clock`）|
| 推理延迟实现 | `time.sleep(predicted_latency)` 真的睡 | 直接把 `predicted_latency` 加到虚拟时钟 |
| 请求到达 | 按真实到达即时处理 | 全部收齐后按 `created_time` 用最小堆回放 |
| 速度 | 与被仿真系统同速（慢）| 远快于实时（无需真实等待）|
| 适用 | 实时联调、和真实客户端交互 | 大规模 trace 批量回放（主用）|
| 环境变量 | `HISIM_SIMULATION_MODE=BLOCKING` | `HISIM_SIMULATION_MODE=OFFLINE`（默认）|

> 两者最本质的差异在于：BLOCKING 通过 `time.sleep` 真实等待，而 OFFLINE 把"等待"变成虚拟时钟上的"加法"。因此 5 个请求 1024+1024 的 demo 能在几秒内跑完，而被仿真的真实延迟可能是几十秒。

#### 4.3.4 OFFLINE 仿真主循环（逐步）

```
阶段 1：收齐请求
  wrapped_recv_requests()：
    time.sleep(0.05) 等一批请求到达 → 逐个提取 simulation 参数
    enqueue_time = queue_start or created_time or server_created_time
    放入 FUTURE_QUEUE：(enqueue_time, time_ns()做salt, req)   # salt 防止 req 不可比较
    当 len(FUTURE_QUEUE) == total_request：
        OFFLINE_RECV_ALL_REQUEST = True
        heapq.heapify(FUTURE_QUEUE)                         # 转最小堆（按 enqueue_time）
    # /flush_cache /profile 等非生成请求直接放行（extra_requests）

阶段 2：仿真推进（event_loop 每次迭代）
  ① recv_requests：
       current = global_clock
       把 FUTURE_QUEUE 中 enqueue_time ≤ current 的请求出队 → SGLang waiting_queue
       （首个请求进来时：LAST_CPU_TS=now，global_clock 置 0）

  ② get_new_batch_prefill：
       SGLang 原始逻辑按调度策略（默认 FCFS）组 batch、做前缀缓存匹配
       对新入 batch 的每个 req：记录 final_reused_tokens=cached_tokens
                                  queue_end = global_clock（首次）
       特殊情形：
         - new_batch=None 且 running 空 且 waiting 非空 → step_clock(0.005)（等 prefetch）
         - new_batch=None 且 running 空 且 FUTURE_QUEUE 非空（OFFLINE）
              → global_clock 跳到下一个请求的 enqueue_time + 1e-6（快进空闲期）

  ③ run_batch：构造 HisimScheduleBatch → AIC 预测 → set_current_inference_dur

  ④ process_batch_result：
       l2_load   = pop_hicache_l2_load_dur()
       l2_backup = pop_hicache_l2_backup_dur()
       infer     = get_current_inference_dur()
       [默认 非 overlap]：
         step_clock(l2_load + infer + l2_backup)
         response_time = global_clock
       [overlap]（当前不会走到，见 §4.3.5）：
         step_clock(max(l2_load - last_infer, 0)); step_clock(infer)
         response_time = global_clock + l2_backup
       for req in batch.reqs（仅 is_chunked==0）：
         gen_token_latencies.append(response_time - last_event_time)
         last_event_time = response_time
       [PD 分离 + prefill]：step_clock(kv_transfer_dur)   # §3.2.2
       记录 ITERATION_STATS（含 forward/l2_load/l2_backup/kv_transfer 分解）

循环直到 FUTURE_QUEUE 与所有 batch 处理完
```

#### 4.3.5 关于 overlap schedule 的说明

`wrapped_init` 里强制 `disable_overlap_schedule = True`，且 `event_loop_overlap` 被改写为直接调 `event_loop_normal`。原因：overlap（计算与调度重叠）会让时序大幅复杂化，仿真为简化把它关掉。`OVERLAP_SCHEDULE` 标志虽保留并影响 `process_batch_result` 的两分支，但在默认运行下走的是**非 overlap 分支**。

#### 4.3.6 时钟与请求统计的对应

```
global_clock 轴：

 0   q0      q1            p0           p0+kv      p0+kv+d0     p0+kv+d0+d1
 │   │       │              │            │           │             │
 │   │──wait→│──[prefill]──→│──[KV tx]──→│──[decode0]→│──[decode1]─→ ...
 │   │       │              │            │           │             │
created  queue_end      TTFT=p0−q0    （PD才有）   ITL[0]        ITL[1]
_time   =global_clock  →记入[0]，并    step_clock  =kv+d0        =d1
        （首次入batch） 更新last_event  (kv_tx)    →记入[1]      →记入[2]
```

`gen_token_latencies` 语义：

- `[0]` = TTFT = prefill 完成时刻 − 被调度时刻（`last_event_time` 初值=created_time / queue_start）
- `[1]` = 首个 decode token 延迟（PD 模式含 KV 传输）
- `[2..N-1]` = 后续每个 decode token 延迟

#### 4.3.7 指标计算（`calc_metrics`）

```
对每个完成的请求 req：
  ttft  = gen_token_latencies[0]
  tpot  = mean(gen_token_latencies[1:])         # 若 output_len>1
  itl  += gen_token_latencies[1:]                # 展开收集所有 token 间延迟
  e2e   = sum(gen_token_latencies)
  queue_dur = queue_end - queue_start
  total_dur_s = max over reqs of last_event_time # 仿真总时长 = 最后事件时刻
吞吐：
  request_throughput = num_requests / total_dur_s
  input_throughput   = Σinput / total_dur_s
  output_throughput  = Σoutput / total_dur_s
缓存：
  prefix_cache_reused_ratio = Σfinal_reused_tokens / Σinput
  disk_prefetch_ratio       = Σprefetch_complete_tokens / Σinput
```

随后对 ttft/tpot/itl/e2e 各算 mean/median/std/p90/p95/p99 并 ×1000（转 ms）。warmup 请求（`HISIM_NUM_WARMUP`，默认 0）会从指标统计中剔除（但仍参与时间对齐）。

#### 4.3.8 HiCache 分层缓存延迟仿真（L2 prefetch / backup）

开启 HiCache 后（HBM=L1 → DRAM=L2 → Disk=L3），`C_HiCacheController` 把异步线程改为**同步、按时钟预算切片执行**：

**L2 Prefetch（Disk → DRAM），`handle_prefetch_operation`：**

```
remain_dur = current_inference_dur            # 用本轮推理时间做"可重叠预算"
（先续上次未完成的 chunked_prefetch_operation）
while remain_dur > 0 且 prefetch_queue 非空：
    hash_value, storage_hit_count = _storage_hit_query(op)   # 查 Mock 存储命中
    if storage_hit_count < prefetch_threshold: 放弃该 op
    completed, dur = calc_prefetch_pages(need_pages, page_bytes, remain_dur, disk_read_bw)
        # dur = need_pages*page_bytes/bw；若 dur>remain_dur 则只完成 remain_dur*bw/page_bytes 页
    if 未完成：记录 chunked_prefetch_operation，remain_dur=0（下轮继续）
    else：op.mark_terminate()；remain_dur -= dur
    req_stats.prefetch_complete_tokens = op.completed_tokens
→ 命中量越大、磁盘越慢，预取延迟越大，叠加进 l2_load_dur
```

`calc_prefetch_pages` 体现"**预取与计算重叠**"：能在本轮推理时间窗口内传完的部分不额外加时间；传不完的部分按磁盘带宽折算成额外延迟，并在后续轮次继续。

**L2 Backup（DRAM → Disk）**：当前实现追踪 backup 操作但**按即时完成处理**（`inc_hicache_l2_backup_dur` 记录，未做精确带宽建模），属于可改进点。

#### 4.3.9 KVCM 存储后端集成与 Instance 隔离

`MockHiCacheStorage` 有两种后端：

- 若能 `import kv_cache_manager.optimizer.pybind.kvcm_py_optimizer` → 走 **KVCM 优化器**（真实的 KV cache 管理器），`WriteCache/读路径` 通过 `OptimizerManager` 执行；
- 否则 → 退化为**纯文件集合**（`/tmp/hisim/hicache/storage_keys.txt` 记录命中 key 集合），只判存在性。

> **对接仓库级约束（见 AGENTS.md）**：KVCache 仅在**同一个 `instance_id`** 内复用，跨 Instance 不匹配。当前 Mock 用单一共享 `instance_id = "3780643326877293460"`（代码注释明确 "Multi-instance not supported yet"）。这意味着当前仿真**不建模多 Instance 间的缓存隔离**——如果未来要仿真多实例场景，需要为不同实例分配不同 `instance_id` 并保证跨实例不命中。

`HISIM_RESET_HICACHE_STORAGE=1` 可在启动时清空存储 key，用于多次实验间重置缓存状态。

#### 4.3.10 KV Cache 容量估算（未指定 max_total_tokens 时）

```python
def estimate_kv_cache_pool_capacity(model, device, scheduler_config):
    perf_model = get_perf_model(scheduler_config, model)
    weights = sum(op.get_weights() for op in perf_model.context_ops) / pp_size  # 单卡权重
    rest = (mem_fraction_static * device.hbm_capacity_gb - 1.4GB)*(1<<30) - weights
    per_token = kv_cache_cell_elems(model, tp, pp) * kv_cache_data_type.bytes
    return int(rest / per_token)
```

每 token KV 元素数：

```
标准 MHA: max(num_kv_heads//tp, 1) * head_dim * (num_layers//pp) * 2(K+V)
MLA(DeepSeek): (kv_lora_rank + qk_rope_head_dim) * (num_layers//pp)
```

容量直接决定能并发多少请求（`max_running_requests`、能 batch 多大），从而间接影响排队与吞吐——所以它虽是"内存"参数，却深刻影响时延仿真结果。

### 4.4 完整数据流图

```
trace.jsonl ──► bench_serving.py ──► SGLang HTTP Server（被 Hook）
                  │                        │
                  │ POST /generate         ▼ TokenizerManager(Hook)：注入 server_created_time
                  │ {simulation:{          │
                  │   created_time,        ▼ Scheduler(Hook) ── OFFLINE 仿真循环 ──────────────┐
                  │   total_request}}      │  FUTURE_QUEUE(最小堆，按 created_time)            │
                  │                        │        │ 按 global_clock 出队                      │
                  │                        │        ▼                                          │
                  │                        │  waiting_queue → get_new_batch_prefill            │
                  │                        │        │（FCFS + 前缀缓存匹配）                    │
                  │                        │        ▼                                          │
                  │                        │  Prefill batch(il>1) / Decode batch(il=1)         │
                  │                        │        │ run_batch                                 │
                  │                        │        ▼ HisimScheduleBatch[FakeRequest(il,kv)…]   │
                  │                        │        ▼ AIConfiguratorTimePredictor               │
                  │                        │           prefill: run_static(static_ctx)          │
                  │                        │           decode : run_static(static_gen)+XGB校正  │
                  │                        │           +TP通信 ×scale → predicted_latency(s)     │
                  │                        │        ▼ process_batch_result                       │
                  │                        │           global_clock += l2_load+infer+l2_backup  │
                  │                        │           记录 gen_token_latencies                  │
                  │                        │           [PD] global_clock += kv_transfer_dur      │
                  │                        └───────────── 循环至全部完成 ──────────────────────┘
                  │                                 │
                  └── 等全部响应 → POST /profile ───► wrapped_profile：calc_metrics
                                                       ├─ metrics.json   （TTFT/TPOT/ITL/吞吐/KV传输）
                                                       ├─ request.jsonl  （逐请求统计）
                                                       └─ iteration.jsonl（逐迭代延迟分解）
```

---

## 五、常见疑问 FAQ

**Q1：Hisim 既然不算 GPU，为什么还要跑完整的 SGLang？**
因为时延高度依赖**调度行为**（batching、排队、抢占）和**缓存命中**（radix prefix cache、HiCache）。这些逻辑极其复杂，重写易失真。Hisim 让 SGLang 原样跑这些逻辑，只把"算子计算耗时"换成 AIC 预测，从而高保真。

**Q2：AIC 预测的是"一个 batch"还是"一个请求"的延迟？**
是**整个 batch 一次前向**的延迟。Hisim 把 batch 内请求聚合成平均的 `isl/prefix`，再用 `seq_imbalance_correction_scale` 修正长度不均衡。

**Q3：TTFT 为什么等于 `gen_token_latencies[0]`？**
prefill 完成 = 首 token 产出。该轮 `process_batch_result` 记录 `response_time - last_event_time`，而 `last_event_time` 初值是请求创建/入队时刻，故 `[0]` 正好是首 token 端到端等待 = TTFT。

**Q4：`device_name`（predictor）和 `accelerator.name`（platform）有何区别？**
`device_name` 决定**用哪个 AIC 算子数据库**（影响延迟预测）；`accelerator.name` 决定**硬件规格**（HBM 容量、互联带宽，影响容量估算与 TP 通信）。二者可以是同一张卡的不同标识。

**Q5：合成 workload（random）和真实 trace 在推理逻辑上有何不同？**
仅"请求从哪来"不同：random 由 bench_serving 按 `--request-rate` 现场生成 token，trace 由 `HisimCollectionDataset` 回放真实 `input_ids` 与到达时间。进入 Scheduler 之后的仿真逻辑完全一致。

**Q6：OOM 怎么处理？**
`predict_infer_time` 检测到 AIC `summary.check_oom()` 为真时，返回**负延迟**（`-|infer_time|/1e3`）作为信号并打 warning，避免直接崩溃，便于上层识别该配置不可行。

**Q7：KV Transfer 延迟到底在哪里计算并注入？**
在 `C_SchedulerHook.wrapped_process_batch_result` 中。只有满足 `pd_disagg_enabled=True`、当前 batch 是 prefill、且请求完成最终 prefill chunk 时，才会统计 `final_prefill_tokens`，按 `tokens × kv_bytes_per_token / bandwidth` 计算 `kv_transfer_dur`，再执行 `StateManager.step_global_clock(kv_transfer_dur)`。它被放在请求统计之后，因此**不计入 TTFT**，而是吸收到第一个 decode 的 ITL 里。

**Q8：`wrapped_process_batch_result` 主要做什么？**
它是一次 batch 收尾时的核心仿真入口：弹出 HiCache L2 load/backup 时延，结合当前 batch 的推理时延推进虚拟时钟，记录请求级 `gen_token_latencies`，按需注入 PD KV transfer 延迟，并把本轮的前向/L2/KV transfer 信息写入 `ITERATION_STATS`。

**Q9：`ctx_attn_flops_ratio_with_avg(reqs)` 到底返回什么？为什么不直接跑正式推理系统去拿 FLOPS？**
它返回的不是硬件实测 FLOPS，而是一个**长度不均衡校正系数**：
`实际 attention 工作量 / 用平均长度估出来的 attention 工作量`。单请求工作量近似写成 `(2 × past_kv_length + input_length) × input_length / 2`，只依赖序列长度。这样做的目的不是复现真实 kernel 计数，而是修正 AIC 用平均长度建模 prefill attention 带来的误差，因此用解析式即可，不需要真的把模型跑起来。

**Q10：`get_inference_time_predictor` 和 `estimate_kv_cache_pool_capacity` 分别负责什么？**
`get_inference_time_predictor` 是预测器工厂：读取 JSON 里的 `predictor` 配置，构造具体的 `AIConfiguratorTimePredictor`。`estimate_kv_cache_pool_capacity` 是容量估算器：先估单卡权重占用，再扣除框架预留和静态显存比例限制，最后用剩余显存除以单 token KV 开销，得到默认的 `max_total_num_tokens`。

**Q11：为什么 `tp_comm_ms` 只补 TP，不补 PP/EP/DP？**
因为在当前集成方式里，batch 前向时延主要由 AIC 的静态性能模型返回，而**TP 通信**是最明确从真实执行路径中被拿掉、又最影响推理时延的一块，所以 Hisim 显式补了一份 ring collective 时间。PP/EP 更多依赖 AIC 的模型骨架和 perf database 建模；DP 在 serving 场景下通常也没有训练式梯度同步那种固定通信开销。

**Q12：说“SGLang 没真正开 TP / 被替换成 Mock 版本”是什么意思？从哪里体现出来？**
意思是：Hisim 不是让 SGLang 真的加载完整模型并执行多卡 TP forward，而是保留 SGLang 的调度/缓存外壳，把 `ModelRunner` 的关键方法在运行时改绑到 mock 实现上。典型体现包括：`initialize` 里 `self.model = MockModel()`；KV pool 改为 `MockTokenToKVPool`；`forward` 只返回空 logits；`sample` 直接返回全 1 token；真正的 batch 时延来自 `INFERENCE_PREDICTOR.predict_infer_time(...)`，而不是 GPU 前向执行。

**Q13：既然会 OOM，为什么不自动把更早的上下文裁掉，像滑动窗口那样继续跑？SGLang 本体会这样做吗？**
当前 Hisim 这条链没有实现这种“超限后自动降级”的策略。`check_oom()` 只负责判定超显存并返回 OOM 标志，Hisim 只把它转换成 warning 和负延迟信号；不会自动把 `past_kv_length` 改成更小的窗口值。并且当前仿真链是按完整历史上下文记账的：`past_kv_length` 直接取 `len(prefix_indices) + len(output_ids)`，没有做 `min(window_size, past_kv_length)` 之类的裁剪。

SGLang 本体**有**滑动窗口注意力（Sliding Window Attention / hybrid SWA）支持，但这是**模型级能力**，不是“任何模型 OOM 了就自动裁剪上下文”的通用兜底。只有模型配置显式带 `sliding_window`、SGLang 识别为 `is_hybrid_swa` 后，才会启用对应的 attention/window 逻辑和 SWA 专用 KV allocator。当前 Hisim 代码虽然在 `ModelInfo` 里保留了 `attn_sliding_window` 字段，但这条仿真链并没有消费它，因此仍按完整上下文建模，超限时直接打 OOM 标志。

---

## 六、总结

> 下表从组件职责角度汇总全文要点。

| 组件                           | 职责                                                                                            |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| **AIC（黑盒视角）**                | 硬件 profiling 算子库；给定 `batch_size/isl/osl/prefix` 返回各算子延迟分解                                     |
| **AIC（白盒视角）**                | PerfDatabase（CSV 实测表）+ SOL roofline 公式（算力/访存取大）+ 插值/外插；算子组成 context_ops/generation_ops（详见第二章） |
| **AIC SystemSpec**           | 从 `systems/*.yaml` 提供 flops/带宽/拓扑，是 SOL 公式与 TP 通信的物理参数源                                       |
| **Hook 层（class_hook_entry）** | 覆盖 `__build_class__`，在类定义时注入补丁，零侵入接管 SGLang                                                   |
| **C_SchedulerHook**          | 仿真核心：请求回放、batch 构造、调 AIC、推进时钟、记录指标                                                            |
| **C_ModelRunnerHook**        | 旁路真实 GPU 计算（forward 返回空 logits、sample 返回全 1）                                                  |
| **StateManager**             | 维护全局虚拟时钟与各类 duration，驱动 OFFLINE 时间线                                                           |
| **ConfigManager**            | 解析三段 JSON 配置，构建 ModelInfo/AcceleratorInfo/SchedulerConfig 与 AIC 预测器                           |
| **PD disagg**                | AIC 不建模 KV 传输（有 TODO）；Hisim 自行注入 `tokens×kv_bytes/带宽`，吸收进 ITL[0]、不计入 TTFT                     |
| **HiCache + KVCM**           | 仿真 L2 prefetch 延迟（与计算重叠的预算切片）；存储后端可对接 KVCM 优化器（单 instance）                                    |

> 一句话总结：**AIC 负责"算得多快"（算子级 SOL/实测延迟），Hisim 负责"传得多慢 + 何时发生"（调度时间线 + PD KV 传输注入）——二者互补，缺一不可。**

