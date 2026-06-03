# Hisim × AIConfigurator 技术解析

---

## 一、Hisim 整体架构

### 1.1 背景与目标

在大语言模型（LLM）推理服务规模化落地的背景下，首 Token 延迟（TTFT）、每输出 Token 延迟（TPOT）、系统吞吐量等关键指标，高度依赖于模型结构、硬件平台、推理引擎及其运行时配置的复杂耦合关系。传统基于真实 GPU 集群的端到端压测成本高昂、周期漫长，难以高效探索海量配置组合。

**Hisim**（High-fidelity Inference Simulator）是基于 CPU 的高保真推理仿真工具，通过回放真实或代表性的推理 Workload Trace，在通用 CPU 上快速、低成本地预测不同模型、目标硬件、推理引擎及配置下的关键性能指标，从而支撑推理系统的高效设计与优化。

### 1.2 核心设计：框架劫持（Framework Hooking）

Hisim 的核心机制是**非侵入式框架劫持**：通过覆盖 Python 内置的 `builtins.__build_class__` 函数，在目标框架（如 SGLang）加载类定义时动态注入 Hook，替换关键方法，从而拦截真实 GPU 推理的执行流程，转而使用仿真逻辑完成计算。

**劫持流程概述：**

```
启动仿真服务
    │
    ├─ install_class_hooks()          # 覆盖 __build_class__
    │
    ├─ SGLang 正常加载
    │       │
    │       ├─ class ModelRunner      # 触发 Hook → 替换 initialize / forward / sample
    │       ├─ class Scheduler        # 触发 Hook → 替换 run_batch / process_batch_result
    │       ├─ class HiCacheController # 触发 Hook → 替换预取/备份逻辑
    │       └─ ...
    │
    └─ 推理请求到达 → 使用仿真逻辑计算延迟，驱动全局虚拟时钟前进
```

### 1.3 仿真流程

每一个 Batch 的仿真执行步骤如下：

1. **Scheduler 调度请求** — SGLang 原生调度逻辑保持不变（Prefix Cache 命中、请求排队等均真实执行）
2. **ModelRunner.forward 被替换** — 不执行真实 GPU 计算，直接返回空张量占位
3. **`run_batch` 调用 TimePredictor** — 根据当前 Batch 的属性（batch_size、ISL、prefix 长度等）查询 AIConfigurator，获取预测推理延迟
4. **虚拟时钟推进** — `process_batch_result` 根据预测延迟和 KV Cache I/O 时间更新全局虚拟时钟，并记录各请求的时间戳
5. **指标计算** — 仿真结束后，统计 TTFT、TPOT、ITL、吞吐量等指标

---

## 二、AIConfigurator 延迟预测机制

### 2.1 项目简介

**AIConfigurator（AIC）** 是 NVIDIA 开源的 LLM 推理性能预测工具（[项目地址](https://github.com/ai-dynamo/aiconfigurator)），通过查询预先收集的算子性能数据库，实现对 Prefill 和 Decode 阶段推理延迟的快速、准确预测，无需实际运行 GPU 计算。

### 2.2 核心组件

| 组件 | 作用 |
|------|------|
| `PerfDatabase` | 算子性能数据库，存储不同硬件/量化配置下各算子（GEMM、Attention 等）的实测延迟数据 |
| `InferenceSession` | 推理会话，封装 Model + Backend + Database，提供 `run_static()` 接口 |
| `RuntimeConfig` | 单次推理请求的运行时参数（batch_size、ISL、OSL、prefix 等） |
| `SystemSpec` | 硬件规格 YAML 文件，描述 GPU 计算/内存带宽等参数 |
| `DatabaseMode` | 查询模式，支持 `SILICON`（实测值）、`SOL`（Roofline 上界）等 |

### 2.3 延迟预测接口

Hisim 通过 `AIConfiguratorTimePredictor` 类调用 AIC，核心接口如下：

**Prefill 阶段：**
```python
runtime_config = RuntimeConfig(
    batch_size=batch.batch_size,
    isl=mean_past + mean_input,   # 平均序列总长（历史 KV + 新输入）
    prefix=mean_past,              # 平均 Prefix 长度（已命中缓存部分）
    osl=1,                         # Prefill 输出固定为 1 token
)
summary = session.run_static(runtime_config, mode="static_ctx")
latency_ms = sum(summary.get_context_latency_dict().values())
```

**Decode 阶段：**
```python
runtime_config = RuntimeConfig(
    batch_size=batch.batch_size,
    isl=mean_past_kv_length,  # 平均历史 KV 长度
    osl=2,                     # Decode 固定输入参数
)
summary = session.run_static(runtime_config, mode="static_gen")
latency_ms = sum(summary.get_generation_latency_dict().values())
```

### 2.4 序列不均衡修正

当 Batch 内各请求序列长度差异较大时，直接使用均值会低估注意力计算延迟。Hisim 引入**序列不均衡修正系数**（`seq_imbalance_correction_scale`）进行补偿：

```
correction_scale = Σ actual_flops_per_req / (N × avg_flops_per_req)
```

该系数反映了实际注意力计算量相对于均值假设的比例，当其超过阈值（0.4）时生效。

### 2.5 XGBoost 增强（可选）

对于 Decode 阶段，可配置 XGBoost 模型对 AIC 的 Generation Attention 延迟预测进行修正，进一步提升长序列场景下的预测精度：

```python
# 加载 XGBoost 模型，按 batch_size 分 Bucket 选择
pred_ratio = xgb_model.predict(features)       # 预测 AIC/实测 比值
gen_attn_scale = 1.0 / pred_ratio              # 反向修正
```

---

## 三、Hisim 与 AIConfigurator 的集成

### 3.1 初始化流程

仿真服务启动时，`ConfigManager` 读取配置文件（`sim-config-path`），完成以下初始化：

```
配置文件 (JSON)
    ├── platform     → AcceleratorInfo（硬件规格）
    ├── predictor    → AIConfiguratorTimePredictor（延迟预测器）
    │       ├── database_path     # 算子数据包路径
    │       ├── device_name       # AIC 内部硬件名称（如 h20_sxm）
    │       ├── prefill_scale_factor  # Prefill 延迟校准系数
    │       └── decode_scale_factor   # Decode 延迟校准系数
    └── scheduler    → SchedulerConfig（并行策略、数据类型等）
```

> **注意**：配置文件中 `predictor.device_name` 决定 AIC 数据库的选择，`platform.accelerator.name` 仅用于推算 KV Cache 容量。

### 3.2 推理延迟计算

每次 `run_batch` 时，Hook 将当前 Batch 转换为 Hisim 的 `ScheduleBatch` 数据结构，调用 `TimePredictor.predict_infer_time()`，流程如下：

```
run_batch(batch)
    │
    ├─ 判断 Prefill or Decode
    │
    ├─ 构造 RuntimeConfig（batch_size、ISL、prefix 等）
    │
    ├─ InferenceSession.run_static()   →  查询 PerfDatabase
    │
    ├─ 汇总各算子延迟
    │
    └─ × scale_factor  →  返回最终预测延迟（秒）
```

### 3.3 KV Cache 层级仿真

Hisim 支持多级 KV Cache（L1/L2/L3）的带宽仿真：

| 层级 | 对应存储 | 仿真方式 |
|------|----------|----------|
| L1 | GPU HBM | 由 AIC 的 Prefill/Decode 延迟隐含覆盖 |
| L2 | Host DRAM | 按配置的 `memory_*_bandwidth_gb` 计算 I/O 时间 |
| L3 | SSD/磁盘 | 按配置的 `disk_*_bandwidth_gb` 计算 I/O 时间 |

L2/L3 的 I/O 时间叠加到推理延迟上（支持调度重叠优化），最终驱动虚拟时钟前进。

---

## 四、PD 分离场景

### 4.1 场景说明

PD 分离（Prefill-Decode Disaggregation）是将 Prefill 和 Decode 阶段部署在独立节点上的架构优化方案，可有效避免两者对 GPU 资源的竞争，提升整体吞吐量和尾延迟表现。

### 4.2 Hisim 的 PD 分离支持

当前 Hisim 的 PD 分离仿真通过 `bench_serving` 的 `--pd-separated` 参数启用，支持对 Prefill 节点和 Decode 节点分别进行压测。

**关于 KV 传输延迟**：AIC 本身不对 Prefill→Decode 之间的 KV Cache 传输延迟建模；在 PD 分离场景下，如需精确仿真网络传输开销，需在配置层面单独注入传输延迟参数。

---

## 五、精度验证

### 5.1 验证方法

采用平均绝对百分比误差（**MAPE**，Mean Absolute Percentage Error）评估仿真精度，对比仿真指标与真实 GPU 集群实测值。

### 5.2 验证结果

在 H20 上对 Qwen3-8B 和 Qwen3-32B-FP8 的三种 KV Cache 命中场景（无缓存 / L1 / L2）进行验证，**各指标预测误差均在 5% 以内**：

| 模型 | 场景 | Mean TTFT | Mean TPOT | Mean ITL |
|------|------|:---------:|:---------:|:--------:|
| Qwen3-8B | no_cache | 3.42% | 1.64% | 1.78% |
| Qwen3-8B | L1 | 4.15% | 3.47% | 3.54% |
| Qwen3-8B | L2 | 2.38% | 4.05% | 4.07% |
| Qwen3-32B-FP8 | no_cache | 2.77% | 0.58% | 0.52% |
| Qwen3-32B-FP8 | L1 | 2.40% | 1.03% | 1.02% |
| Qwen3-32B-FP8 | L2 | 3.05% | 1.11% | 1.02% |

**场景定义：**
- `no_cache`：无缓存命中，全量 Prefill
- `L1`：仅 GPU HBM（L1）命中
- `L2`：GPU HBM（L1）与 Host DRAM（L2）同时命中

---

## 六、总结

Hisim 通过非侵入式框架劫持机制，将 AIConfigurator 的算子级延迟预测能力无缝集成到 SGLang 推理框架中，实现了在 CPU 上对真实 GPU 推理行为的高保真仿真。其核心价值在于：

- **低成本**：无需 GPU 资源即可完成仿真，显著降低配置探索成本
- **高保真**：保留了 SGLang 原生的调度、Prefix Cache、KV Cache 分层等核心逻辑
- **高精度**：借助 AIConfigurator 的精确算子数据库，配合序列不均衡修正，各指标误差控制在 5% 以内
- **可扩展**：通过 Hook 机制，可灵活支持更多推理框架版本和硬件平台
