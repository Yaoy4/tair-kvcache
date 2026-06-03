# Hisim_AIC 文档修改记录（Changelog）

> 本文件记录围绕 **Hisim ↔ AIConfigurator（AIC）集成文档** 的历次修改，按"修改次"分组，便于追溯每一轮的目标、产物与改动要点。
>
> 涉及目录：`C:\Gitrepo\tair-kvcache-demo\hisim\docs\`
> 参考源码：
> - Hisim：`C:\Gitrepo\tair-kvcache-demo\hisim\src\hisim\`
> - AIC：`C:\Gitrepo\aiconfigurator\src\aiconfigurator\sdk\`

---

## 文件总览（当前状态）

| 文件 | 大小 | 角色 |
|------|------|------|
| `Hisim_AIC.md` | ~30 KB | 用户提供的源文档 / 基线（保持不变） |
| `Hisim_AIC_detailed.md` | ~65 KB（1031 行） | **主交付物**：在基线之上增补细节、澄清、AIC 内部机制 |
| `Hisim_AIC_changelog.md` | 本文件 | 历次修改记录 |

> 说明：第 1 次产出的 `hisim_aic_pd_deep_dive.md` 后续被用户整理为 `Hisim_AIC.md`，故当前目录中已无该独立文件。

---

## 第 1 次修改：创建首版深度文档（回答 3 个问题）

**用户诉求**
回答三个问题，并要求"非常详细"：
1. Hisim 和 AIC 的整体逻辑是什么？
2. 针对 PD 分离场景，Hisim 需要修改什么？
3. Hisim 对 AIC 的互联是什么？Hisim 对真实 workload 的推理逻辑是什么？

**主要动作**
- 通读 Hisim 代码：`simulation/sglang/`（`sglang_hook.py` 等）、`time_predictor/`（`aiconfigurator.py`、`base.py`）、`spec/`、`dataset/`、`hook/`。
- 重点阅读 `sglang_hook.py`（~1066 行核心 Hook）、`aiconfigurator.py`、`config.py`、`types.py`、`state.py`、`launch_server.py`、`utils.py`。

**产物**
- 创建 `hisim_aic_pd_deep_dive.md`（~23 KB）。

**覆盖要点**
- Hisim 通过 monkey-patch"寄生"SGLang，保留真实调度/缓存，仅把 GPU 计算替换为 AIC 延迟预测。
- PD 分离修改点（`pd_disagg_enabled` / `pd_kv_transfer_bandwidth_gb`、KV 传输延迟注入）。
- Hisim↔AIC 互联（`device_name` 决定加载哪个 AIC 数据库）与真实 workload 推理逻辑。
- 含 ASCII 架构图、代码片段、数据流图。

---

## 第 2 次修改：增补细节 + 澄清，形成 `Hisim_AIC_detailed.md`

**用户诉求**
对 `Hisim_AIC.md` 增加更多细节，对表达不清楚处增加新描述，形成一份**新文件**。

**主要动作**
- 阅读现有 `Hisim_AIC.md`（~29.5 KB，611 行）。
- 为保证准确性，补读源码：`class_hook_entry.py`（`__build_class__` 覆写机制）、`env.py`、`sim_args.py`、`sglang_mock_class.py`（MockHiCacheStorage + KVCM）、`model/base.py`（字段映射）。

**产物**
- 创建 `Hisim_AIC_detailed.md`（初版 ~45 KB，760 行）。

**新增/澄清要点**
- **§0 术语表**：统一 TTFT / ITL / ISL / OSL / PD / SOL 等。
- **§2.3 PD 数值示例**：用具体数字演示 KV 传输延迟如何被吸收进 ITL[0]。
- **§四 FAQ**：常见易混淆点集中答疑。
- 关键澄清：
  - Hook 通过覆写 Python 内建 `__build_class__` 实现，必须在 SGLang 类导入**之前**安装。
  - 时钟推进发生在 `wrapped_process_batch_result`（非 `wrapped_run_batch`）。
  - `device_name` 决定 AIC 数据库；`platform.accelerator.name` 仅决定硬件规格（二者常被混淆）。

---

## 第 3 次修改：基于 AIC 源码增补"AIC 内部机制"

**用户诉求**
依据真实仓库 `C:\Gitrepo\aiconfigurator`，为 `Hisim_AIC_detailed.md` 增补 AIC 的详细内部机制。

**主要动作**
- 探查 `aiconfigurator/src/aiconfigurator/sdk/` 结构。
- 阅读 `inference_session.py`（`run_static` + `DisaggInferenceSession`）、`config.py`、`common.py`（枚举）、`operations/attention.py`（SOL 公式）、`system_spec.py`。
- 启动 explore 子代理汇总 PerfDatabase / operations / backends / models / systems 等细节。

**产物**
- 在 `Hisim_AIC_detailed.md` 末尾追加 **第六章「AIConfigurator（AIC）内部机制深度解析」**，并将总结升级为 **第七章**。
- 文档由 760 行扩展至 **1031 行（~65 KB）**。

**新增章节明细（§6）**
| 小节 | 内容 |
|------|------|
| 6.0 | AIC 原始定位：本是 Dynamo 分离式服务的配置优化器，Hisim 仅借用底层 `run_static` 延迟建模 |
| 6.1–6.2 | AIC SDK 模块全景 + `run_static → backend → 遍历 ops → latency_dict` 调用层次 |
| 6.3 | **PerfDatabase**：CSV 实测表清单、`DatabaseMode` 五种来源（SILICON/HYBRID/EMPIRICAL/SOL/SOL_FULL）、插值/外插（呼应 Hisim 放宽 `inner_only`） |
| 6.4 | **算子 SOL 公式**（核对源码）：Attention roofline `max(sol_math, sol_mem)`、GEMM、MoE（`sol/0.4`）、CustomAllReduce |
| 6.5 | **QuantMapping 量化因子表**：各 QuantMode 的 (memory, compute)，解释选错量化延迟翻倍 |
| 6.6 | 模型族 `context_ops` / `generation_ops` 组成（LLAMA / MoE / MLA / 混合 / 多模态） |
| 6.7 | InferenceSummary 与 OOM 判定（呼应 Hisim 返回负延迟） |
| 6.8 | **SystemSpec**：`h100_sxm` YAML 的 flops/带宽/拓扑，解释 `device_name` 决定精度 |
| 6.9 | **关键对比**：AIC 原生 `DisaggInferenceSession` 有 PD 但**不建模 KV 传输（源码 TODO）**，揭示 Hisim 为何要自行补 KV 传输延迟 |

**核心洞察**
> **AIC 算"算得多快"，Hisim 算"传得多慢 + 何时发生"——二者互补。**

---

## 第 4 次修改：创建本修改记录文件

**用户诉求**
将以上几次修改的记录，按不同修改次统计，写入一个新的 md 文件。

**产物**
- 创建本文件 `Hisim_AIC_changelog.md`。

---

## 第 5 次修改：重排 `Hisim_AIC_detailed.md` 章节结构

**用户诉求**
1. 将第五章与第七章（两份"总结"）合并，放到文末作为统一总结。
2. 将第六章「AIC 内部机制深度解析」移动到第二章的位置。
3. 第一章只保留 Hisim 的深度解析。

**主要动作**
- 重排章节顺序，并同步更新章节号、子节号与所有正文交叉引用（`§x.y`）。
- 删除原 §1.2「AIC 是什么」（AIC 内容已由新第二章完整覆盖），原 §1.3 工作流程降为 §1.2。
- 交叉引用按"首位章节号"统一重映射：原 2.x→3.x、3.x→4.x、6.x→2.x，章节级 §2→§3。

**产物（新章节顺序）**
| 新章节 | 内容 | 来源 |
|--------|------|------|
| 一 | Hisim 深度解析（仅 Hisim） | 原 §一（去掉 AIC 概述）|
| 二 | AIConfigurator（AIC）内部机制深度解析 | 原 §六 |
| 三 | PD 分离场景下 Hisim 需要做哪些修改 | 原 §二 |
| 四 | Hisim 对 AIC 的互联 + 真实 Workload 推理逻辑 | 原 §三 |
| 五 | 常见疑问 FAQ | 原 §四 |
| 六 | 总结（合并原 §五 + §七）| 原 §五 + §七 |

**体量**：1031 → 1004 行 / ~60 KB（删除重复总结表与 §1.2 概述后略有缩减）。

---

## 汇总表

| 修改次 | 目标 | 产物 | 体量变化 |
|--------|------|------|----------|
| 第 1 次 | 回答 3 个问题的深度文档 | `hisim_aic_pd_deep_dive.md`（后整理为 `Hisim_AIC.md`） | 新建 ~23 KB |
| 第 2 次 | 增补细节 + 澄清，形成新文件 | `Hisim_AIC_detailed.md` | 新建 ~45 KB / 760 行 |
| 第 3 次 | 基于源码增补 AIC 内部机制 | `Hisim_AIC_detailed.md`（追加 §6、升级 §7） | 760 → 1031 行 / ~65 KB |
| 第 4 次 | 撰写修改记录 | `Hisim_AIC_changelog.md` | 新建（本文件） |
| 第 5 次 | 重排章节（合并总结、AIC 提到第二章、第一章仅 Hisim） | `Hisim_AIC_detailed.md` | 1031 → 1004 行 / ~60 KB |

> 注：所有修改均为**文档变更**，未改动任何源码。`Hisim_AIC.md` 作为基线全程保持不变。
