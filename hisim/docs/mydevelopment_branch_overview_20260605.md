# MyDevelopment 分支实现总览（2026-06-05）

本文是当前 `MyDevelopment` 分支的高层笔记，聚焦“做了什么”和“怎么用”，不展开底层实现细节。

## 1. 这条分支主要在做什么

当前分支在原 HiSim 基础上，重点形成了三条能力线：

- PD（Prefill/Decode）分离仿真能力
- AIC 到 HiSim 的桥接与 sweep 回放能力
- 可视化分析（报表 + dashboard + single run）

最近还新增了两类偏工程化工具：

- W1-W6 时序 trace 工具（stub 与 real-hw）
- B60 下 PD 拓扑鲁棒性 sweep 工具

## 2. 功能全貌

### 2.1 PD 分离仿真（核心）

- 支持将 prefill 与 decode 作为两个角色进行虚拟时间仿真。
- 运行后端有两类：
  - Backend A：单进程虚拟时间
  - Backend B：双进程 worker（队列通信）
- 提供统一协议层，便于 A/B 切换与一致性对照。
- 支持通过配置传入分离参数（包括 prefill/decode 各自配置）。

对应能力入口：

- `hisim/src/hisim/simulation/pd_factory.py`
- `hisim/src/hisim/simulation/pd_backend_a.py`
- `hisim/src/hisim/simulation/pd_backend_b.py`
- `hisim/src/hisim/simulation/pd_runtime.py`

### 2.2 AIC ↔ HiSim 桥接与 sweep

- 可将 AIC 的结果（包括 top-N）转换成 HiSim 可执行配置。
- 支持 disagg 参数透传（含 emit-disagg 路径）。
- 已补齐 `moe_tp_size` 端到端传递，避免 AIC pareto 行回放失真。
- 提供 PD 与 AGG 的对比脚手架和 runbook。

对应能力入口：

- `hisim/tools/aic_to_hisim_bridge.py`
- `hisim/tools/aic_topn_to_hisim_sweep.py`
- `hisim/tools/pd_vs_agg_compare.py`
- `hisim/docs/pd_vs_agg_runbook.md`

### 2.3 可视化（分析闭环）

- 从 sweep CSV 到图表、HTML 报告、streamlit dashboard 的完整链路。
- dashboard 支持筛选、分标签页查看、single run 对比、comparison tray。

对应能力入口：

- `hisim/tools/sweep_plot.py`
- `hisim/tools/sweep_dashboard.py`
- `hisim/docs/dashboard_handbook.md`

### 2.4 W1-W6 与 B60 拓扑工具（近期新增）

- W1-W6 trace：输出每个请求分阶段时间点，支持 HTML。
- real-hw trace：通过 AIC perf-db 跑真实硬件配置路径。
- B60 topology trace：做 C1-C6 配置扫面，验证分离拓扑组合可运行性。

对应工具：

- `hisim/tools/trace_w1_w6.py`
- `hisim/tools/trace_w1_w6_html.py`
- `hisim/tools/trace_w1_w6_real_hw.py`
- `hisim/tools/trace_pd_topology_b60.py`

## 3. 典型使用路径（从快到全）

1. 快速验证分离链路
   - 运行：`hisim/tools/pd_demo_e2e.sh`

2. 从 AIC 结果回放到 HiSim
   - 先桥接：`aic_to_hisim_bridge.py`
   - 再 sweep：`aic_topn_to_hisim_sweep.py`

3. 做 PD vs AGG 对照
   - 参考：`hisim/docs/pd_vs_agg_runbook.md`
   - 工具：`hisim/tools/pd_vs_agg_compare.py`

4. 看可视化结果
   - 报表：`sweep_plot.py` / 自动 HTML
   - 交互：`streamlit run hisim/tools/sweep_dashboard.py -- --csv <summary.csv>`

5. 做时序排查
   - 先用 stub：`trace_w1_w6.py`
   - 再用 real-hw：`trace_w1_w6_real_hw.py`

## 4. 使用注意（高层）

- 该分支核心是“虚拟时间仿真与相对比较”，不是实际执行模型。
- AIC 回放时需确保关键调度字段完整（尤其 MoE 相关字段）。
- 在 B60 场景做 AIC 相关回放时，优先按 runbook 推荐模式执行（包含 HYBRID 路径）。

## 5. 与已有文档的关系

- 本文定位：分支全貌导航（给人快速建立地图）
- 细节操作：看 `pd_vs_agg_runbook.md` 与 `dashboard_handbook.md`
- 协议/判定口径：看 `pd_ab_equivalence_spec.md`
