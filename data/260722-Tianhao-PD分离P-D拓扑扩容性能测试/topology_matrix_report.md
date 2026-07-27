# HiSim PD 拓扑矩阵验证报告（2P1D/4P1D/8P1D/16P1D/1P2D/1P4D/1P8D/1P16D）

> 本文档记录对 `hisim/src/hisim/simulation/sglang/sglang_hook.py` 三处 bug（Bug A/B 崩溃修复 +
> Bug C 延迟核算修复）修复后，在 8 个新增拓扑（加基线 1P1D 共 9 个）下，用真实 SGLang 调度器 +
> HiSim 仿真计算的方式重新压测的完整结果。**Bug C 已于第二轮验证中确认修复完成**，详见第 7 节。

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 仓库 | `Yaoy4/tair-kvcache`（分支 `Thjiang_Dev`），修复代码位于此仓库 |
| 验证工作目录 | `/mnt/nfs02/users/tjiang/pd_verify_e0735cc/`（独立 venv，含真实 `sglang==0.5.6.post2`） |
| AIConfigurator 依赖 | `/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator`（commit `e0735ccca08c790b37c511a20f484a62a4127118`） |
| 模型 | `Qwen/Qwen3-8B`（本地 HF 缓存，无需联网下载） |
| GPU | 2× NVIDIA RTX PRO 6000（本次测试未实际占用算力，见下方架构说明） |
| 第一轮测试日期（Bug A/B 修复后，Bug C 尚未修复） | 2026-07-21 17:11 ~ 17:22（CST） |
| 第二轮测试日期（Bug A/B/C 全部修复后） | 2026-07-21 17:55 ~ 18:08（CST） |
| Bug A/B 修复代码提交时间 | 2026-07-21 16:23:54 |
| Bug C 修复代码完成时间 | 2026-07-21 17:45（早于第二轮测试） |

## 2. 背景

用户此前报告：2P1D 与基线持平（合理，prefill 侧未到瓶颈），但 **1P2D 在真实 server 场景下崩溃**：

```
RuntimeError: PD decode batch contained no admissible request state;
native and PD capacity/state tracking diverged
```

定位到 `sglang_hook.py:1445`，根因是 `decode_queue_mode="per_replica_queue"`（2+ decode 副本模式）下的准入
逻辑 bug（详见第 5 节)。已完成两处修复（Bug A + Bug B），单测 285 passed / 1 skipped，无回归。本轮任务是把
修复后的代码放到 **真实 server** 场景下，跑 2P1D / 4P1D / 8P1D / 16P1D（prefill 扩容）与 1P2D / 1P4D / 1P8D /
1P16D（decode 扩容，即触发过崩溃的模式）共 8 个新拓扑，验证修复效果并给出完整报告。

## 3. 测试方法

### 3.1 架构说明："真实 server + 仿真计算"

- `python3 -m hisim.simulation.sglang.launch_server`：启动一个**真正的 SGLang 调度器进程**（真实 HTTP
  API、真实的请求批处理/排队/调度逻辑，`SGLANG_USE_CPU_ENGINE=1`），但通过 `sglang_hook.py` 把
  `run_batch`/`process_batch_result` 换成 AIConfigurator 预测延迟（日志中的 `Simulation triggered` /
  `Simulation stop triggered`），不做真实 GPU 前向计算。
- 这正是为什么 2 张物理 GPU 能测出 16 个 decode 副本的拓扑——所有"副本"都只是同一进程内的虚拟时钟/队列
  bucket，没有真实显存/算力占用。
- 压测客户端：`python3 -m hisim.simulation.bench_serving`（与 SGLang 官方 `bench_serving.py` 对齐），
  输出完整 `Serving Benchmark Result` 摘要 + 逐请求 JSONL。

### 3.2 固定负载 cell（与用户原始 2P1D/1P2D 报告方法一致）

```
model=Qwen/Qwen3-8B  request-rate=8  random-input-len=1024  random-output-len=512
num-prompts=200  seed=1  random-range-ratio=0  warmup-requests=0  flush-cache
port=127.0.0.1:30003
```

### 3.3 拓扑矩阵

| 拓扑 | prefill.replicas | decode.replicas | decode_queue_mode |
|---|---|---|---|
| 1P1D（基线） | 1 | 1 | single_replica |
| 2P1D | 2 | 1 | single_replica |
| 4P1D | 4 | 1 | single_replica |
| 8P1D | 8 | 1 | single_replica |
| 16P1D | 16 | 1 | single_replica |
| 1P2D | 1 | 2 | **per_replica_queue**（含本次修复的代码路径） |
| 1P4D | 1 | 4 | **per_replica_queue** |
| 1P8D | 1 | 8 | **per_replica_queue** |
| 1P16D | 1 | 16 | **per_replica_queue** |

驱动脚本：`run_topology_matrix.sh`，对每个拓扑：起 server → 等端口 → 跑固定负载 cell → 抓取结果 → 停 server →
下一个拓扑。全部 9 次运行**零崩溃**、**零请求失败**（200/200 全部成功）。

## 4. 完整结果汇总表（第二轮：Bug A/B/C 全部修复后，最终结果）

| 拓扑 | 成功请求 | 总吞吐 (tok/s) | Mean E2E (ms) | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|---|
| 1P1D（基线） | 200/200 | 1518.88 | 4861.41 | 46.98 | 19.57 |
| 2P1D | 200/200 | 1519.16 | 4822.96 | 44.30 | 19.41 |
| 4P1D | 200/200 | 1519.16 | 4822.96 | 44.30 | 19.41 |
| 8P1D | 200/200 | 1519.16 | 4822.96 | 44.30 | 19.41 |
| 16P1D | 200/200 | 1519.16 | 4822.96 | 44.30 | 19.41 |
| 1P2D | 200/200 | 1534.86 | 4149.05 | 46.45 | 16.62 |
| 1P4D | 200/200 | 1544.74 | 3923.11 | 45.24 | 15.56 |
| 1P8D | 200/200 | 1547.35 | 3681.31 | 45.29 | 14.37 |
| 1P16D | 200/200 | 1555.03 | 3385.91 | 45.62 | 13.18 |

> 第一轮（Bug A/B 已修复、Bug C 尚未修复）时 decode 扩容组的原始异常数据已归档在
> `pre_bugc_fix_results/topology_matrix_summary/`，完整前后对比见第 7.3 节。

## 5. 关键结论一：崩溃修复已验证通过 ✅

- **1P2D/1P4D/1P8D/1P16D 全部成功完成，200/200 请求无一失败，server 日志中零 `RuntimeError`/`Traceback`。**
- 修复前证据保留在 `server_1P2D_PRE_FIX_CRASH.log.bak`（原始崩溃时间戳 15:32:47-58，修复提交于
  16:23:54，即修复前的崩溃复现）：
  ```
  RuntimeError: PD decode batch contained no admissible request state;
  native and PD capacity/state tracking diverged
  ```
- 修复内容回顾（`sglang_hook.py` 的 `per_replica_queue` decode 准入逻辑）：
  - **Bug A（本次报告崩溃的直接原因）**：原代码用"跨副本 busy_until 的最小值"去 poll 一次 KV 就绪状态，
    但各 bucket 自己的 `step_start`（经 `sync_decode_start` 同步后）可能比这个最小值更晚；落在两者之间的
    请求永远不会被 promote 成 `WAITING_DECODE`，如果这轮所有 bucket 都遇到这种情况，`token_times` 就会
    是空的，从而触发这个 RuntimeError。修复为"两遍扫描"：先算出每个 bucket 自己的 `step_start`，再用其中
    的**最大值**去 poll 一次（数学上等价于逐 bucket poll，且仍是 O(N) 单次扫描）。
  - **Bug B（验证过程中发现的伴生 bug）**：`admit_decode_for_replica` 会把"这一轮没被准入"的请求的粘性
    副本绑定清掉——包括本来就不是准入候选、只是继续跑的 `RUNNING_DECODE` 请求，导致其绑定被每轮误删，
    下一轮可能被重新分配到不同副本（物理上不合理，KV cache 只存在于原副本上）。修复为准入前过滤掉已经
    是 `RUNNING_DECODE` 的请求。
- 两处修复均补充了回归测试（`test_decode_admits_both_buckets_when_busy_until_diverges_from_kv_ready`、
  `test_running_decode_request_keeps_sticky_replica_across_rounds`），`hisim/tests/unit/simulation/` 全量
  285 passed / 1 skipped。

## 6. 关键结论二：prefill 扩容（2/4/8/16P1D）符合预期

2P1D/4P1D/8P1D/16P1D 四组结果**完全一致**（非 bug，已核实）：
- 已核实各 server 日志中 `PD disaggregation enabled (... prefill_replicas=N ...)` 确实按配置生效（4P1D
  日志显示 `prefill_replicas=4`，16P1D 显示 `prefill_replicas=16`），并非脚本没有正确启动。
- 原因：在 `request-rate=8` 这个负载点下，1 个 prefill 副本本身就没有排队（这与用户此前对 2P1D 的分析
  一致："prefill 侧显然还没到瓶颈"）。因此继续增加 prefill 副本数对该负载点没有任何可观测影响，这是预期
  内的合理结果，不代表配置无效或 replica 数没生效。

## 7. 关键结论三：decode 扩容延迟核算 bug（Bug C）已定位、修复并验证通过 ✅

第一轮压测中发现 decode 扩容（1P2D→1P16D）延迟随副本数增长单调恶化（1P16D TTFT 相对基线 +7434%），
与直觉相反。本节记录该问题（Bug C）的根因、修复方案、以及修复后的完整复测结果。

### 7.1 现象回顾（第一轮，Bug C 修复前）

| 拓扑 | E2E 变化 | TTFT 变化 | TPOT 变化 | 总吞吐变化 |
|---|---|---|---|---|
| 1P2D | +177% | +307% | +223% | -4.4% |
| 1P4D | +219% | +841% | +361% | -3.6% |
| 1P8D | +242% | +2350% | +505% | -8.0% |
| 1P16D | +239% | **+7434%** | +432% | -13.9% |

decode 副本数越多，TTFT/TPOT/E2E 越差、总吞吐越低，且全程只有 1 个 prefill 副本却也观测到 TTFT 暴涨——
不应由 decode 副本数决定。

### 7.2 根因

`sglang_hook.py` 的 `per_replica_queue` decode 分支中，每轮向原生 SGLang 调度器汇报的"这一步用了多久"
（`predicted_latency`，随后被 `time.sleep(abs(predicted_latency))` 真实睡眠掉）计算方式是：

```python
predicted_latency = max(bucket_step_ends) - min(bucket_step_starts)
```

即"跨所有 decode 副本 bucket 的时间跨度"。`per_replica_queue` 模式下各 decode 副本本来就是**故意设计
成互相独立的时钟**（较轻负载的 bucket 天然会跑得比较忙的 bucket 快，这是该模式存在的意义，也是既有回归
测试 `test_concurrent_decode_replicas_use_own_busy_until_not_slowest` 明确断言、不能破坏的不变量）。
`max(ends) - min(starts)` 这个公式会把"两个独立 bucket 之间偶然出现的时钟错位"误当成这一轮的真实计算
延迟，每轮都重新计入一次；一个请求最多要跑 512 个 decode round（本次 `random-output-len=512`），哪怕
每轮多算出一点"幽灵延迟"，乘以几百轮也会迅速累积成秒级的额外延迟，且副本数越多、忙闲越不均、错位幅度
越大，恰好吻合观测到的单调恶化趋势。

### 7.3 修复方案

只修改"这一轮延迟该报多少"这一个聚合公式，**完全不触碰** `bucket_step_start` 的计算方式、`sync_decode_
start` 门控逻辑、准入逻辑、`busy_until` 记账——即完全不影响 `per_replica_queue` 的 bucket 独立性设计
（保证 `test_concurrent_decode_replicas_use_own_busy_until_not_slowest` 等既有测试不受影响）：

```python
# 旧（bug）：跨 bucket 的时间跨度，把时钟错位也算成了延迟
predicted_latency = max(bucket_step_ends) - min(bucket_step_starts)

# 新（修复后）：每个 bucket 自己的 round 延迟（自己的 end - 自己的 start），
# 取所有 bucket 中最慢的那个——不再混用不同 bucket 的 start/end
predicted_latency = max(
    end - start
    for start, end in zip(bucket_step_starts, bucket_step_ends)
)
```

修复位置：`hisim/src/hisim/simulation/sglang/sglang_hook.py`（约 1480-1490 行）。
同步更新了单测 harness `HookDriver`（`test_pd_role_clock_integration.py`）以镜像同一处逻辑，并新增回归
测试 `test_decode_round_latency_ignores_idle_bucket_clock_skew`：构造两个人为差异巨大的 decode 副本
（重请求 vs 轻请求）跑 6 轮使其时钟明显错位，断言修复后汇报的延迟等于"最慢 bucket 自己的当轮延迟"，
且严格小于旧公式会算出的"跨 bucket 错位延迟"；并用 revert-and-reproduce 方法确认该测试在旧代码上会
失败、在新代码上通过。`hisim/tests/unit/simulation/` 全量回归：**286 passed / 1 skipped**（较 Bug A/B
阶段的 285 passed 多出这 1 个新测试，无任何回归）。

### 7.4 真实 server 复测结果：修复前 vs 修复后

| 拓扑 | 指标 | 修复前（Bug C 仍在） | 修复后 | 变化 |
|---|---|---|---|---|
| 1P2D | Mean E2E | 13471.96 ms | **4149.05 ms** | -69.2% |
| 1P2D | Mean TTFT | 191.01 ms | **46.45 ms** | -75.7% |
| 1P2D | Mean TPOT | 63.25 ms | **16.62 ms** | -73.7% |
| 1P4D | Mean E2E | 15496.21 ms | **3923.11 ms** | -74.7% |
| 1P4D | Mean TTFT | 441.93 ms | **45.24 ms** | -89.8% |
| 1P4D | Mean TPOT | 90.18 ms | **15.56 ms** | -82.7% |
| 1P8D | Mean E2E | 16640.92 ms | **3681.31 ms** | -77.9% |
| 1P8D | Mean TTFT | 1151.13 ms | **45.29 ms** | -96.1% |
| 1P8D | Mean TPOT | 118.45 ms | **14.37 ms** | -87.9% |
| 1P16D | Mean E2E | 16482.35 ms | **3385.91 ms** | -79.5% |
| 1P16D | Mean TTFT | 3539.36 ms | **45.62 ms** | -98.7% |
| 1P16D | Mean TPOT | 104.07 ms | **13.18 ms** | -87.3% |

修复后，decode 扩容组呈现的趋势完全反转为**符合直觉的单调改善**（相对 1P1D 基线）：

| 拓扑 | Mean E2E 变化 | Mean TTFT 变化 | Mean TPOT 变化 | 总吞吐变化 |
|---|---|---|---|---|
| 1P2D | -14.6% | -1.1%（噪声范围内） | -15.1% | +1.1% |
| 1P4D | -19.3% | -3.7%（噪声范围内） | -20.5% | +1.7% |
| 1P8D | -24.3% | -3.6%（噪声范围内） | -26.6% | +1.9% |
| 1P16D | -30.3% | -2.9%（噪声范围内） | -32.6% | +2.4% |

- TPOT/E2E 随 decode 副本数增加单调下降（更多 decode 容量分摊负载，延迟应该变好，现在确实变好了）。
- TTFT 在各拓扑间基本持平（±5% 内），符合预期——本组全程只有 1 个 prefill 副本，TTFT 主要由 prefill
  阶段决定，不应该受 decode 副本数影响，Bug C 修复前"TTFT 随 decode 副本数暴涨"的反常现象已完全消失。
- 全部 9 个拓扑复测 200/200 请求成功、零崩溃，Bug A/B 的崩溃修复在最终代码上再次确认有效。

## 8. 文件位置清单

全部路径均在 `/mnt/nfs02/users/tjiang/pd_verify_e0735cc/` 下（除代码修复本身在 `tair-kvcache` 仓库）。

### 8.1 代码修复（本次修复对象，前序任务已完成，未在本轮改动）
- `/mnt/nfs02/users/tjiang/Gitrepo/tair-kvcache/hisim/src/hisim/simulation/sglang/sglang_hook.py`
- `/mnt/nfs02/users/tjiang/Gitrepo/tair-kvcache/hisim/tests/unit/simulation/test_pd_role_clock_integration.py`

### 8.2 本轮报告与汇总
- **本报告**：`/mnt/nfs02/users/tjiang/pd_verify_e0735cc/topology_matrix_report.md`

### 8.3 拓扑配置（9 个）
`/mnt/nfs02/users/tjiang/pd_verify_e0735cc/topo_{1P1D,2P1D,4P1D,8P1D,16P1D,1P2D,1P4D,1P8D,1P16D}.json`
（`database_path` 均指向永久仓库 `/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator/src/aiconfigurator/systems`）

### 8.4 驱动脚本与运行日志
- `/mnt/nfs02/users/tjiang/pd_verify_e0735cc/run_topology_matrix.sh`
- `/mnt/nfs02/users/tjiang/pd_verify_e0735cc/run_matrix_full.log`（**第二轮**：Bug A/B/C 全部修复后的
  9 拓扑批量运行驱动日志，17:55~18:08）

### 8.5 Server 端日志（每拓扑一份，含完整 SGLang 调度器输出，均为第二轮/最终结果）
`/mnt/nfs02/users/tjiang/pd_verify_e0735cc/server_{1P1D,2P1D,4P1D,8P1D,16P1D,1P2D,1P4D,1P8D,1P16D}.log`
- 额外保留：`server_1P2D_PRE_FIX_CRASH.log.bak`（Bug A 修复前的原始崩溃日志，Bug A/B 阶段 before/after
  对比证据）

### 8.6 压测结果摘要（每拓扑一份，含完整 `Serving Benchmark Result`，均为第二轮/最终结果）
`/mnt/nfs02/users/tjiang/pd_verify_e0735cc/topology_matrix_summary/result_{1P1D,2P1D,4P1D,8P1D,16P1D,1P2D,1P4D,1P8D,1P16D}.txt`

### 8.7 逐请求指标 JSONL（每拓扑一份，均为第二轮/最终结果）
`/mnt/nfs02/users/tjiang/pd_verify_e0735cc/cases_topo/metrics_{TAG}_rr8_il1024_ol512.jsonl`
- `bench_serving` 的 `--output-file` 是追加写入，第二轮复测后这 9 个文件一度都变成 2 行（第一轮 +
  第二轮历史）；已清理为仅保留最新（第二轮/最终修复后）一行，完整历史分别保留在同目录下对应的
  `metrics_{TAG}_rr8_il1024_ol512.jsonl.with_history.bak`。

### 8.8 第一轮（Bug C 修复前）结果归档，供 before/after 对比
- `/mnt/nfs02/users/tjiang/pd_verify_e0735cc/pre_bugc_fix_results/topology_matrix_summary/result_{TAG}.txt`
- `/mnt/nfs02/users/tjiang/pd_verify_e0735cc/pre_bugc_fix_results/cases_topo/metrics_{TAG}_*.jsonl`
- `/mnt/nfs02/users/tjiang/pd_verify_e0735cc/pre_bugc_fix_results/server_{TAG}.log`
（`pre_bugc_fix_results/cases_topo/metrics_2P1D_rr8_il1024_ol512.jsonl.with_history.bak` 是更早阶段
（Bug A 修复前后）遗留的独立历史备份，与本次 Bug C 前后对比归档相互独立，不代表本轮新增内容。）

### 8.9 Bug C 单测修复验证过程记录
- 生产代码修复：`hisim/src/hisim/simulation/sglang/sglang_hook.py`（约 1480-1490 行，`round_latency` 聚合
  公式）
- 测试 harness 同步修复：`hisim/tests/unit/simulation/test_pd_role_clock_integration.py`
  （`HookDriver.decode()` 的 `self.last_round_latency` 计算，及 `extend()`/单副本分支的对应赋值）
- 新增回归测试：`test_decode_round_latency_ignores_idle_bucket_clock_skew`（同文件内，紧跟 Bug B 回归
  测试之后）

## 9. 结论与建议

1. ✅ **崩溃（Bug A）与伴生的粘性绑定损坏（Bug B）均已修复并在真实 server 场景下验证通过**：
   1P2D/1P4D/1P8D/1P16D 全部 200/200 成功，零崩溃（第一轮、第二轮复测均确认）。
2. ✅ prefill 扩容（2/4/8/16P1D）结果符合预期，无异常。
3. ✅ **decode 扩容延迟核算问题（Bug C）已定位、修复并完成真实 server 复测验证**：
   - 根因：`per_replica_queue` decode 分支把"跨副本 bucket 时间跨度"误当作每轮真实延迟汇报，每轮
     `time.sleep()` 都多算一次副本间的历史时钟错位，随 decode 副本数与迭代轮次（最多 512 轮/请求）
     复合放大。
   - 修复：只改"每轮延迟怎么汇报"这一个聚合公式（`max(每个 bucket 自己的 end-start)`），完全不触碰
     bucket 独立时钟设计、准入逻辑或既有测试所验证的不变量。
   - 单测：新增回归测试 + 全量 `hisim/tests/unit/simulation/`：**286 passed / 1 skipped**，无回归
     （较 Bug A/B 阶段的 285 passed 多 1 个新测试）。
   - 真实 server 复测：1P16D 的 Mean TTFT 从修复前的 3539ms（+7434% vs 基线）降到 45.62ms
     （与基线基本持平），decode 扩容组的 TPOT/E2E 从"单调恶化"完全反转为"单调改善"（1P16D TPOT 较
     基线 -32.6%），全部 9 拓扑 200/200 成功、零崩溃。
4. 本次任务涉及的三个 bug（Bug A 崩溃、Bug B 粘性绑定损坏、Bug C 延迟核算）均已修复、均有独立回归测试
   覆盖、均在真实 server 场景下完成 before/after 复测验证，未发现进一步异常。
