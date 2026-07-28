# HiSim PD 分离模式 L2 时延未计入性能指标 BUG 报告

**影响模块：** HiSim PD Disaggregation Simulation / HiCache L2 timing  
**代码版本：** `49b144b9fd48a22d4730d1d0180a228f87b86f21`  
**调查状态：** L2 write-through backup 已复现并定位；L2 load 存在同类代码风险，待非零 load 用例验证  
**修复状态：** 尚未修改

---

## 1. 问题摘要

PD 分离模式下，HiCache L2 的 Host I/O 耗时已经根据 `memory_read_bandwidth_gb` 和 `memory_write_bandwidth_gb` 正确计算，但 PD 请求的 TTFT、TPOT、E2E 和吞吐使用另一套 Prefill/Decode replica-local 时钟生成，没有使用包含 L2 耗时的 `request_response_time`。

已确认的直接结果是：开启 `write_through` 后，降低 Host Memory Bandwidth 会显著增加 L2 backup 耗时，但 18 个 workload 的核心性能指标逐项完全不变。

---

## 2. 复现环境与现象

| 项目 | 配置 |
|---|---|
| HiSim | commit `49b144b` |
| SGLang | 0.5.6post2 |
| AIConfigurator | commit `e0735cc` |
| 拓扑 | Replica-P2D2，Prefill/Decode TP=1 |
| HiCache | `--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through --hicache-io-backend kernel` |
| 对比项 | Host read/write bandwidth：480 GB/s 与 64 GB/s |
| 480 GB/s 基线组 | `cases_same_seed_1-P2D2-tp1-dp1-Preplica2-Dreplica2-PDdisagg` |
| 64 GB/s 对照组 | `cases_same_seed_1-P2D2-tp1-dp1-Preplica2-Dreplica2-MemBW64-PDdisagg` |
| Workload | `RR={1,8,64}`、`IL={1024,4096,16384}`、`OL={1024,4096}`，共 18 组 |

观察结果：

| Host BW | 非零 L2 backup iteration | 累计 backup | 非零 L2 load iteration | 累计 load |
|---:|---:|---:|---:|---:|
| 480 GB/s | 3,209 | 19.779 s | 0 | 0 s |
| 64 GB/s | 3,209 | 80.072 s | 0 | 0 s |

Host BW 从 480 降至 64 GB/s 后，累计 backup 增加约 60.293 s、达到原来的 4.05 倍，但两组实验的 TTFT、TPOT、E2E 和吞吐在 18/18 个 workload 中逐项完全相同。

---

## 3. 预期行为与实际行为

### 3.1 预期行为

`write_through` 产生的 D2H backup 时延应按照现有 overlap 语义进入请求完成时间：

- overlap 开启时，backup 作为当前 batch 的尾部时延计入 `request_response_time`；
- overlap 关闭时，load、inference 和 backup 均推进请求时钟；
- 降低 `memory_write_bandwidth_gb` 后，至少 E2E 或可服务时间应发生可解释的变化。

L2 load 也应在扣除可被上一轮 inference 覆盖的部分后，阻塞对应 batch 的 Prefill/Decode 时间轴，而不是只推进独立的 global clock。

### 3.2 实际行为

| 阶段 | 非 PD 模式 | PD 分离模式 |
|---|---|---|
| L2 load/backup 计时 | 写入 `StateManager` | 同左，计时本身正常 |
| batch 完成时间 | 使用含 L2 时延的 `request_response_time` | P/D replica 使用各自 `busy_until` 和 token time |
| token latency | 使用 `request_response_time` | 优先使用 `PD_BATCH_TOKEN_TIMES` |
| 最终结果 | L2 时延进入指标 | L2 backup 未进入 PD token 指标 |

---

## 4. 根因分析

### 4.1 L2 时延只进入 global/request-response 时钟

`MockTokenToKVPoolHost` 会正确估算并累计 Host I/O 时间：

```python
load_to_device_per_layer(...)
    -> StateManager.inc_hicache_l2_load_dur(total_time_cost)

backup_from_device_all_layer(...)
    -> StateManager.inc_hicache_l2_backup_dur(total_time_cost)
```

`wrapped_process_batch_result()` 随后弹出这两个耗时。overlap 模式下：

```python
StateManager.step_global_clock(
    max(hicache_l2_load_dur - StateManager.get_last_inference_dur(), 0)
)
StateManager.step_global_clock(current_inference_dur)
request_response_time = (
    StateManager.get_global_clock() + hicache_l2_backup_dur
)
```

因此，L2 load/backup 已进入 `request_response_time`，且 backup 会随 Host write bandwidth 正常变化。

### 4.2 PD token 指标绕过 `request_response_time`

PD 分离下，Prefill/Decode 完成时间来自 Backend A 的 replica-local `busy_until`。Decode token 时间写入：

```python
C_SchedulerHook.PD_BATCH_TOKEN_TIMES = token_times
```

生成请求指标时，PD 分支优先读取该时间：

```python
token_time = C_SchedulerHook.PD_BATCH_TOKEN_TIMES.get(
    req.rid,
    C_SchedulerHook.PD_LAST_DECODE_STEP_END,
)
req_stats.gen_token_latencies.append(
    token_time - req_stats.last_event_time
)
```

只有非 PD 分支使用包含 L2 时延的 `request_response_time`。因此，即使 `hicache_l2_backup_dur` 已经增大，PD 的 token time、TTFT、TPOT 和 E2E 仍不会变化。

根因可概括为：

```text
HiCache L2 I/O -> StateManager global clock -> request_response_time
                                                  X
PD Backend A -> replica busy_until -> PD_BATCH_TOKEN_TIMES -> PD metrics
```

两条时间轴没有在 L2 I/O 边界进行同步。

---

## 5. L2 load 调查结论

现有 Replica-P2D2 对比数据中，两档带宽的 L2 load 均为 0，因此当前实验不能证明 load 已经影响或未影响最终指标。

但代码审查发现同类风险：

1. `hicache_l2_load_dur` 在 `wrapped_process_batch_result()` 中才推进 global clock。
2. 当前 PD batch 的 `now_clock`、Prefill/Decode `busy_until` 和 token time 此前已经由 role-local 时间轴计算完成。
3. `prefill_batch_start()` 只取 Prefill replica 空闲时间和 request admission time，不直接读取 L2 load clock。
4. load 可能通过后续请求的 global queue/arrival 时间产生间接影响，但当前 batch 的非重叠 load 部分没有明确同步回对应 P/D replica 时钟。

因此，L2 load 应标记为 **代码层面高度疑似遗漏**，但仍需构造存在 L2 hit/load 的 PD workload 做最终验证，不能仅凭本轮零 load 数据下实证结论。

---

## 6. 影响范围

- 影响启用 PD 分离且启用 HiCache L2 的仿真结果。
- `write_through` backup 已确认未反映到 PD 的 TTFT、TPOT、E2E 和吞吐。
- Host write bandwidth 敏感性会被错误隐藏，可能导致配置选择和瓶颈判断失真。
- L2 load 是否遗漏及影响程度取决于 workload 的 L2 hit/load 行为，尚需专项复现。
- PD 合并模式仍使用 `request_response_time`，不经过上述 PD token time 分支，本问题未在该路径复现。

---

## 7. 修复建议与验收标准

建议在 PD Backend A 的 role-local 时间轴内统一处理 HiCache I/O，而不是仅修改最终指标：

1. 明确 L2 load 属于 Prefill 还是 Decode 的前置依赖，并将不可重叠部分加入对应 replica 的 batch start/end。
2. 按现有 overlap 语义，将 write-through backup 加入请求完成时间；若 backup 会占用共享资源，还应同步相关 replica 的可服务时间。
3. 保持 PD token time、request completion time 和 global diagnostics 使用同一套可组合时间语义，避免在指标导出阶段重复补时。
4. 增加 Host BW 480/64 GB/s 的 PD 回归测试，断言 backup 时延变化会反映到 E2E 或明确规定的服务时间指标。
5. 新增可稳定产生非零 L2 load 的 PD 用例，分别覆盖 overlap 开启和关闭场景，并断言不可覆盖的 load 时延进入对应 P/D 时间轴。

修复验收至少应满足：L2 iteration 诊断值、PD replica 时钟和最终请求指标三者能够闭环解释；不得再次出现 backup 增加约 60 s、而 18/18 组业务指标逐项完全不变的情况。