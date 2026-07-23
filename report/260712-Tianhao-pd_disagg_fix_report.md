# HiSim PD 分离问题修复报告

> 本文档记录 `Thjiang_Dev` 分支 PD（Prefill/Decode Disaggregation）实现的完整复查、问题根因、核心代码修改和回归结果。

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 仓库 | `Yaoy4/tair-kvcache` |
| 分支 | `Thjiang_Dev` |
| 基线提交 | `8d3e631357d8bb567349291716b44d135a93da89` |
| 检查日期 | 2026-07-12 |
| 主要范围 | `hisim/src/hisim/simulation` |
| 后端 | Backend A（单进程虚拟时间）、Backend B（Predictor worker） |
| SGLang 兼容范围 | 项目已注册的 SGLang 0.5.x 版本 |

本文档对应的是基线提交之上的本地修复内容。代码尚未在本文档中假定已经提交或推送。

## 2. 修复结果概览

本轮修复完成后，PD 请求的主要执行链路如下：

```text
请求进入 SGLang waiting queue
        ↓
记录 queue_start / queue_end
        ↓
按 Prefill 容量、负载和副本亲和性分配请求
        ↓
每个 Prefill 副本形成自己的真实 fused batch
        ↓
Chunked Prefill 保持副本和容量槽位
        ↓
按本地 batch 完成时间进入共享 KV 传输链路
        ↓
KV_TRANSIT → WAITING_DECODE
        ↓
按 Decode 容量和 queue mode admission
        ↓
RUNNING_DECODE
        ↓
正常请求达到 OSL / 外部 abort
        ↓
释放全部槽位、清理副本绑定、写入最终指标
```

最终解决的问题包括：

1. PD 状态标志位存在但未被完整执行。
2. Prefill 容量没有与 SGLang 原生调度对齐。
3. Chunked Prefill 跨 scheduler batch 的容量和副本亲和性不正确。
4. Prefill 请求没有正确分配到独立服务副本形成 replica-local batch。
5. Predictor 接收到的是 token 总和，而不是真实 batch shape。
6. `prefill_queue_wait` 结构性接近 0。
7. 不同 Prefill 副本的 KV handoff 被错误同步到最慢请求。
8. 多个 KV 传输错误地各自独占完整网络带宽。
9. Decode 容量在默认 `single_replica` 模式下未完整生效。
10. 外部 abort 后 PD 状态、Prefill/Decode 槽位和副本绑定不能及时释放。
11. 复核并撤回错误的 EOS/推测解码兼容逻辑，保证标准 HiSim 严格按 OSL 结束。
12. PD 时间戳和普通请求时间戳使用不同时间原点。
13. PD 模式仍重复执行随后会被覆盖的聚合 Predictor。
14. 仅解析 PD 配置时也强制依赖外部 AIC SDK。
15. SGLang 最终 Prefill sampling 首 token 未计入 OSL，且其时延不能按完整 Decode 计价。

## 2.1 核心代码修复前后对照

本节集中展示最关键的代码变化。修复前代码为旧实现的核心逻辑摘录或等价简化，省略了与问题无关的日志和参数。

### 2.1.1 状态标志位：从“记录字段”改为“执行约束”

修复前的主要问题是调度入口没有完整验证请求 phase。请求只要出现在 Prefill batch 中，就可能再次执行 arrival/admission：

```python
# 修复前：没有在 Backend 边界验证当前 phase
controller.on_request_arrival(req, now)
controller.admit_prefill(capacity=1, now=start)
req.prefill_end_time = end
```

为什么有问题：

1. 已经进入 `KV_TRANSIT` 或 `RUNNING_DECODE` 的请求仍可能重新走 Prefill。
2. 重复 arrival 会重复入队。
3. `prefill_end_time` 可能覆盖已经完成阶段的时间戳。
4. phase 发生错误后，容量计数和指标也会继续被污染。

修复后在 Backend 修改任何副本状态之前执行强校验：

```python
def _validate_prefill_phase(self, req):
    if req.phase not in (
        RequestPhase.WAITING_PREFILL,
        RequestPhase.RUNNING_PREFILL,
    ):
        raise ValueError(
            f"cannot schedule prefill for rid={req.rid!r} "
            f"in phase={req.phase.value}"
        )

    if (
        req.phase == RequestPhase.RUNNING_PREFILL
        and req.rid not in self._prefill_slot_reserved
    ):
        raise ValueError(
            f"running prefill request rid={req.rid!r} "
            "has no reserved slot"
        )
```

为什么修复后成立：

- `WAITING_PREFILL` 表示新请求，可以执行 arrival 和首次 admission。
- `RUNNING_PREFILL` 只允许已持有槽位的 Chunked 请求继续运行。
- `KV_TRANSIT`、`WAITING_DECODE`、`RUNNING_DECODE` 和 `FINISHED` 都不能重新进入 Prefill。
- 校验发生在 Predictor、副本绑定和时间戳修改之前，非法调用不会产生部分状态修改。

### 2.1.2 Prefill 容量：从只限制 Decode 改为同时限制两个阶段

修复前 Hook 只计算 Decode 容量：

```python
# 修复前
decode_capacity = disagg_cfg.decode_admission_capacity()

if configured_capacity is not None:
    decode_capacity = min(
        int(configured_capacity),
        int(decode_capacity),
    )

server_args.max_running_requests = decode_capacity
```

为什么有问题：

假设 Prefill 总容量是 1，Decode 容量是 64，原生 Scheduler 仍可能激活 64 个请求。非最终 chunk 持有 Prefill 槽位后，其余请求进入 Backend 时只能报容量耗尽。

修复后同时取 Prefill、Decode 和用户配置的最小值：

```python
# 修复后
pd_capacity = min(
    disagg_cfg.prefill_admission_capacity(),
    disagg_cfg.decode_admission_capacity(),
)

if configured_capacity is not None:
    pd_capacity = min(
        int(configured_capacity),
        int(pd_capacity),
    )

server_args.max_running_requests = pd_capacity
```

Prefill 总容量的计算为：

```python
def prefill_admission_capacity(self) -> int:
    return (
        self.prefill.max_running_per_replica
        * self.prefill.replicas
    )
```

为什么修复后成立：

- 原生 Scheduler 不会激活超过整个 Prefill pool 可容纳数量的请求。
- Backend 的持久槽位限制与 SGLang 活动请求上限一致。
- 用户显式配置更小时仍然尊重用户配置。
- 多个 Prefill 服务副本的容量会正确累加，但不会错误乘上副本内部 `dp_size`。

### 2.1.3 Chunked Prefill：从单次 wave 限制改为跨调用持久槽位

修复前只有副本的 `busy_until`，没有记录哪些未完成请求仍占用槽位：

```python
# 修复前的等价逻辑
idx = min(
    range(len(prefill_busy_until)),
    key=lambda i: prefill_busy_until[i],
)
prefill_busy_until[idx] = start + duration
```

为什么有问题：

`busy_until` 只能描述计算时间，不能描述 Chunked 请求的生命周期。一个非最终 chunk 计算结束后，副本时钟会变为空闲，但该请求仍属于 Prefill 运行集合，新请求可能突破 `max_running_per_replica`。

修复后同时维护请求亲和性、运行计数和预留集合：

```python
self._prefill_replica_by_rid: dict[str, int] = {}
self._prefill_running_count: dict[int, int] = defaultdict(int)
self._prefill_slot_reserved: set[str] = set()

def _reserve_prefill_slot(self, req):
    replica_idx = self._prefill_replica_by_rid.get(req.rid)
    if replica_idx is not None:
        return replica_idx

    candidates = [
        idx
        for idx in range(len(self._prefill_pool.busy_until))
        if self._prefill_running_count[idx]
        < self._prefill_max_running_per_replica
    ]
    if not candidates:
        return None

    replica_idx = min(
        candidates,
        key=lambda idx: (
            self._prefill_running_count[idx],
            self._prefill_pool.busy_until[idx],
            idx,
        ),
    )
    self._prefill_replica_by_rid[req.rid] = replica_idx
    self._prefill_slot_reserved.add(req.rid)
    self._prefill_running_count[replica_idx] += 1
    return replica_idx
```

最终 chunk 或外部终止时释放：

```python
if req.prefill_is_final_chunk:
    self._release_prefill_slot(req)
```

为什么修复后成立：

- 非最终 chunk 的请求 ID 一直保留在 reserved set 中。
- 后续 chunk 通过 `rid → replica` 映射回到同一副本。
- running count 在完整 Prefill 生命周期结束前不会被新请求重复占用。
- 最终 Prefill chunk 正常完成时释放；外部 abort 通过独立终止接口释放。

### 2.1.4 Prefill batch：从 token 求和改为保留每条 sequence

修复前：

```python
# 修复前
total_tokens = sum(req.input_length for req in wave)
duration = predictor.predict_prefill_seconds(total_tokens)
```

输入 `[100, 300, 500]` 会退化为单请求 `[900]`。

为什么有问题：

- Predictor 看到的 batch size 从 3 变成 1。
- 丢失长短 sequence 混合特征。
- 无法估算 batch 内 padding、并行计算和显存访问影响。

修复后：

```python
def _predict_prefill_batch(self, wave):
    input_lengths = [int(req.input_length) for req in wave]
    batch_predict = getattr(
        self._bundle.prefill,
        "predict_prefill_batch_seconds",
        None,
    )
    if batch_predict is not None:
        return float(batch_predict(input_lengths))

    # 仅供旧 Predictor 兼容
    return float(
        self._bundle.prefill.predict_prefill_seconds(
            sum(input_lengths)
        )
    )
```

Adapter 为每个长度创建独立 FakeRequest：

```python
batch = ScheduleBatch(
    reqs=[
        FakeRequest(input_length=length, past_kv_length=0)
        for length in input_lengths
    ]
)
```

为什么修复后成立：

- Predictor 接收到的 batch size 与原请求数一致。
- `[100, 300, 500]` 不再变成 `[900]`。
- Backend A 和 Backend B 使用相同的每请求长度语义。
- 旧 Predictor 仍可通过明确的 fallback 运行。

### 2.1.5 Prefill 排队指标：从请求创建时间改为真实 queue 时间线

修复前的等价计算：

```python
# 修复前
arrival_time = now_when_extend_batch_appears
prefill_start_time = max(replica_free_time, arrival_time)
prefill_queue_wait = prefill_start_time - arrival_time
```

为什么有问题：

请求只有在离开 SGLang waiting queue、进入 extend batch 时才创建 PD state。此时 `arrival_time` 和 `prefill_start_time` 几乎相同，所以指标结构性接近 0。

修复后分离三个时间概念：

```python
arrival = request_stats.created_time

prefill_queue_start_time = prefill_queue_baseline(
    arrival,
    request_stats.queue_start,
)

admission_time = prefill_admission_baseline(
    request_stats.created_time,
    request_stats.queue_end,
)

prefill_start_time = max(
    replica_free_time,
    admission_time,
)
```

指标使用：

```python
prefill_queue_wait = max(
    prefill_start_time - prefill_queue_start_time,
    0.0,
)
```

为什么修复后成立：

- `queue_start` 表示进入服务端排队的时间。
- `queue_end` 表示原生 Scheduler 实际选中请求的时间。
- Prefill 不可能在 `queue_end` 之前开始。
- E2E 仍使用 `created_time/arrival_time`，不会因为修正阶段指标而改变全链路基准。

### 2.1.6 KV handoff：从全局最慢同步改为 replica-local batch

修复前：

```python
# 修复前
global_end = max(req.prefill_end_time for req in states)
total_tokens = sum(req.input_length for req in states)
kv_ready = backend.compute_batch_kv_ready_time(
    total_tokens,
    global_end,
)

for req in states:
    backend.on_prefill_done(req, global_end, kv_ready)
```

为什么有问题：

不同副本不是一个物理 fused batch。10 μs 完成的请求被迫等待 30 μs 完成的另一个副本，额外等待还会被算入 KV transfer。

修复后每个 replica-local wave 使用唯一 batch ID：

```python
req.prefill_replica_idx = replica_idx
req.prefill_batch_id = batch_id
req.prefill_end_time = end
```

最终按 ID 分组：

```python
groups = {}
for state in states:
    batch_id = state.prefill_batch_id
    key = (
        ("batch", batch_id)
        if batch_id is not None
        else ("legacy", 0)
    )
    groups.setdefault(key, []).append(state)
```

为什么修复后成立：

- 只有同一副本、同一 Predictor wave 的请求共享完成时间。
- 快副本可以在自己的 Prefill 完成后立即提交 handoff。
- legacy 状态仍保留原来的共享行为，不会破坏旧接口。

### 2.1.7 KV 网络：从每个请求独占带宽改为共享链路

修复前：

```python
# 修复前
transfer_dur = transfer_model.estimate(tokens, config)
return now + transfer_dur
```

为什么有问题：

每次调用都从 `now` 独立开始。10 个同时传输的请求相当于各自获得一张完整带宽的 NIC，总吞吐被放大约 10 倍。

修复后：

```python
self._kv_link_busy_until = 0.0

transfer_dur = self._transfer_model.estimate(
    total_tokens,
    self._kv_model_cfg,
)
start = max(now, self._kv_link_busy_until)
self._kv_link_busy_until = start + transfer_dur
return self._kv_link_busy_until
```

提交前还会按 Prefill 完成时间排序：

```python
scheduled_groups.sort(key=lambda item: item[0])
```

为什么修复后成立：

- 任意时刻共享链路只提供一份配置带宽容量。
- 后到的传输必须等待 `busy_until`。
- 排序保证较早完成的 Prefill 不会因为 Python 列表顺序排到较晚请求之后。
- P95 现在包含真实的链路服务时间和排队时间。

### 2.1.8 外部 abort：为 OSL 主路径补充异步终止清理

修复前只有长度条件：

```python
# 修复前
req.decode_step_count += 1
if req.decode_step_count >= req.output_length:
    req.phase = RequestPhase.FINISHED
```

在标准 HiSim 仿真中，`ignore_eos=True`，正常请求严格在达到 OSL 后结束，所以上面的 OSL 判断本身没有问题。真正缺失的是外部 abort：abort 可以在 WAITING_PREFILL、RUNNING_PREFILL、KV_TRANSIT、WAITING_DECODE 或 RUNNING_DECODE 任一阶段到达，但旧 Backend 没有统一的异步清理入口，可能遗留 Decode 槽位、Prefill affinity 和状态对象。

修复后首先从 abort message 中提取请求 ID：

```python
def abort_request_ids(message):
    if "abort" not in message.__class__.__name__.lower():
        return set()

    result = set()
    for name in ("rid", "rids", "request_id", "request_ids"):
        value = getattr(message, name, None)
        if isinstance(value, str):
            result.add(value)
        elif value is not None:
            result.update(str(item) for item in value)
    return result
```

统一终止接口释放全部资源：

```python
def terminate_request(self, req, now):
    self._controller.terminate_request(req, now)
    self._release_prefill_slot(req)
    self._prefill_replica_by_rid.pop(req.rid, None)
    self._single_decode_running_rids.discard(req.rid)

    replica_idx = self._decode_replica_by_rid.pop(
        req.rid,
        None,
    )
    if replica_idx is not None:
        self._decode_running_count[replica_idx] = max(
            0,
            self._decode_running_count[replica_idx] - 1,
        )
```

为什么修复后成立：

- 正常请求仍由 Controller 在 `decode_step_count == OSL` 时完成，不改变 HiSim 主语义。
- abort 在 `recv_requests()` 阶段根据 abort message 主动清理。
- 接口幂等，重复终止不会重复扣减容量。
- Controller 的三个等待队列和两个 Backend 的全部 affinity/count 都有清理路径。
- 正常完成与 abort 使用两个独立入口：正常完成只由 OSL 驱动，abort 只由外部 abort message 驱动。

### 2.1.9 OSL 主路径：撤回基于 `output_ids` 的错误对齐

修复前：

```python
# 修复前
stats.output_length = sampling_params.max_new_tokens
state.decode_step_count += 1
state.current_past_kv_length += 1
```

标准 HiSim 使用 `ignore_eos=True`。最终 Prefill logits 经 sampling 产生首 token，后续每轮普通 mock Decode 再产生一个 token。复查期间一度加入“读取 `output_ids` 并覆盖 PD 进度”的兼容代码，但它无法表达 token 的时间、计算归属和 KV 是否已经物化，现已完整撤回。最终改为显式记录 Prefill sampling 事件。

最终保留的核心代码：

```python
# Prefill sampling：计一个输出 token，不增加 KV，不运行 Decode Predictor。
req.decode_step_count += 1

# 后续完整 Decode forward：再计一个输出 token，并增加一格 KV。
req.decode_step_count += 1
req.current_past_kv_length += 1
if req.decode_step_count >= req.output_length:
    req.phase = RequestPhase.FINISHED
    req.decode_end_time = now
```

为什么最终代码成立：

- `max_new_tokens`、workload `output_length` 和 OSL 在标准仿真中是同一个值。
- `ignore_eos=True`，不会依赖真实 EOS 或 stop condition。
- mock sampler 不执行真实模型语义，也没有实现 speculative decoding 接受 token 的逻辑。
- PD 进度由明确的 Prefill sampling 事件和后续 Decode forward 事件推进，不再由 SGLang 内部容器长度覆盖。

### 2.1.10 时间戳：从混合时间原点改为统一相对时间

修复前只平移普通字段：

```python
# 修复前
item.created_time -= min_created_time
item.queue_start -= min_created_time
item.queue_end -= min_created_time
item.last_event_time -= min_created_time
```

为什么有问题：

输出 JSON 中普通字段以 0 为起点，而 `pd_prefill_start_time`、`pd_kv_ready_time` 等仍是绝对虚拟时间，直接比较和画图都会错位。

修复后集中平移所有 PD 绝对时间字段：

```python
PD_ABSOLUTE_TIME_FIELDS = (
    "pd_arrival_time",
    "pd_prefill_queue_start_time",
    "pd_prefill_start_time",
    "pd_prefill_end_time",
    "pd_kv_ready_time",
    "pd_decode_start_time",
    "pd_decode_end_time",
)

def shift_pd_time_origin(stats, origin):
    for name in PD_ABSOLUTE_TIME_FIELDS:
        value = getattr(stats, name, None)
        if value is not None:
            setattr(stats, name, float(value) - float(origin))
```

为什么修复后成立：

- 普通时间戳和 PD 时间戳使用完全相同的 origin。
- duration 指标不受平移影响。
- `request.jsonl` 可以直接用于阶段时间线可视化。

### 2.1.11 Predictor 和可选依赖：从无条件初始化改为按需加载

修复前每个 batch 都先执行聚合 Predictor：

```python
# 修复前
predicted_latency = float(
    INFERENCE_PREDICTOR.predict_infer_time(hisim_batch)
)

# 后面的 PD 分支立即覆盖 predicted_latency
predicted_latency = pd_latency
```

同时 `ConfigManager` 在模块导入时加载 AIC：

```python
# 修复前
from hisim.time_predictor import (
    InferTimePredictor,
    AIConfiguratorTimePredictor,
)
```

为什么有问题：

- 聚合 Predictor 结果被立即丢弃，产生无意义开销。
- 没有 AIC SDK 时，甚至无法导入只负责读取 JSON 的 `ConfigManager`。
- Predictor 初始化错误和配置解析错误混在一起。

修复后只在非 PD 定价 batch 调用聚合 Predictor：

```python
pd_priced_batch = (
    PD_BACKEND is not None
    and (
        batch.forward_mode.is_extend()
        or batch.forward_mode.is_decode()
    )
)

if not pd_priced_batch:
    predicted_latency = float(
        INFERENCE_PREDICTOR.predict_infer_time(hisim_batch)
    )
```

AIC Predictor 移到实际构造分支：

```python
if predictor_config.get("name") == "aiconfigurator":
    from hisim.time_predictor import (
        AIConfiguratorTimePredictor,
    )
```

为什么修复后成立：

- PD batch 只有角色 Predictor 是权威定价来源。
- 聚合 Predictor 不再重复计算。
- 读取 `DisaggConfig` 不需要安装 AIC SDK。
- 真正选择 AIC Predictor 时仍会正常检查依赖，缺少 SDK 会在正确的功能边界报错。

## 3. 问题一：PD 状态标志位没有形成严格状态机

### 3.1 问题现象

旧实现中，请求虽然包含 `RequestPhase`，但部分 Backend 路径没有在修改状态前验证当前 phase。理论上可能出现：

- `RUNNING_DECODE` 请求再次进入 Prefill；
- 重复提交 `WAITING_PREFILL`；
- 非 Decode 运行态请求进入 Decode Predictor；
- 非法调用已经修改部分时间戳后才报错。

这会导致 `prefill_end_time`、`decode_start_time` 和容量计数被覆盖或重复累计。

### 3.2 修复方法

Controller 和 Backend 都执行 phase 校验。当前合法主状态流为：

```text
WAITING_PREFILL
  → RUNNING_PREFILL
  → KV_TRANSIT
  → WAITING_DECODE
  → RUNNING_DECODE
  → FINISHED
```

Prefill Backend 只接受：

```python
if req.phase not in (
    RequestPhase.WAITING_PREFILL,
    RequestPhase.RUNNING_PREFILL,
):
    raise ValueError(...)
```

`RUNNING_PREFILL` 只有属于已预留 Chunked Prefill 槽位时才允许继续。

### 3.3 代码位置

- [`pd_controller.py`](hisim/src/hisim/simulation/pd_controller.py)：状态迁移和 `_require_phase()`。
- [`pd_backend_a.py`](hisim/src/hisim/simulation/pd_backend_a.py)：Backend A Prefill/Decode phase guard。
- [`pd_backend_b.py`](hisim/src/hisim/simulation/pd_backend_b.py)：Backend B 对等 phase guard。
- [`pd_types.py`](hisim/src/hisim/simulation/pd_types.py)：`RequestPhase` 和 `PDRequestState`。

## 4. 问题二：Prefill 容量检查与原生调度不一致

### 4.1 问题现象

旧 Hook 只根据 Decode 容量修改 SGLang 的 `max_running_requests`。当 Prefill 容量比 Decode 小时，SGLang 仍可能形成超过 Prefill 总容量的活动请求集合。

最典型的失败场景是：

```text
Prefill：1 replica × 1 request
Decode：1 replica × 64 requests
```

第一个非最终 Prefill chunk 持有唯一槽位后，新请求仍可能被原生 Scheduler 选入，随后 Backend 抛出容量耗尽异常。

### 4.2 修复方法

为 Prefill 增加总 admission capacity：

```python
def prefill_admission_capacity(self) -> int:
    if not self.enabled or self.prefill is None:
        return DEFAULT_MAX_RUNNING
    return (
        self.prefill.max_running_per_replica
        * self.prefill.replicas
    )
```

Hook 初始化时同时考虑用户配置、Prefill 总容量和 Decode 总容量：

```python
pd_capacity = min(
    disagg_cfg.prefill_admission_capacity(),
    disagg_cfg.decode_admission_capacity(),
)

if configured_capacity is not None:
    pd_capacity = min(int(configured_capacity), int(pd_capacity))

server_args.max_running_requests = pd_capacity
```

### 4.3 代码位置

- [`pd_config.py:111`](hisim/src/hisim/simulation/pd_config.py)：`prefill_admission_capacity()`。
- [`sglang_hook.py:659`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：统一计算原生 PD capacity。
- [`test_pd_config.py:157`](hisim/tests/unit/simulation/test_pd_config.py)：多 Prefill 副本总容量测试。

## 5. 问题三：Chunked Prefill 槽位和副本亲和性丢失

### 5.1 问题现象

单次 batch 切 wave 只能限制一次调用中的 batch size，无法表示一个未完成的 Chunked Prefill 请求仍然占用服务副本资源。

如果不保存跨调用状态，会发生：

1. 请求 A 执行非最终 chunk；
2. 下一轮请求 B 被放入同一个已满副本；
3. A 和 B 同时被视为 `RUNNING_PREFILL`；
4. A 的下一 chunk 还可能被重新分配到其他副本。

### 5.2 修复方法

Backend A 和 Backend B 均维护：

```text
request_id → prefill replica
per-replica running count
reserved slot request ID set
request final-chunk flag
```

请求状态新增：

```python
prefill_is_final_chunk: bool = True
prefill_replica_idx: Optional[int] = None
prefill_batch_id: Optional[int] = None
```

Hook 每轮使用 SGLang 的 `is_chunked` 刷新最终 chunk 标志：

```python
s.prefill_is_final_chunk = (
    getattr(req, "is_chunked", 0) == 0
)
```

只有最终 chunk 完成后才释放持久槽位和请求亲和性；abort 等外部终止路径也会强制释放。

### 5.3 代码位置

- [`pd_types.py:24`](hisim/src/hisim/simulation/pd_types.py)：新增 Prefill 时间线及 chunk 字段。
- [`pd_backend_a.py`](hisim/src/hisim/simulation/pd_backend_a.py)：Backend A 槽位预留、释放、亲和性。
- [`pd_backend_b.py`](hisim/src/hisim/simulation/pd_backend_b.py)：Backend B 对等实现。
- [`sglang_hook.py:1167`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：最终 chunk 判断。
- [`test_pd_backend_a.py:379`](hisim/tests/unit/simulation/test_pd_backend_a.py)：外部终止后 Prefill 容量释放测试。
- [`test_pd_backend_b_skeleton.py:214`](hisim/tests/unit/simulation/test_pd_backend_b_skeleton.py)：Backend B 容量释放测试。

## 6. 问题四：Prefill 请求没有按服务副本形成真正的 batch

### 6.1 问题现象

旧逻辑可能把整个 SGLang Prefill batch 直接放到一个最早空闲副本，导致：

- 多个服务副本没有并行利用；
- 请求集中到单个副本；
- Chunked Prefill 无法保持亲和性；
- Predictor batch 与实际 replica-local batch 不一致。

### 6.2 修复方法

Backend 逐请求执行容量感知、负载感知分配，然后按副本分桶：

```text
SGLang extend batch
    ↓
逐请求选择可用服务副本
    ↓
按 replica_idx 分桶
    ↓
每个桶调用一次 Prefill Predictor
```

选择副本时综合考虑：

- 当前 running count；
- `max_running_per_replica`；
- 副本 `busy_until`；
- Chunked 请求已有亲和关系；
- 稳定的 replica index 作为最终 tie-breaker。

配置语义同时被明确：

- `replicas`：独立调度的服务副本数量；
- `dp_size`：单个 Predictor 服务副本内部的拓扑参数；
- 运行时不会使用 `replicas × dp_size` 重复计算路由实例。

### 6.3 代码位置

- [`pd_config.py`](hisim/src/hisim/simulation/pd_config.py)：`RolePredictorConfig` 语义说明。
- [`pd_backend_a.py`](hisim/src/hisim/simulation/pd_backend_a.py)：`_reserve_prefill_slot()`、`try_admit_prefill_batch()`。
- [`pd_backend_b.py`](hisim/src/hisim/simulation/pd_backend_b.py)：Backend B 对等路由。

## 7. 问题五：Predictor 没有收到真实 Prefill batch shape

### 7.1 问题现象

旧逻辑把 batch 内所有请求 token 数相加，再构造一个 FakeRequest：

```text
实际输入：[100, 300, 500]
旧 Predictor 输入：[900]
```

这样会丢失真实 batch size 和每条 sequence 的长度分布。

### 7.2 修复方法

新增批量预测接口：

```python
def predict_prefill_batch_seconds(
    self, input_lengths: Sequence[int]
) -> float:
    ...
```

它为每个原始请求构造一个独立 FakeRequest，再组成 ScheduleBatch。Backend 优先调用该接口；旧 Predictor 没有该接口时才回退到 token 求和路径。

Backend B 的 `_PrefillJob` 同样传递每请求长度 tuple，而不是一个总长度。

### 7.3 代码位置

- [`pd_aic_adapter.py:40`](hisim/src/hisim/simulation/pd_aic_adapter.py)：真实 Prefill batch 构造。
- [`pd_backend_a.py:185`](hisim/src/hisim/simulation/pd_backend_a.py)：Backend A batch Predictor 调用。
- [`pd_backend_b.py:97`](hisim/src/hisim/simulation/pd_backend_b.py)：Backend B worker batch Predictor 调用。
- [`test_pd_aic_adapter.py`](hisim/tests/unit/simulation/test_pd_aic_adapter.py)：batch size 和输入长度回归测试。

## 8. 问题六：`prefill_queue_wait` 结构性接近 0

### 8.1 问题现象

旧 Prefill 角色时钟从请求创建时间或副本空闲时间开始，没有使用 SGLang 实际选中请求的 `queue_end`。

例如请求在原生 waiting queue 中等待 10 秒，模型仍可能得到：

```text
prefill_start_time = 0
arrival_time = 0
prefill_queue_wait = 0
```

### 8.2 修复方法

请求状态新增独立的队列基准：

```python
prefill_queue_start_time: Optional[float] = None
```

Prefill admission 起点受 `queue_end` 约束：

```python
prefill_start = max(
    prefill_replica_free_time,
    native_queue_end,
)
```

指标计算改为：

```python
prefill_queue_wait = (
    prefill_start_time - prefill_queue_start_time
)
```

其中：

- `arrival_time` 继续作为 E2E/TTFT 基准；
- `prefill_queue_start_time` 作为服务端 Prefill 排队基准；
- `queue_end` 用于保证 Prefill 不会在原生 Scheduler 选中请求之前开始。

### 8.3 代码位置

- [`pd_timeline.py`](hisim/src/hisim/simulation/pd_timeline.py)：`prefill_queue_baseline()`、`prefill_admission_baseline()`。
- [`pd_metrics.py:34`](hisim/src/hisim/simulation/pd_metrics.py)：阶段时间计算。
- [`sglang_hook.py:1146`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：状态创建与队列基准传递。
- [`test_pd_metrics.py`](hisim/tests/unit/simulation/test_pd_metrics.py)：10 秒排队不再得到 0 的回归测试。

## 9. 问题七：KV handoff 被错误同步到全局最慢 Prefill

### 9.1 问题现象

两个不同副本的 Prefill 完成时间可能是：

```text
Replica 0：10 μs
Replica 1：30 μs
```

旧 `finalize_prefill_batch()` 使用一个全局最大完成时间，导致 Replica 0 的请求也从 30 μs 才开始 KV transfer。

### 9.2 修复方法

每个 replica-local wave 获得唯一 `prefill_batch_id`。最终处理时按 batch ID 分组，并按每组自己的完成时间提交 KV handoff：

```python
batch_id = getattr(state, "prefill_batch_id", None)
key = ("batch", batch_id) if batch_id is not None else ("legacy", 0)
groups.setdefault(key, []).append(state)
```

旧状态没有 batch ID 时仍保留 legacy 单组行为，确保兼容旧调用方。

### 9.3 代码位置

- [`pd_backend_a.py:199`](hisim/src/hisim/simulation/pd_backend_a.py)：Backend A 分配 batch ID。
- [`pd_backend_b.py:358`](hisim/src/hisim/simulation/pd_backend_b.py)：Backend B 分配 batch ID。
- [`pd_runtime.py:289`](hisim/src/hisim/simulation/pd_runtime.py)：按 batch ID 完成 Prefill。

## 10. 问题八：KV 传输没有共享链路带宽竞争

### 10.1 问题现象

旧模型为每个 KV transfer 独立计算：

```text
latency + bytes / bandwidth
```

多个同时传输的请求都能使用完整带宽，会高估 KV handoff 吞吐并低估 `kv_transfer_time` P95。

### 10.2 修复方法

Controller 增加共享 KV 链路时钟：

```python
self._kv_link_busy_until = 0.0

start = max(now, self._kv_link_busy_until)
self._kv_link_busy_until = start + transfer_dur
return self._kv_link_busy_until
```

同时，多个 Prefill wave 按物理完成时间排序后再提交共享链路：

```python
scheduled_groups.sort(key=lambda item: item[0])

for group_end, _key, group in scheduled_groups:
    kv_ready = backend.compute_batch_kv_ready_time(
        total_tokens, group_end
    )
```

这样可以同时保证：

- 快副本不会因为调用列表顺序而排在慢副本之后；
- 多个传输不能各自占用完整配置带宽；
- `kv_transfer_time` 包含链路服务时间和必要的排队时间。

### 10.3 代码位置

- [`pd_controller.py:44`](hisim/src/hisim/simulation/pd_controller.py)：共享链路时钟。
- [`pd_runtime.py:302`](hisim/src/hisim/simulation/pd_runtime.py)：按 Prefill 完成时间排序。
- [`test_pd_controller.py:90`](hisim/tests/unit/simulation/test_pd_controller.py)：共享链路串行容量测试。
- [`test_pd_backend_a.py:529`](hisim/tests/unit/simulation/test_pd_backend_a.py)：颠倒调用顺序仍由快请求先传输。

## 11. 问题九：Decode 容量在默认模式下没有完整生效

### 11.1 问题现象

旧实现主要在 `per_replica_queue` 路径中限制容量，默认 `single_replica` 直接执行 targeted admission，可能忽略 `max_running_per_replica`。

### 11.2 修复方法

Backend 提供统一的 Decode 容量接口：

```text
decode_batch_capacity()
admit_decode_single_replica()
admit_decode_for_replica()
```

`single_replica` 使用运行 RID 集合维护容量：

```python
available = max(
    0,
    max_running_per_replica - len(single_decode_running_rids),
)
```

`per_replica_queue` 使用每副本 running count，并在副本已满时选择其他可用副本。请求正常完成或外部终止后都释放计数。

### 11.3 代码位置

- [`pd_backend_a.py:308`](hisim/src/hisim/simulation/pd_backend_a.py)：Backend A single-replica admission。
- [`pd_backend_b.py:461`](hisim/src/hisim/simulation/pd_backend_b.py)：Backend B 对等实现。
- [`sglang_hook.py:1438`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：Hook 默认 Decode 路径。

## 12. 问题十：外部 abort 后状态和容量泄漏

### 12.1 先说明标准 HiSim 的真实结束条件

你指出的 OSL 语义是正确的。HiSim 标准 SGLang 仿真明确设置：

```python
# hisim/src/hisim/simulation/sglang/sglang_bench.py
sampling_params = {
    "ignore_eos": True,
    "max_new_tokens": req.output_length,
}
```

因此标准路径中：

```text
OSL = request.output_length
    = sampling_params.max_new_tokens

ignore_eos = True
```

mock forward 不执行真实模型语义，普通 Decode 每个虚拟 step 产生一个 token。PD Controller 使用：

```python
req.decode_step_count += 1
req.current_past_kv_length += 1

if req.decode_step_count >= req.output_length:
    req.phase = RequestPhase.FINISHED
    req.decode_end_time = now
```

所以标准请求的正常结束条件就是：

```text
decode_step_count 达到 OSL
```

这段 OSL 结束逻辑没有错误，也没有被本轮修改替换。

### 12.2 真正缺失的是外部 abort 清理

abort 与 EOS 不同。它不是模型根据 token 内容产生的结束条件，而是外部控制请求，可以在任意阶段到达：

```text
WAITING_PREFILL
RUNNING_PREFILL
KV_TRANSIT
WAITING_DECODE
RUNNING_DECODE
```

旧代码只有“自然达到 OSL”的完成入口，没有“外部立即终止”的统一入口。如果一个请求在达到 OSL 前被 abort，可能造成：

- `_single_decode_running_rids` 不释放；
- `_decode_running_count` 不减少；
- Prefill chunk 槽位或 affinity 残留；
- `PD_REQUEST_STATES` 无法清理；
- 请求仍残留在 Controller 的等待队列或 KV_TRANSIT 列表中；
- 后续请求逐渐无法 admission。

### 12.3 修复第一步：识别 abort message

不同 SGLang 版本或单请求/批量 abort 使用的字段可能不同，因此新增统一提取函数：

```python
def abort_request_ids(message: Any) -> Set[str]:
    if "abort" not in message.__class__.__name__.lower():
        return set()

    result: Set[str] = set()
    for name in (
        "rid",
        "rids",
        "request_id",
        "request_ids",
    ):
        value = getattr(message, name, None)
        if value is None:
            continue
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, Iterable):
            result.update(
                str(item)
                for item in value
                if item is not None
            )
    return result
```

Hook 在 `recv_requests()` 中处理外部消息：

```python
aborted_rids = set()
for message in recv_reqs:
    aborted_rids.update(
        abort_request_ids(message)
    )

for rid in aborted_rids:
    state = PD_REQUEST_STATES.get(rid)
    if state is not None:
        PD_BACKEND.terminate_request(
            state,
            termination_time,
        )
```

### 12.4 修复第二步：Controller 从所有队列移除请求

Controller 新增幂等终止接口，可从任意 live phase 清除队列：

```python
def terminate_request(
    self,
    req: PDRequestState,
    now: float,
) -> None:
    self._prefill_waiting = deque(
        queued
        for queued in self._prefill_waiting
        if queued.rid != req.rid
    )
    self._kv_transit = [
        queued
        for queued in self._kv_transit
        if queued.rid != req.rid
    ]
    self._decode_waiting = deque(
        queued
        for queued in self._decode_waiting
        if queued.rid != req.rid
    )

    if req.phase == RequestPhase.FINISHED:
        return

    req.phase = RequestPhase.FINISHED

    if req.decode_start_time is not None:
        req.decode_end_time = now
```

幂等性的含义是：同一个 abort 即使因为重试被处理两次，第二次看到 `FINISHED` 后也不会重复修改状态。

### 12.5 修复第三步：Backend 释放全部容量和亲和关系

仅从 Controller 队列移除还不够，因为容量和副本映射由 Backend 持有。Backend A/B 都执行：

```python
def terminate_request(self, req, now):
    self._controller.terminate_request(req, now)

    # 释放 Chunked Prefill 槽位和亲和性
    self._release_prefill_slot(req)
    self._prefill_replica_by_rid.pop(
        req.rid,
        None,
    )

    # 释放 single_replica Decode 容量
    self._single_decode_running_rids.discard(
        req.rid
    )

    # 释放 per_replica_queue Decode 容量
    replica_idx = self._decode_replica_by_rid.pop(
        req.rid,
        None,
    )
    if replica_idx is not None:
        self._decode_running_count[replica_idx] = max(
            0,
            self._decode_running_count[replica_idx] - 1,
        )
```

### 12.6 修复第四步：清理 Hook 辅助状态和 future queue

```python
PD_PREFILL_KV_SERVICE.pop(rid, None)
PD_CHUNK_ACCUM.pop(rid, None)

FUTURE_QUEUE = [
    item
    for item in FUTURE_QUEUE
    if getattr(item[2], "rid", None)
    not in aborted_rids
]
heapq.heapify(FUTURE_QUEUE)
```

如果请求尚未到达、仍位于离线仿真的 `FUTURE_QUEUE`，也必须移除，否则未来时间到达时它会再次进入 Scheduler。

### 12.7 EOS/stop 与本修复无关

最终代码没有通过 `finished_reason`、EOS 或 stop condition 终止 PD 请求。原因是标准 HiSim 明确设置 `ignore_eos=True`，mock sampler 也不执行真实模型结束语义。正常完成只走 OSL 分支；`terminate_request()` 只由外部 abort/cancel 分支调用。

这样避免了把两类完全不同的事件混在一起：

```text
正常完成：虚拟 Decode step 达到 OSL → FINISHED
外部取消：收到 AbortReq → terminate_request() → FINISHED
```

如果未来确实要模拟 EOS、stop 或 speculative decoding，应先定义对应的 workload 和虚拟时间模型，再新增独立状态转换；不能直接用 SGLang 内部 `output_ids` 或 `finished_reason` 覆盖当前 OSL 计数。

### 12.8 修复前后的标准路径

正常 OSL 请求的路径没有改变：

```text
Decode step 1
    ↓
Decode step 2
    ↓
...
    ↓
Decode step OSL
    ↓
Controller 标记 FINISHED
    ↓
Backend 正常完成路径释放 Decode 容量
```

新增的是独立 abort 分支：

```text
任意 live phase
    ↓
收到 AbortReq
    ↓
提取 rid
    ↓
Controller 从所有队列移除
    ↓
Backend 释放 Prefill/Decode 槽位
    ↓
清理 affinity、chunk 累计和 future queue
    ↓
FINISHED
```

### 12.9 为什么修改后能够解决 abort 泄漏

对于每一种请求位置，都存在对应清理动作：

| abort 时所在位置 | 清理动作 |
|---|---|
| Prefill waiting queue | 从 `_prefill_waiting` 删除 |
| Chunked Prefill running | 释放 reserved slot 和 replica affinity |
| KV_TRANSIT | 从 `_kv_transit` 删除 |
| Decode waiting queue | 从 `_decode_waiting` 删除 |
| single-replica Decode | 从 `_single_decode_running_rids` 删除 |
| per-replica Decode | 减少 `_decode_running_count` 并删除绑定 |
| FUTURE_QUEUE | 删除尚未到达的请求项 |

因此不会再留下占用容量但永远不会继续执行的请求。

### 12.10 代码与测试位置

- [`pd_sglang_lifecycle.py`](hisim/src/hisim/simulation/pd_sglang_lifecycle.py)：SGLang 版本兼容适配。
- [`pd_controller.py:181`](hisim/src/hisim/simulation/pd_controller.py)：Controller 幂等终止。
- [`pd_backend_a.py:532`](hisim/src/hisim/simulation/pd_backend_a.py)：Backend A 全资源释放。
- [`pd_backend_b.py:708`](hisim/src/hisim/simulation/pd_backend_b.py)：Backend B 全资源释放。
- [`sglang_hook.py:850`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：abort 处理。
- [`test_pd_controller.py:221`](hisim/tests/unit/simulation/test_pd_controller.py)：从所有 live phase 幂等终止。
- [`test_pd_sglang_lifecycle.py`](hisim/tests/unit/simulation/test_pd_sglang_lifecycle.py)：单请求和批量 abort 字段兼容测试。

## 13. 问题十一复核：正常结束必须严格由 OSL 驱动

### 13.1 复核结论

这一项最初被误判为“EOS 提前结束和推测解码 token 数不准确”。该判断仍然错误：标准模拟使用 `ignore_eos=True`，也没有 speculative decoding 接受逻辑，普通请求必须且只会在达到 OSL 时结束。但进一步核对 SGLang 结果处理后发现了另一个独立问题：最终 Prefill logits 会经过 sampling 产生首 token，这个 token 必须计入 OSL，但不能按完整 Decode forward 计价，详见问题十五。

因此本轮最终处理不是增加兼容逻辑，而是：

1. 保留“达到 OSL 才结束”的逻辑；显式 Decode forward 仍然一次推进一个 token。
2. 撤回曾加入的 `request_reports_finished()`、`request_actual_output_length()` 和 `reconcile_decode_progress()`。
3. 删除 Hook 中根据 `finished_reason` 提前终止、根据 `output_ids` 覆盖 PD 进度以及补零 ITL 的代码。
4. 保留独立的外部 abort 清理路径，因为 abort 是控制事件，不是模型结束条件。
5. 为最终 Prefill sampling 增加一个明确的输出 token 事件，而不是读取 `len(output_ids)` 覆盖状态。

### 13.2 代码证据：标准请求明确忽略 EOS

`sglang_bench.py` 构造请求时写入：

```python
sampling_params = {
    "ignore_eos": True,
    "max_new_tokens": req.output_length,
}
```

两项含义分别是：

```text
ignore_eos=True                 不因模型输出 EOS 提前结束
max_new_tokens=req.output_length 目标生成长度就是 workload OSL
```

mock sampler 返回固定占位 token：

```python
def wrapped_sample(self, logits, sampling_info, ...):
    ids = torch.ones(
        (logits.shape[0],),
        dtype=torch.int32,
        device=logits.device,
    )
    return ids
```

这段代码不运行真实语言模型的结束逻辑，所以不能从 token 内容推导 EOS、stop 或推测解码接受数量。

### 13.3 最终保留的 OSL 核心代码

最终 Prefill sampling 先计入首 token，但不增加 KV 长度：

```python
def on_prefill_token_sampled(self, req, now):
    req.decode_step_count += 1
    if req.decode_step_count >= req.output_length:
        self._kv_transit = [
            queued
            for queued in self._kv_transit
            if queued.rid != req.rid
        ]
        req.phase = RequestPhase.FINISHED
        req.decode_end_time = now
```

后续每个完整 Decode forward 生成下一个 token，并物化上一个输出 token 的 KV：

```python
def on_decode_step_done(self, reqs, now):
    for req in reqs:
        self._require_phase(
            req,
            RequestPhase.RUNNING_DECODE,
            "decode completion",
        )

        # output_length 是 workload OSL；
        # 一个虚拟 Decode step 只推进一个 token。
        req.decode_step_count += 1
        req.current_past_kv_length += 1

        if req.decode_step_count >= req.output_length:
            req.phase = RequestPhase.FINISHED
            req.decode_end_time = now
```

假设 OSL 为 5：

```text
Prefill sampling → decode_step_count=1
Decode forward 1 → decode_step_count=2
Decode forward 2 → decode_step_count=3
Decode forward 3 → decode_step_count=4
Decode forward 4 → decode_step_count=5 → FINISHED
```

所以结束条件仍是 `decode_step_count >= output_length`，而 `output_length` 等于 OSL。这里的 `decode_step_count` 历史名称不够准确，其实际用途是累计已生成输出 token 数；完整 Decode forward 次数对于普通请求是 `OSL-1`。

### 13.4 一度加入的代码为什么有问题

复查中一度尝试用 SGLang 请求对象的输出数组反推 PD 进度：

```python
actual_output_length = len(req.output_ids)
state.decode_step_count = actual_output_length
state.current_past_kv_length = (
    state.input_length + actual_output_length
)
```

这段代码现已删除，原因是它混淆了两个计数域：

```text
PD decode_step_count
    = 已建模并具备明确完成时间的输出 token 数

SGLang len(output_ids)
    = SGLang 请求对象内部当前保存的输出 ID 数
```

二者不保证在所有 Hook 时点严格相等。最终 Prefill 首 token 必须通过明确的 sampling 事件计数；若直接用 `len(output_ids)` 覆盖，不知道该 token 的完成时间和计算归属，并且旧同步代码还同时写入 `input_length + actual_output_length`，会把 sampling token 错当成已经物化 KV 的完整 Decode 结果，导致：

- `current_past_kv_length` 被错误增加；
- 下一轮 Decode Predictor 输入长度偏大；
- ITL 数量与虚拟 Decode step 数不一致。

所以“读取真实输出长度”在真实推理系统里可能合理，但在当前 mock OSL 仿真中不是正确数据源。

### 13.5 为什么不能声称支持 speculative decoding

当前模拟器没有实现：

- draft model；
- token proposal；
- verification；
- acceptance length；
- 一次 forward 接受多个 token 对应的时延模型；
- 多 token 的逐 token 完成时间。

因此不能仅凭 `output_ids` 一次增长多个元素，就把它解释成已经支持 speculative decoding。若未来要支持，至少需要显式建模“本轮接受 token 数”和对应 Predictor 输入/输出，再让：

```text
decode_step_count += accepted_token_count
current_past_kv_length += accepted_token_count
```

当前代码没有这套模型，所以最终保持“Prefill sampling 明确产生一个 token；每个普通 Decode forward 再产生一个 token；严格按 OSL 结束”。

### 13.6 与问题十 abort 的边界

OSL 完成和 abort 都会进入 `FINISHED`，但触发来源完全不同：

| 场景 | 触发条件 | 是否修改目标 OSL | 资源释放入口 |
|---|---|---:|---|
| 正常完成 | `decode_step_count >= output_length` | 否 | Decode 正常完成路径 |
| 外部 abort | 收到 abort message | 统计时记录已完成 step 数 | `terminate_request()` |

abort 时 Hook 会先执行：

```python
state.output_length = state.decode_step_count
PD_BACKEND.terminate_request(
    state,
    termination_time,
)
```

这里把统计输出长度改成“abort 前已经完成的虚拟 token 数”，是为了避免指标仍声称请求完成了原始 OSL；但它不会改变正常请求的 OSL，也不会让普通请求提前结束。

### 13.7 修复后的结果和测试

最终行为满足：

- 标准请求严格在第 OSL 个输出 token 事件结束；
- Prefill sampling 首 token 不调用完整 Decode Predictor；
- `output_ids` 不再覆盖 PD 仿真进度；
- 不再声称模拟 EOS/stop/speculative decoding；
- 外部 abort 仍能从任意 live phase 幂等清理资源。

对应回归测试：

- `test_decode_forward_advances_one_output_token_until_osl`：每个完整 Decode forward 推进一个输出 token，到 OSL 时结束。
- `test_terminate_request_is_idempotent_from_every_live_phase`：abort 清理在所有 live phase 幂等。
- `test_abort_request_ids_supports_single_and_batch_messages`：兼容单请求和批量 abort message。

代码位置：

- [`sglang_bench.py:145`](hisim/src/hisim/simulation/sglang/sglang_bench.py)：`ignore_eos=True` 和 `max_new_tokens=OSL`。
- [`sglang_hook.py:243`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：mock sampler 固定占位 token。
- [`pd_controller.py:170`](hisim/src/hisim/simulation/pd_controller.py)：OSL Decode step 与结束条件。
- [`sglang_hook.py:850`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：外部 abort 清理。
- [`test_pd_controller.py:188`](hisim/tests/unit/simulation/test_pd_controller.py)：严格 OSL 回归测试。

## 14. 问题十二：PD 时间戳输出使用不同时间原点

### 14.1 问题现象

HiSim 输出 `request.jsonl` 前，会把请求时间平移到以第一个有效请求为 0 的相对时间轴。

假设：

```text
min_created_time       = 100.0 s
created_time           = 101.0 s
pd_prefill_start_time  = 103.0 s
pd_kv_ready_time       = 104.0 s
```

旧代码只平移普通字段，输出变成：

```text
created_time           = 1.0 s
pd_prefill_start_time  = 103.0 s
pd_kv_ready_time       = 104.0 s
```

同一个 JSON 文件中出现两套时间原点。

### 14.2 原核心代码

```python
# 修复前：只处理四个旧字段
min_created_time = metrics_stats[0].created_time

for item in stats:
    item.created_time -= min_created_time
    item.queue_start -= min_created_time
    item.queue_end -= min_created_time
    item.last_event_time -= min_created_time
```

### 14.3 为什么原代码有问题

PD 新增了以下绝对时间戳：

```text
pd_arrival_time
pd_prefill_queue_start_time
pd_prefill_start_time
pd_prefill_end_time
pd_kv_ready_time
pd_decode_start_time
pd_decode_end_time
```

这些字段在 `populate_request_stats()` 中直接复制自 PD 虚拟时间：

```python
stats.pd_prefill_start_time = state.prefill_start_time
stats.pd_prefill_end_time = state.prefill_end_time
stats.pd_kv_ready_time = state.kv_ready_time
stats.pd_decode_start_time = state.decode_start_time
```

如果不做相同平移，会造成：

1. `request.jsonl` 中字段不能直接比较。
2. 可视化时 PD 阶段整体向右偏移 `min_created_time`。
3. 用户可能误以为 Prefill 排队或 KV 传输耗时数十秒。
4. 跨请求时间线无法叠加到同一坐标系。

阶段 duration 本身可能仍然正确，因为：

```text
(end - origin) - (start - origin)
= end - start
```

但绝对时间线输出是错误的，所以仍然必须修复。

### 14.4 修改后的字段集中定义

```python
PD_ABSOLUTE_TIME_FIELDS = (
    "pd_arrival_time",
    "pd_prefill_queue_start_time",
    "pd_prefill_start_time",
    "pd_prefill_end_time",
    "pd_kv_ready_time",
    "pd_decode_start_time",
    "pd_decode_end_time",
)
```

把字段集中列出，而不是在 Hook 中写七个重复语句，原因是后续新增 PD 时间戳时更容易审查是否遗漏。

### 14.5 修改后的平移函数

```python
def shift_pd_time_origin(
    stats: "RequestStats",
    origin: float,
) -> None:
    for name in PD_ABSOLUTE_TIME_FIELDS:
        value = getattr(stats, name, None)

        if value is not None:
            setattr(
                stats,
                name,
                float(value) - float(origin),
            )
```

这里必须判断 `None`，因为一个尚未进入 Decode 的请求可能没有 `pd_decode_start_time` 或 `pd_decode_end_time`。

### 14.6 Hook 输出前的完整调用

```python
min_created_time = metrics_stats[0].created_time

for item in stats:
    item.created_time -= min_created_time
    item.queue_start -= min_created_time
    item.queue_end -= min_created_time
    item.last_event_time -= min_created_time

    shift_pd_time_origin(
        item,
        min_created_time,
    )
```

前面的例子修复后输出为：

```text
created_time           = 1.0 s
pd_prefill_start_time  = 3.0 s
pd_kv_ready_time       = 4.0 s
```

所有字段都以 100.0 秒为 origin。

### 14.7 为什么修复后没有原问题

- 所有普通时间戳和 PD 时间戳使用同一个 `min_created_time`。
- `None` 字段不会被错误参与减法。
- duration 是平移不变量，修复不会改变阶段耗时。
- `request.jsonl` 可以直接用于时间线绘图和跨字段比较。

### 14.8 代码与测试位置

- [`pd_metrics.py:44`](hisim/src/hisim/simulation/pd_metrics.py)：PD 时间戳写入 RequestStats。
- [`pd_metrics.py:62`](hisim/src/hisim/simulation/pd_metrics.py)：绝对时间字段清单。
- [`pd_metrics.py:73`](hisim/src/hisim/simulation/pd_metrics.py)：统一平移函数。
- [`sglang_hook.py:1792`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：输出阶段调用。
- [`test_pd_metrics.py:145`](hisim/tests/unit/simulation/test_pd_metrics.py)：七个字段平移结果测试。

## 15. 问题十三：PD 模式重复运行聚合 Predictor

### 15.1 这个问题涉及哪两个 Predictor

非 PD 模式只有一个统一 Predictor：

```text
INFERENCE_PREDICTOR
    └─ 同时估算当前 SGLang batch 的执行时间
```

PD 模式已经建立两个角色 Predictor：

```text
PD_BACKEND
    ├─ prefill predictor：使用 Prefill 设备和拓扑配置
    └─ decode predictor：使用 Decode 设备和拓扑配置
```

例如：

```text
统一 Predictor：H100，TP=8
Prefill Predictor：H100，TP=8，replicas=2
Decode Predictor：H20，TP=4，replicas=4
```

PD 模式的权威结果必须来自角色 Predictor，因为统一 Predictor 不知道：

- 当前工作属于 Prefill 还是 Decode；
- 请求被分到了哪个服务副本；
- Prefill 和 Decode 使用的不同硬件；
- replica-local batch 的真实组成；
- 各副本独立的 `busy_until`。

### 15.2 原核心代码

旧 Hook 对任何非空 batch 都先调用统一 Predictor：

```python
# 修复前
if not hisim_batch.is_empty():
    StateManager.inc_iteration()

    predicted_latency = (
        C_SchedulerHook.INFERENCE_PREDICTOR
        .predict_infer_time(hisim_batch)
    )
    predicted_latency = float(predicted_latency)
```

随后进入 PD Prefill 分支，再次调用角色 Predictor：

```python
# 修复前：第一次结果马上被覆盖
pd_latency = admit_prefill_batch_latency(
    C_SchedulerHook.PD_BACKEND,
    states,
    now_clock,
)

predicted_latency = pd_latency
```

Decode 分支也同样覆盖：

```python
# 修复前
pd_latency = decode_batch_latency(
    C_SchedulerHook.PD_BACKEND,
    states,
    step_start,
)

predicted_latency = pd_latency
```

旧执行流程实际是：

```text
同一个 Prefill/Decode batch
        ↓
调用统一 INFERENCE_PREDICTOR
        ↓
得到 aggregate_latency
        ↓
调用 PD role predictor
        ↓
得到 pd_latency
        ↓
用 pd_latency 覆盖 aggregate_latency
```

第一个结果没有参与最终时钟推进。

### 15.3 为什么原代码有问题

#### 问题 A：每个 batch 重复预测

一次 PD batch 会执行两次 Predictor。即使 Predictor 只是查表，也会增加无意义开销；如果 Predictor 会加载模型、执行 XGBoost 或访问性能数据库，开销会更明显。

#### 问题 B：错误地把非权威 Predictor 变成启动依赖

PD Backend 的两个 Predictor 已经足够完成定价，但旧代码仍要求：

```text
INFERENCE_PREDICTOR 必须成功初始化
```

如果统一 Predictor 的数据库配置不存在，服务可能启动失败，即使 Prefill/Decode Predictor 配置完全正确。

#### 问题 C：两个 Predictor 的输入语义不同

统一 Predictor 使用原生 SGLang batch；PD Predictor 使用拆分后的 replica-local batch。二者的 batch size、设备和时间线不同，不能混称为同一个预测结果。

#### 问题 D：日志容易误导

日志可能同时打印：

```text
agg_pred = 8 ms
pd_pred  = 3 ms
```

但最终只使用 3 ms。读者可能误认为 8 ms 也参与了调度或时钟推进。

### 15.4 修改后的 Predictor 选择代码

先判断当前 batch 是否由 PD Backend 定价：

```python
pd_priced_batch = (
    C_SchedulerHook.PD_BACKEND is not None
    and (
        batch.forward_mode.is_extend()
        or batch.forward_mode.is_decode()
    )
)
```

然后只选择一个权威路径：

```python
if pd_priced_batch:
    # 这里只初始化占位值。
    # 后续 Prefill 或 Decode PD 分支必须覆盖它。
    predicted_latency = 0.0
else:
    predicted_latency = float(
        C_SchedulerHook.INFERENCE_PREDICTOR
        .predict_infer_time(hisim_batch)
    )
```

新的执行流程是：

```text
是否为 PD Prefill/Decode batch？
        ├─ 否 → 只调用 INFERENCE_PREDICTOR
        │
        └─ 是
             ├─ Extend → 只调用 Prefill Predictor
             └─ Decode → 只调用 Decode Predictor
```

### 15.5 为什么 PD 分支可以先设置 `0.0`

`0.0` 不是最终预测结果，只是进入分支前的安全占位值。

Prefill 分支一定执行：

```python
pd_latency = admit_prefill_batch_latency(...)
predicted_latency = pd_latency
```

Decode 分支一定执行：

```python
pd_latency = decode_batch_latency(...)
predicted_latency = pd_latency
```

不能使用 `NaN` 作为占位，因为一旦状态异常漏过覆盖，`NaN` 会污染：

```text
current_inference_dur
global_clock
request_response_time
最终 metrics
```

### 15.6 防止占位值掩盖 Decode 状态错误

如果 SGLang 形成了 Decode batch，但 PD 状态中没有任何可 admission 请求，不能把 `0.0` 当成真实延迟继续运行。

修复后明确报错：

```python
if token_times:
    predicted_latency = (
        max(bucket_step_ends)
        - min(bucket_step_starts)
    )
elif batch.reqs:
    raise RuntimeError(
        "PD decode batch contained no admissible "
        "request state; native and PD capacity/state "
        "tracking diverged"
    )
```

默认 `single_replica` 路径也有同样保护：

```python
if states:
    pd_latency = decode_batch_latency(...)
    predicted_latency = pd_latency
else:
    raise RuntimeError(
        "PD decode batch contained no admissible "
        "request state; native and PD capacity/state "
        "tracking diverged"
    )
```

为什么必须 fail-fast：

- SGLang 已经执行了 Decode，但 PD 没有定价，说明两个状态机发生分叉。
- 静默使用 0 秒会少算一次 Decode latency。
- 回退到统一 Predictor 会掩盖容量或 phase bug。
- 明确异常能直接定位 native/PD 状态不一致。

### 15.7 修改后的初始化容错

统一 Predictor 初始化失败时，PD 模式不再构造一个重复的 fallback Predictor：

```python
try:
    C_SchedulerHook.INFERENCE_PREDICTOR = (
        ConfigManager.get_inference_time_predictor(
            model,
            hw,
            sched_config,
        )
    )
except Exception as e:
    if disagg_cfg.enabled:
        C_SchedulerHook.INFERENCE_PREDICTOR = None
        logger.warning(
            "Failed to initialize global inference "
            "predictor (%s); PD role predictors "
            "remain authoritative.",
            e,
        )
    else:
        raise
```

这里的语义是：

- 非 PD 模式仍然依赖统一 Predictor，初始化失败必须报错。
- PD 模式的 Extend/Decode 不使用统一 Predictor，因此失败不会影响角色定价。
- PD Backend 自身的 Prefill/Decode Predictor 如果初始化失败，仍然会在 Backend 构造阶段正常报错，不会被忽略。

### 15.8 为什么修复后没有原问题

修复后满足：

```text
每个 batch 只存在一个权威定价来源
```

具体为：

| Batch 类型 | 权威 Predictor |
|---|---|
| 非 PD batch | `INFERENCE_PREDICTOR` |
| PD Extend | `PD_BACKEND.prefill` |
| PD Decode | `PD_BACKEND.decode` |

因此：

- 不再重复预测；
- 不再计算随后被覆盖的结果；
- 统一 Predictor 配置不会阻止合法 PD 运行；
- PD 状态失配不会被 fallback 掩盖；
- 最终 `predicted_latency` 一定来自正确角色和正确副本 batch。

### 15.9 代码与验证位置

- [`sglang_hook.py:702`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：统一 Predictor 初始化容错。
- [`sglang_hook.py:1043`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：选择唯一权威 Predictor。
- [`sglang_hook.py:1266`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：Prefill 结果覆盖安全占位。
- [`sglang_hook.py:1412`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：per-replica Decode 状态失配保护。
- [`sglang_hook.py:1492`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：single-replica Decode 结果写入。
- [`sglang_hook.py:1501`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：single-replica 状态失配保护。

## 16. 问题十四：读取配置时强制依赖 AIC SDK

### 16.1 问题现象

`ConfigManager` 同时负责：

1. 读取 JSON 配置；
2. 构造 SchedulerConfig；
3. 构造可选的 AIC Predictor。

旧代码在模块顶层直接导入 AIC Predictor，导致这三个职责被错误绑定。

即使测试只执行：

```python
from hisim.simulation.manager import ConfigManager

cfg = ConfigManager.get_disagg_config()
```

也会因为没有安装 AIC SDK 而在导入阶段失败。

### 16.2 原核心代码

```python
# 修复前：模块一加载就解析两个导入
from hisim.time_predictor import (
    InferTimePredictor,
    AIConfiguratorTimePredictor,
)
```

`hisim.time_predictor` 通过 `__getattr__` 延迟解析类名，但上面的显式导入仍然会触发：

```python
from hisim.time_predictor.aiconfigurator import (
    AIConfiguratorTimePredictor,
)
```

而该模块继续导入外部 SDK：

```python
from aiconfigurator.sdk import models
```

本机没有 SDK 时，实际错误是：

```text
ModuleNotFoundError:
No module named 'aiconfigurator'
```

错误发生在 pytest 收集测试文件时，甚至还没有调用 Predictor。

### 16.3 为什么原代码设计不正确

AIC 是可选 Predictor 后端，不应该成为以下功能的强制依赖：

- 读取 `DisaggConfig`；
- 验证 Prefill/Decode role 参数；
- 使用测试 stub Predictor；
- 运行不依赖 AIC 的 Backend 单元测试；
- 导入 `ConfigManager`。

只有用户配置：

```json
{
  "predictor": {
    "name": "aiconfigurator"
  }
}
```

并且真正调用 `get_inference_time_predictor()` 时，才应该要求安装 AIC SDK。

### 16.4 修改后的模块顶层代码

```python
from __future__ import annotations

import json

from hisim.time_predictor import InferTimePredictor
```

模块顶层只导入基础协议 `InferTimePredictor`，它不依赖外部 AIC SDK。

### 16.5 修改后的按需导入代码

```python
@classmethod
def get_inference_time_predictor(
    cls,
    model: ModelInfo,
    hw: AcceleratorInfo,
    sched_config: SchedulerConfig,
) -> InferTimePredictor:
    with open(Envs.config_path()) as f:
        config: dict = json.load(f)

    predictor_config = config.get("predictor", {})

    if predictor_config.get("name") == "aiconfigurator":
        # 只有真正选择 AIC 时才加载可选 SDK。
        from hisim.time_predictor import (
            AIConfiguratorTimePredictor,
        )

        return AIConfiguratorTimePredictor(
            model,
            hw=hw,
            config=sched_config,
            database_path=predictor_config.get(
                "database_path"
            ),
            database_mode=predictor_config.get(
                "database_mode",
                "SILICON",
            ),
            prefill_scale_factor=predictor_config.get(
                "prefill_scale_factor",
                1,
            ),
            decode_scale_factor=predictor_config.get(
                "decode_scale_factor",
                1,
            ),
            xgb_model_path=predictor_config.get(
                "xgb_model_path"
            ),
        )

    raise ValueError(
        "Unknown predictor name: "
        f"{predictor_config.get('name')}"
    )
```

### 16.6 为什么增加 `from __future__ import annotations`

移除 AIC 强制导入后，配置测试继续暴露了 Python 3.8 兼容问题。代码中存在：

```python
def get_model_info(
    cls,
    hf_config: dict | None,
) -> ModelInfo:
    ...
```

Python 3.10 原生支持 `dict | None`，但较旧 Python 会在函数定义阶段立即计算该表达式并报错。

增加：

```python
from __future__ import annotations
```

后，类型注解以字符串形式延迟求值：

```text
导入 ConfigManager
    ↓
不立即计算 dict | None
    ↓
Python 3.8 可以正常加载模块
```

### 16.7 修改前后的依赖边界

修复前：

```text
读取 PD JSON 配置
    ↓
导入 ConfigManager
    ↓
导入 AIConfiguratorTimePredictor
    ↓
强制导入 aiconfigurator SDK
    ↓
没有 SDK → 失败
```

修复后：

```text
读取 PD JSON 配置
    ↓
导入 ConfigManager
    ↓
只导入基础 Predictor 协议
    ↓
配置读取成功

真正选择 AIC Predictor
    ↓
按需导入 AIConfiguratorTimePredictor
    ↓
此时才检查 AIC SDK
```

### 16.8 为什么修复后没有原问题

- `get_disagg_config()` 不触发 AIC 导入。
- stub Predictor 和非 AIC 测试不需要安装外部 SDK。
- 真正选择 AIC 时仍然严格检查 SDK，不会把配置错误静默隐藏。
- Python 3.8 可以导入带 `dict | None` 注解的配置模块。
- 配置解析错误与 Predictor 运行依赖错误被分离到正确边界。

### 16.9 代码与测试位置

- [`manager/config.py:1`](hisim/src/hisim/simulation/manager/config.py)：延迟注解。
- [`manager/config.py:13`](hisim/src/hisim/simulation/manager/config.py)：模块顶层只导入基础协议。
- [`manager/config.py:181`](hisim/src/hisim/simulation/manager/config.py)：按需加载 AIC Predictor。
- [`test_config_manager_disagg.py`](hisim/tests/unit/simulation/test_config_manager_disagg.py)：无 AIC SDK 环境下配置 round-trip 测试。

测试结果：

```text
2 passed
```

该测试此前在收集阶段直接失败，修复后可以完整执行。

## 17. 问题十五：最终 Prefill 首 token 的 OSL 计数与时延归属

### 17.1 复核后的准确结论

SGLang 0.5.x 在最终 Extend 结果中执行：

```python
if req.is_chunked <= 0:
    req.output_ids.append(next_token_id)
    req.check_finished()
```

HiSim mock sampler 虽然不执行真实模型语义，但仍返回一个固定占位 token，所以该 append 会真实推进 SGLang 的 OSL 计数。

需要特别澄清：这个首 token 来自 Prefill 已经计算出的 logits，只需要 sampling，不需要再执行一次完整 Decode transformer forward。PD 仿真当前没有单独的 sampling Predictor，因此将 sampling 开销视为 0。

正确的时间线是：

```text
P 副本完成 Prefill，得到 logits
        ↓
logits / KV 传输到选定的 D 实例
        ↓
在 D 实例 sampling 产生 first token（当前建模耗时 0）
        ↓
第一次完整 Decode forward 生成 second token
```

### 17.2 原代码为什么仍有问题

旧 PD Controller 只在显式 Decode batch 完成时执行：

```python
req.decode_step_count += 1
```

Hook 又对最终 Extend 中的 token 直接跳过。因此 OSL=3 时会出现：

```text
SGLang:
Prefill sampling = token 1
Decode forward 1 = token 2
Decode forward 2 = token 3 → FINISHED

旧 PD:
Prefill sampling = 未计数
Decode forward 1 = count 1
Decode forward 2 = count 2 → 尚未达到 OSL=3
```

结果是 SGLang 已经完成，而 PD state 永远少 1 个输出 token。OSL=1 时甚至不会再出现显式 Decode batch，PD state 会直接残留。

### 17.3 一度采用但已撤回的错误修复

复查过程中一度把首 token 当成完整 Decode forward：

```python
schedule_first_decode_after_prefill(...)
backend.try_admit_decode_batch(...)
backend.on_decode_step_done_batch(...)
```

这会错误地：

- 调用一次 Decode Predictor；
- 占用 D 副本完整 Decode 容量；
- 推进 D replica busy clock；
- 给 TTFT 增加一次完整 Decode 时延；
- 把 `current_past_kv_length` 增加 1。

该处理与你指出的 logits/sampling 语义不符，现已从代码和测试中完整撤回。

### 17.4 最终修复：只计输出 token，不运行 Decode forward

Controller 新增首 token sampling 事件：

```python
def on_prefill_token_sampled(self, req, now):
    self._require_phase(
        req,
        RequestPhase.KV_TRANSIT,
        "prefill token sampling",
    )

    if req.output_length > 0:
        req.decode_step_count += 1

    if req.decode_step_count >= req.output_length:
        req.phase = RequestPhase.FINISHED
        req.decode_end_time = now
```

历史字段名 `decode_step_count` 容易产生误解。它在 OSL 判断和 abort 输出统计中的真实语义是“已经生成的输出 token 数”。本次没有大范围改名以避免破坏接口，但在代码注释中已经明确。

### 17.5 Sampling 时刻如何记录

新增 `record_prefill_sampled_tokens()`：

```python
def record_prefill_sampled_tokens(backend, states):
    # 选定接收 logits/KV 的 D 实例；不占 Decode running capacity。
    backend.bind_decode_replicas(states)
    token_times = {}

    for state in states:
        token_time = state.kv_ready_time
        backend.on_prefill_token_sampled(
            state,
            token_time,
        )
        token_times[state.rid] = token_time

    return token_times
```

当前将 token 完成时刻记为 `kv_ready_time`，含义是 Prefill logits/KV 到达选定的 D 实例后立即完成 sampling。由于 sampling latency 没有独立 Predictor，所以不额外增加时延。

该函数明确不会调用：

```text
try_admit_decode_batch()
predict_decode_seconds()
admit_decode_single_replica()
admit_decode_for_replica()
on_decode_step_done_batch()
```

因此首 token：

- 不占用完整 Decode forward 容量；
- 不推进 D replica busy clock；
- 不增加 `current_past_kv_length`；
- 只增加输出 token/OSL 计数。

### 17.6 为什么 KV 长度不能在 sampling 时增加

Prefill 完成后，KV cache 中已经物化的是 prompt token：

```text
current KV length = ISL
```

sampling 只是从 logits 选择 first token，并没有对 first token 再跑 transformer，所以 first token 自身的 KV 尚未产生。

第一次显式 Decode forward 以 first token 为输入，完成后才同时：

```text
生成 second token
物化 first token 的 KV
current_past_kv_length: ISL → ISL + 1
输出 token 计数: 1 → 2
```

所以 sampling 事件只增加 `decode_step_count`，不增加 `current_past_kv_length`。

### 17.7 OSL=1 和 OSL=3 的最终行为

OSL=1：

```text
Prefill logits → sampling token 1
decode_step_count: 0 → 1
立即 FINISHED
Decode Predictor 调用次数 = 0
D replica busy clock 增量 = 0
```

OSL=3：

```text
Prefill sampling → count 1，KV length=ISL
Decode forward 1 → count 2，KV length=ISL+1
Decode forward 2 → count 3，KV length=ISL+2 → FINISHED
```

这样 SGLang 的 output token 数与 PD 的 OSL 计数同步，但完整 Decode forward 只执行 OSL-1 次。

### 17.8 TTFT 与完成状态清理

Extend 和 Decode 统一使用 `PD_BATCH_TOKEN_TIMES` 记录 token。对于 Prefill sampling token：

```python
PD_BATCH_TOKEN_TIMES[rid] = state.kv_ready_time
```

closed-loop TTFT 在 `decode_start_time is None` 时使用：

```text
token_time - prefill_start_time
= Prefill + logits/KV handoff + D 侧 sampling
```

不会增加完整 Decode 时延。

FINISHED state 仍在 `process_batch_result()` 完成 token 指标后清理，同时删除：

```python
PD_PREFILL_KV_SERVICE.pop(rid, None)
PD_CHUNK_ACCUM.pop(rid, None)
```

### 17.9 代码与测试位置

- [`pd_controller.py`](hisim/src/hisim/simulation/pd_controller.py)：`on_prefill_token_sampled()`。
- [`pd_runtime.py`](hisim/src/hisim/simulation/pd_runtime.py)：`record_prefill_sampled_tokens()`。
- [`pd_timeline.py`](hisim/src/hisim/simulation/pd_timeline.py)：无 Decode forward 的 closed-loop TTFT。
- [`sglang_hook.py`](hisim/src/hisim/simulation/sglang/sglang_hook.py)：最终 Prefill sampling 事件接入。
- [`test_pd_backend_a.py`](hisim/tests/unit/simulation/test_pd_backend_a.py)：OSL=1/3、KV 长度和 D clock 回归。
- [`test_pd_backend_b_skeleton.py`](hisim/tests/unit/simulation/test_pd_backend_b_skeleton.py)：Backend B 不调用 Decode worker 的对等测试。

测试名称：

```text
test_final_prefill_token_is_sampled_without_decode_forward
test_prefill_sample_plus_explicit_decode_steps_reaches_osl
test_prefill_sampling_binds_decode_instances_without_running_forward
test_backend_b_samples_final_prefill_token_without_decode_worker
test_closed_loop_prefill_sampled_token_has_no_decode_forward_cost
```

## 18. 核心数据结构变化

`PDRequestState` 当前承担三类信息：

### 18.1 生命周期

```text
phase
decode_step_count
current_past_kv_length
```

### 18.2 阶段时间戳

```text
arrival_time
prefill_queue_start_time
prefill_start_time
prefill_end_time
kv_ready_time
decode_start_time
decode_end_time
```

### 18.3 路由与 Chunked Prefill 元数据

```text
prefill_is_final_chunk
prefill_replica_idx
prefill_batch_id
```

这些字段分别用于状态机校验、阶段指标计算、服务副本亲和以及 replica-local KV handoff。

## 19. 回归测试结果

### 19.1 PD / simulation 全量测试

执行命令：

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -m pytest tests/unit/simulation -q
```

结果：

```text
283 passed, 1 skipped
```

唯一 skip 是本地没有可用 AIC 性能数据库组合时跳过的真实 AIC worker smoke test，不是实现失败。

### 19.2 扩大后的 HiSim unit tests

执行命令：

```powershell
python -m pytest tests/unit -q \
  --ignore=tests/unit/visualization/test_sweep_dashboard.py
```

结果：

```text
308 passed, 1 skipped
```

排除的可视化测试依赖 `streamlit.testing`，当前本机 Streamlit 版本不提供该模块，与 PD 修改无关。

### 19.3 静态检查

执行内容：

```powershell
python -m compileall -q \
  hisim/src/hisim/simulation \
  hisim/tests/unit/simulation

git diff --check
```

结果：

```text
Python 编译检查：通过
Git 空白符检查：通过
```

## 20. 重点回归测试与覆盖目标

| 测试 | 覆盖目标 |
|---|---|
| `test_prefill_admission_capacity_uses_all_service_replicas` | Prefill 总容量计算 |
| `test_external_termination_releases_chunked_prefill_capacity` | abort 后释放 Prefill 槽位 |
| `test_external_termination_releases_single_decode_capacity` | abort 后释放 Decode 容量 |
| `test_terminate_request_is_idempotent_from_every_live_phase` | 任意 live phase 幂等终止 |
| `test_external_termination_releases_backend_b_prefill_capacity` | Backend B 终止语义对等 |
| `test_kv_transfers_share_one_capacity_accounted_link` | 多 KV 传输共享链路 |
| `test_kv_handoff_submission_is_ordered_by_prefill_completion` | KV handoff 不受调用列表顺序影响 |
| `test_shift_pd_time_origin_aligns_all_optional_pd_timestamps` | PD 时间原点统一 |
| `test_decode_forward_advances_one_output_token_until_osl` | 完整 Decode forward 每次推进一个输出 token |
| `test_final_prefill_token_is_sampled_without_decode_forward` | 最终 Prefill sampling 计入 OSL，但不运行 Decode forward |
| `test_prefill_sample_plus_explicit_decode_steps_reaches_osl` | sampling 首 token 加后续 Decode 与 SGLang 同步达到 OSL |
| `test_prefill_sampling_binds_decode_instances_without_running_forward` | sampling 绑定 D 实例但不推进 Decode clock |
| `test_backend_b_samples_final_prefill_token_without_decode_worker` | Backend B 首 token 不调用 Decode worker |
| `test_closed_loop_prefill_sampled_token_has_no_decode_forward_cost` | TTFT 不增加一次完整 Decode 时延 |
| `test_abort_request_ids_supports_single_and_batch_messages` | 单请求和批量 abort 字段兼容 |
| `test_get_disagg_config_round_trip` | 无 AIC SDK 时配置解析 |

## 21. 修改文件清单

### 21.1 运行时代码

| 文件 | 主要修改 |
|---|---|
| `simulation/manager/config.py` | AIC 可选依赖延迟导入、Python 3.8 注解兼容 |
| `simulation/pd_backend_a.py` | Prefill/Decode 容量、终止清理、batch/replica 路由 |
| `simulation/pd_backend_b.py` | Backend B 对等实现 |
| `simulation/pd_backend_protocol.py` | 增加统一终止协议 |
| `simulation/pd_config.py` | Prefill admission capacity |
| `simulation/pd_controller.py` | 严格状态机、共享 KV 链路、幂等终止 |
| `simulation/pd_metrics.py` | 阶段指标和 PD 时间原点统一 |
| `simulation/pd_runtime.py` | replica-local handoff、共享链路提交顺序、最终 Prefill sampling 记账 |
| `simulation/pd_sglang_lifecycle.py` | 单请求和批量 abort ID 提取 |
| `simulation/pd_timeline.py` | Prefill queue 基准和 Chunked Prefill 文档 |
| `simulation/sglang/sglang_hook.py` | 容量对齐、角色时钟、首 token/OSL 进度、完成状态与 abort 清理 |

### 21.2 测试与文档

| 文件 | 主要修改 |
|---|---|
| `test_pd_backend_a.py` | Backend A 容量、终止、KV 顺序和 Prefill 首 token 测试 |
| `test_pd_backend_b_skeleton.py` | Backend B 终止释放测试 |
| `test_pd_backend_protocol.py` | Backend 协议一致性 |
| `test_pd_config.py` | Prefill 总容量测试 |
| `test_pd_controller.py` | 状态终止和共享链路测试 |
| `test_pd_metrics.py` | PD 时间原点测试 |
| `test_pd_sglang_lifecycle.py` | SGLang abort message 兼容测试 |
| `test_pd_backend_a_smoke.py` | 共享带宽下的 KV P95 不变量 |
| `pd_disagg_modeling_review.md` | 历史问题状态和当前边界更新 |

## 22. 当前架构边界

以下内容是当前模拟器的明确架构范围，不属于本轮仍未修复的逻辑缺陷：

1. SGLang 仍使用一个原生 Scheduler loop 提供 batch composition；PD Backend 在该 batch 基础上进行角色定价和副本拆分。
2. Backend B 使用独立 Predictor worker，不等价于真实部署两套完整 SGLang Scheduler 服务。
3. 共享 KV 链路当前使用容量受限的串行虚拟时钟模型，不模拟更底层的传输协议分片、拥塞窗口或 NIC 包级行为。

这些边界已经在 [`pd_disagg_modeling_review.md`](pd_disagg_modeling_review.md) 中明确记录。它们不会破坏本文修复的状态机、容量、时间线和指标不变量，但在使用模拟结果推导真实集群绝对性能时需要保留这一前提。

## 23. 最终结论

本轮修改已经把 PD 分离从“部分状态记录和角色定价”完善为一套具备以下能力的虚拟时间执行模型：

- 严格且可验证的请求状态机；
- Prefill/Decode 容量约束；
- Chunked Prefill 跨调用亲和性；
- replica-local 真实 batch；
- 真实 Prefill batch shape 预测；
- 正确的 Prefill 排队指标；
- replica-local KV handoff；
- 共享 KV 链路带宽竞争；
- 最终 Prefill sampling 首 token 不运行完整 Decode forward，并与后续 Decode 同步达到标准 OSL；
- 独立的外部 abort 全资源清理；
- Backend A/Backend B 协议一致性；
- 可选 AIC 依赖隔离；
- 完整的回归测试和静态检查。

在当前模拟设计范围和已执行测试范围内，没有发现剩余的 PD 实现缺陷。
