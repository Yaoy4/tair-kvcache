# Hisim PD 分离（single_process / BackendA）准入控制四项 Bug 根因分析与修复验证测试报告

- 测试日期：2026-07-23（全量 24 格矩阵：服务端 15:14:52 启动 ～ 15:37:xx 全部跑完，全程未重启、未崩溃、未卡死）
- 测试目标：在 `260722-Tianhao-PD分离Retract修复验证测试报告.md` 已验证的 Retract 修复基础上，针对本轮新一轮全量矩阵复测中**新发现的 3 个问题（+1 个既有问题的复核）**逐一根因定位、修复，并按 `260721-Tianhao-实验测试统一参数与指令.md` 规定的参数矩阵重新跑一遍完整回归。
- 测试范围：**仅 PD 分离（disagg.enabled=true, backend="single_process"，对应 BackendA），拓扑为 1P2D（1 个 Prefill 副本 + 2 个 Decode 副本）**，与既有的 `260722-Tianhao-PD分离P-D拓扑扩容性能测试报告.md` 使用同一拓扑，便于直接对比修复前后的指标变化。按用户要求未测试 PD 合并（TP=2/DP=2）场景。
- 结论先行：**本轮共定位并修复 4 个问题（KV 预算未按副本数扩容、原生请求数上限用 min 而非 sum、单副本 Decode 缺少 KV 显存感知导致的崩溃、分块预填 KV 成本计算错误导致的死锁）。修复后单元测试 358 passed / 1 skipped（较修复前净增 2 个回归测试），全量 24 格参数矩阵（3 rr × 4 il × 2 ol，每格 200 请求，共 4800 请求）全部 PASS，零崩溃、零卡死、零负延迟。**

---

## 1. 背景：本轮问题是如何被发现的

在 `260722` 的 Retract 修复验证通过之后，用户要求对同一 1P2D 拓扑做更彻底的压力复测（含更高并发、更长上下文的极端格子）。复测过程中：

1. 首先发现 **`il=8192, ol=4096`（rr=64）附近的格子偶发崩溃**，服务端报出 `AIConfigurator predictor returned a negative decode latency`（负延迟被当作字面时间差消费，破坏了模拟时钟），根因追查后牵出了 3 个互相独立的问题（第 3.1～3.3 节）。
2. 在确认上述修复后重跑全量矩阵时，**`rr=1, il=16384, ol=1024` 格子（200 请求）在第 183/200 个请求处卡死**，服务端 `#running-req` 长期冻结在 94、`#queue-req` 冻结在 17，3 分钟内无任何日志/CPU 活动 —— 这是一个新的、独立的死锁问题（第 3.4 节）。

以下按发现顺序逐一说明每个问题的现象、根因、修复方式。

---

## 2. 测试环境

| 项目 | 内容 |
|---|---|
| 仓库/分支 | `tair-kvcache`，分支 `Thjiang_Dev`，基线 commit `bdb1514`（"v2.2 fix retract problem..."），工作区含本轮未提交改动（见第 5 节） |
| Python 环境 | `/mnt/nfs02/users/tjiang/Gitrepo/tair-kvcache/hisim/hisim`（单元测试）；`/mnt/nfs02/users/tjiang/pd_verify_e0735cc/.venv`（矩阵冒烟测试，独立于仓库根目录 `.venv`） |
| hisim 安装方式 | editable install，确认运行的是工作区中**未提交**的修复代码 |
| aiconfigurator | 本地 clone，commit `e0735ccca08c790b37c511a20f484a62a4127118`（与部署指南要求一致） |
| 模型 | `Qwen/Qwen3-8B`（仅缓存 config/tokenizer，Hisim 只做性能模拟不做真实推理） |
| 运行模式 | `SGLANG_USE_CPU_ENGINE=1`，`device=cpu`，GPU 未使用 |
| 拓扑 | `topo_1P2D.json`：Prefill 1 副本 + Decode 2 副本，`decode_queue_mode=per_replica_queue`，`tp_size=1`（与 260722 P-D 拓扑报告的 1P2D 配置一致） |

---

## 3. 本轮定位并修复的 4 个问题

### 3.1 问题一：KV Cache 池容量未按 PD 副本数扩容

- **现象**：多副本拓扑（如 1P2D，共 3 个声明的设备）下，Prefill 侧准入会被 Decode 侧的 KV 占用异常提前"饿死"，尤其在副本数越多的拓扑下越明显。
- **根因**：BackendA（`single_process`）把所有声明的 Prefill/Decode 副本都路由到**同一个**真实（mock）SGLang engine 进程里。而 `estimate_kv_cache_pool_capacity()` 只按**单张卡**的显存预算估算 KV cache 池容量（`max_total_num_tokens`），从未乘以拓扑里声明的副本总数——相当于 3 个副本共用的池子却只按 1 个副本的容量分配，容量被低估到 1/3。
- **修复**：
  - 新增 `DisaggConfig.total_replica_count()`（`pd_config.py`）：返回 `prefill.replicas + decode.replicas`（disagg 未启用时返回 1，聚合单引擎路径不受影响）。
  - `sglang_hook.py::C_ModelRunnerHook` 中，`self.max_total_num_tokens = estimate_kv_cache_pool_capacity(model, hw, config) * ConfigManager.get_disagg_config().total_replica_count()`。

### 3.2 问题二：原生 `max_running_requests` 用 `min()` 取两角色较小值，导致相互饿死

- **现象**：原生 SGLang 调度器只有**一个**统一的 `max_running_requests` 池子，不区分"这是 Prefill 请求"还是"这是 Decode 请求"。
- **根因**：原代码用 `min(prefill_admission_capacity(), decode_admission_capacity())` 设置这个统一上限——如果两个角色的声明容量不同（如 1P2D 下 Prefill 只有 1 副本、Decode 有 2 副本，两者的 `max_running_per_replica × replicas` 并不相等），取较小值会让声明容量更大的角色也被限制在较小值上，无法发挥应有的并发能力，且长时间占用 slot 的 Decode 请求会挤占 Prefill 的准入名额。
- **修复**：
  - 新增 `DisaggConfig.combined_running_request_capacity()`：返回 `prefill_admission_capacity() + decode_admission_capacity()`（求和而非取小），让共享池同时容纳两个角色各自声明的并发量，与真实独立 P/D 硬件的行为一致。
  - 配合新增的 `get_num_allocatable_reqs` 钩子（见 3.3 节机制复用）按 `available_admission_budget()` 动态收紧每次原生调度器实际能新拉取的请求数，让"求和"的宽松上限不会绕开 hisim 自己更细的 P/D 子容量校验。

### 3.3 问题三：单个 Decode 副本缺少真实 KV 显存感知，长上下文下触发负延迟崩溃

- **现象**：`rr=64, il=8192, ol=4096` 附近的格子偶发服务端崩溃，日志报 `RuntimeError: AIConfigurator predictor returned a negative decode latency (OOM sentinel)`；此前（更早版本）甚至观察到大量请求的 TPOT/E2E 变成负数。
- **根因**：`max_running_per_replica` 只是一个**请求数量**上限，完全不知道每个请求真实占用多少 KV token。当多个长上下文请求同时被分到同一个 Decode 副本、请求数虽未超上限但 KV 显存已经超出该副本单卡真实容量时，aiconfigurator 预测器会用**负数**作为"OOM 哨兵值"返回。旧代码把这个负数当作字面时间差直接喂给模拟时钟，导致该副本后续所有请求的延迟被永久污染成负值。
- **修复**（分两部分）：
  1. **快速止损**（`pd_aic_adapter.py`）：预测器返回负值时不再当作时间差消费，而是立即 `raise RuntimeError` 抛出详细诊断信息，防止负数静默污染模拟时钟。
  2. **根治**（`pd_factory.py` / `pd_backend_a.py` / `pd_controller.py`）：引入 `decode_kv_capacity_per_replica`（单个 Decode 副本的真实 tp/pp 感知 KV-token 预算，`build_disagg` 里用与该角色预测器完全一致的 hw/tp/pp 输入计算得到），并在 `admit_decode_for_replica` 里新增 `token_budget`/`token_cost` 门控（`PDController.admit_decode_targeted` 新增对应形参）：只有当某副本"已占用 KV + 本请求 KV 成本"仍不超过该副本真实预算时才准入，超预算的请求留在等待队列，下一轮再试，而不是让请求硬挤进一个会 OOM 的批次。
  - **配套的崩溃回归修复（3b）**：上述 KV 门控生效后，会出现"同一批原生 decode batch 里所有请求都被 KV 门控合法拒绝"的新状态——这在门控加入之前是不可能出现的，此前的代码把"批次非空但 PD 一个都不认可"一律当作状态机损坏，直接 `raise RuntimeError`。本次在 `sglang_hook.py::wrapped_run_batch` 里补充了一个三分支判断：区分"合法背压（KV 门控拒绝，记 0 延迟，下一轮重试）" vs. "真正的状态追踪损坏（rid 完全不在 PD_REQUEST_STATES 里，仍然报错）"，避免把新引入的合法背压误判为崩溃。

### 3.4 问题四（本轮最新发现）：分块预填（chunked prefill）KV 成本计算错误导致永久死锁

- **现象**：全量矩阵复测中，`rr=1, il=16384, ol=1024`（200 请求）稳定卡在第 183/200 个请求，服务端 `#running-req` 冻结在 94、`#queue-req` 冻结在 17，3 分钟内日志和 CPU 活动完全静止（非缓慢，是真死锁）。
- **排查过程**（三轮插桩定位）：
  1. 第一轮插桩聚合准入函数 `available_admission_budget()`：发现 KV 容量还有 66% 空闲（`total_kv_committed=358939` vs `total_kv_capacity=1086938`），说明聚合口径**不是**瓶颈，推翻了"KV 追踪字典泄漏"的最初猜测。
  2. 第二轮插桩交叉核对所有"打开中"的 rid 与其真实阶段：确认 128 个"打开中"请求（62 running_decode + 66 waiting_decode）全部合法，**没有**状态泄漏。
  3. 第三轮插桩具体到 `admit_decode_for_replica` 逐副本诊断：发现两个 Decode 副本的 `capacity`（数量上限）都还有大量空余，但 `token_budget`（KV 预算）几乎耗尽（4057/4512），**每次准入尝试全部被拒绝**——真正的单副本 KV 门控（3.3 节修复）是正确饱和的，但**聚合门控（3.1/3.2 节的 `available_admission_budget()`）却认为还有大把空间**，才是问题所在。
- **根因**：`_decode_token_cost(req) = req.input_length + req.output_length` 用来估算每个请求的 KV 成本，但 `sglang_hook.py` 为了准确预测"这一次 forward pass"的延迟，**故意**把 `input_length` 在分块预填过程中临时改写为"当前这一块 chunk 的大小"——例如一个 16384 token 的请求，在 `--chunked-prefill-size 4096` 下被切成 4 块，第一块的 `input_length` 只有 4096，直到最后一块 finalize 时才被纠正为真实的全量长度。而聚合成本快照（`_reserve_prefill_slot`）**只在第一块调用一次**（后续块被提前 return 跳过），于是把"4096+1024=5120"这个偏小近 4 倍的成本"冻结"进了聚合追踪表，而不是真实的"16384+1024=17408"。聚合门控因此严重低估真实 KV 占用，放行了远超每个 Decode 副本真实能承载的请求数进入系统；这些请求进入具体副本后又被正确核算的单副本门控（3.3 节）正确拦下——形成"进不去、也出不来"的永久积压，表现为死锁而非崩溃。
- **修复**：
  - `pd_types.py`：`PDRequestState` 新增 `total_input_length: Optional[int] = None` 字段，与 `input_length`（chunk 级、供延迟预测用）解耦，代表"稳定不变的全量 prompt 长度"，专供 KV 预算使用。
  - `pd_backend_a.py::_decode_token_cost`：优先使用 `total_input_length`（为空时回退到 `input_length`，保证所有旧调用方/旧测试行为不变）。
  - `sglang_hook.py`：在首块创建请求状态时，用 `req.origin_input_ids`（SGLang 自身在请求创建时就固定不变的全量 token 列表，不受分块影响）设置 `total_input_length=len(origin_input_ids)`；并在最终块 finalize 时同步 `total_input_length = input_length`（覆盖极少数"先被 retract 又重新预填"场景下 `fill_ids` 更精确的长度）。
  - **未修复的已知次要局限（有意不处理）**：一个请求被 retract 之后重新预填时，聚合快照沿用第一次首块预留时的 `total_input_length`，不会重新计算；仅在极端"retract 后又追加大量已生成 token"场景下可能轻微低估，属于比本 bug 小一个数量级的边缘情况，本轮压力测试全程 0 次 retract 触发，故未过度设计修复。

---

## 4. 单元测试结果

修复上述 4 个问题后，新增 2 个针对性回归测试（`test_pd_backend_a.py`）：

- `test_decode_token_cost_prefers_total_input_length_when_set`：直接验证 `_decode_token_cost` 优先使用 `total_input_length` 的语义。
- `test_reserve_prefill_slot_uses_total_input_length_not_chunk_length`：通过真实的 `bind_prefill_replicas()` 公共入口端到端复现——chunk 大小的 `input_length`（4096）配合全量的 `total_input_length`（16384）时，聚合追踪表正确记入全量成本（17408），而非仅按 chunk 计算的成本。

```
SGLANG_USE_CPU_ENGINE=1 hisim/hisim/bin/python -m pytest hisim/tests/unit/ \
  --ignore=hisim/tests/unit/visualization/test_sweep_dashboard.py -q

358 passed, 1 skipped, 1 warning in 19.09s
```

（修复前基线为 356 passed，净增 2 个回归测试，零回归）

---

## 5. 待提交的代码改动（Git，工作区未提交）

```
 M hisim/src/hisim/simulation/pd_aic_adapter.py       |  44 +-   （3.3 节：负延迟哨兵值 fail-loud）
 M hisim/src/hisim/simulation/pd_backend_a.py         | 293 ++-  （3.1/3.3/3.4 节核心逻辑）
 M hisim/src/hisim/simulation/pd_config.py            |  40 +   （3.1/3.2 节：total_replica_count / combined_running_request_capacity）
 M hisim/src/hisim/simulation/pd_controller.py        |  28 +-   （3.3 节：admit_decode_targeted 的 token_budget 门控）
 M hisim/src/hisim/simulation/pd_factory.py           |  47 +-   （3.3 节：decode_kv_capacity_per_replica 的计算与装配）
 M hisim/src/hisim/simulation/pd_types.py             |  13 +   （3.4 节：total_input_length 字段）
 M hisim/src/hisim/simulation/sglang/sglang_hook.py   | 117 +-   （3.1/3.2/3.3b/3.4 节：hook 侧接线）
 M hisim/tests/unit/simulation/test_config_manager_disagg.py
 M hisim/tests/unit/simulation/test_pd_backend_a.py   | 593 +++   （新增/更新回归测试，含 3.4 节新增的 2 个）
 M hisim/tests/unit/simulation/test_pd_config.py      | 110 +
 M hisim/tests/unit/simulation/test_pd_controller.py  |  79 +
 M hisim/tests/unit/simulation/test_pd_role_clock_integration.py | 107 +-
```

**是否提交及提交信息由用户决定，本次测试未擅自提交。**

---

## 6. 测试配置（1P2D 拓扑，与 260722 P-D 拓扑报告一致）

```json
{
  "disagg": {
    "enabled": true,
    "backend": "single_process",
    "decode_queue_mode": "per_replica_queue",
    "prefill": { "replicas": 1, "max_running_per_replica": 64, "tp_size": 1, ... },
    "decode":  { "replicas": 2, "max_running_per_replica": 64, "tp_size": 1, ... },
    "kv_transfer": { "bw_gbps": 128, "latency_us": 20 }
  },
  "scheduler": { "tp_size": 1, "backend_name": "sglang", "backend_version": "0.5.10" }
}
```

服务端启动（同一进程连续跑完全部 24 组，符合 260721 文档"服务端常驻、客户端逐组切换参数"设计）：

```bash
python3 -m hisim.simulation.sglang.launch_server \
  --model-path Qwen/Qwen3-8B --host 127.0.0.1 --port 30021 \
  --sim-config-path topo_1P2D.json \
  --chunked-prefill-size 4096 --skip-server-warmup --disable-radix-cache
```

客户端矩阵（严格按 260721 文档，3 rr × 4 il × 2 ol = 24 组，每组 200 请求，`seed=1`）：

```bash
for rr in 1 8 64; do
  for il in 1024 4096 8192 16384; do
    for ol in 1024 4096; do
      python3 -m hisim.simulation.bench_serving \
        --bench-mode simulation --backend sglang \
        --host 127.0.0.1 --port 30021 --model Qwen/Qwen3-8B \
        --dataset-name random --num-prompts 200 --warmup-requests 0 \
        --request-rate "$rr" --random-input-len "$il" --random-output-len "$ol" \
        --random-range-ratio 1 --seed 1 --flush-cache \
        --output-file "metrics_rr${rr}_il${il}_ol${ol}.jsonl"
    done
  done
done
```

---

## 7. 测试结果总表（24/24 全部 PASS）

| RR | IL | OL | 完成数 | TTFT (ms) | TPOT (ms/token) | E2E (ms) | 输出吞吐 (token/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,024 | 1,024 | 200/200 | 64.31 | 13.54 | 13,915.91 | 994.67 |
| 1 | 1,024 | 4,096 | 200/200 | 65.47 | 28.20 | 115,562.54 | 2,708.23 |
| 1 | 4,096 | 1,024 | 200/200 | 251.40 | 21.38 | 22,122.28 | 971.04 |
| 1 | 4,096 | 4,096 | 200/200 | 27,035.10 | 51.16 | 236,523.90 | 1,810.92 |
| 1 | 8,192 | 1,024 | 200/200 | 605.91 | 60.48 | 62,473.32 | 885.38 |
| 1 | 8,192 | 4,096 | 200/200 | 127,622.36 | 63.80 | 388,867.82 | 1,154.05 |
| **1** | **16,384** | **1,024** | **200/200** | 76,927.05 | 98.46 | 177,650.12 | 473.85 |
| 1 | 16,384 | 4,096 | 200/200 | 384,017.52 | 71.36 | 676,246.35 | 666.88 |
| 8 | 1,024 | 1,024 | 200/200 | 4,556.21 | 25.68 | 30,821.91 | 3,514.03 |
| 8 | 1,024 | 4,096 | 200/200 | 45,961.57 | 32.08 | 177,321.84 | 3,149.50 |
| 8 | 4,096 | 1,024 | 200/200 | 26,332.82 | 53.39 | 80,937.16 | 1,370.03 |
| 8 | 4,096 | 4,096 | 200/200 | 102,331.29 | 56.92 | 335,400.94 | 1,419.36 |
| 8 | 8,192 | 1,024 | 200/200 | 64,993.14 | 80.37 | 147,184.96 | 798.80 |
| 8 | 8,192 | 4,096 | 200/200 | 295,211.45 | 62.12 | 549,580.64 | 802.72 |
| 8 | 16,384 | 1,024 | 200/200 | 209,002.20 | 82.64 | 293,531.86 | 362.50 |
| 8 | 16,384 | 4,096 | 200/200 | 742,471.82 | 66.58 | 1,015,090.26 | 414.82 |
| 64 | 1,024 | 1,024 | 200/200 | 11,843.58 | 24.18 | 36,581.06 | 3,810.01 |
| 64 | 1,024 | 4,096 | 200/200 | 55,083.90 | 31.71 | 184,935.15 | 3,190.22 |
| 64 | 4,096 | 1,024 | 200/200 | 36,745.28 | 53.39 | 91,349.61 | 1,370.03 |
| 64 | 4,096 | 4,096 | 200/200 | 112,743.74 | 56.92 | 345,813.39 | 1,419.36 |
| 64 | 8,192 | 1,024 | 200/200 | 75,405.59 | 80.37 | 157,597.42 | 798.80 |
| 64 | 8,192 | 4,096 | 200/200 | 305,623.90 | 62.12 | 559,993.09 | 802.72 |
| 64 | 16,384 | 1,024 | 200/200 | 219,414.65 | 82.64 | 303,944.31 | 362.50 |
| 64 | 16,384 | 4,096 | 200/200 | 752,884.27 | 66.58 | 1,025,502.71 | 414.82 |

> 加粗标注：`rr=1, il=16384, ol=1024` 是本轮死锁问题（3.4 节）的原始复现格子——修复前稳定卡死在 183/200，本次 200/200 顺利完成。

**全部 24 格：完成数均为 200/200，所有 TTFT/TPOT/E2E 均为正值，服务端全程无崩溃、无异常堆栈、无卡死。**

---

## 8. 与历史报告（260722，同一 1P2D 拓扑）对比

同一拓扑、同一参数矩阵，对比 `260722-Tianhao-PD分离P-D拓扑扩容性能测试报告.md` 中 1P2D 一节的旧数据（当时代码尚未包含本轮 3.3/3.4 节的修复），观察到：

- **除 2 个几乎无竞争的格子（`rr=1,il=1024,ol=1024` 与 `rr=1,il=4096,ol=1024`）外，其余 22 格 TTFT/E2E 普遍显著下降**（多数下降 20%～80%），且**没有一格变差**。
- 下降的主要原因是：3.3 节新增的单副本 KV 门控 + 3.2 节的"求和而非取小"改变了请求被**真正**准入系统的时机——修复前，过于宽松的聚合上限会让请求提前被 hisim 记为"已准入"（计时开始），但实际上要在内部排队等待 KV 释放，这段等待被计入了 TTFT；修复后，原生调度器自身的 `waiting_queue` 提前拦下这些请求，只有真正有 KV 空间时才计入 PD 的准入时钟，因此记录到的延迟更接近真实系统行为，是**修正后更准确的读数，而非性能变差**。
- 变化幅度最大的仍是原本就依赖分块预填的长输入格子（`il=8192/16384`），与 3.4 节根因分析中"分块预填成本低估"直接相关，这也正好回应了本轮调查最初的触发问题——"为什么 Output Length 增大、TTFT 会异常暴涨"：`rr=1,il=16384,ol=4096` 的 TTFT 从 812,216.63ms 降到 384,017.52ms，之前的异常暴涨相当一部分正是本报告 3.4 节 bug 的直接体现。
- 需要说明：本对比是**多个修复（3.1～3.4 全部叠加）的累计效果**，未逐一做消融隔离，因此不能把某一格具体的下降百分比精确归因到某一个单独的 bug；但方向和量级与根因分析（聚合门控过度放行 → 记录到的排队等待被拉长）完全吻合。

---

## 9. 结论

1. **本轮共发现并修复 4 个 PD 分离（single_process/BackendA）准入控制相关问题**：KV 池容量未按副本数扩容（3.1）、原生请求数上限用 min 导致角色互相饿死（3.2）、单副本 Decode 缺少 KV 显存感知导致的负延迟崩溃（3.3，含崩溃回归子问题 3.3b）、分块预填 KV 成本计算错误导致的永久死锁（3.4，本轮最新发现）。每项均已完成根因定位、代码修复、单元测试回归（358 passed / 1 skipped）与独立复现脚本验证。
2. **按 260721 文档规定的全部 24 组参数矩阵（1P2D 拓扑）重新测试，服务端全程零崩溃、零卡死、零负延迟，完整跑完 4800 个请求。**
3. 本轮最初触发死锁的复现格子（`rr=1, il=16384, ol=1024`）修复前稳定卡在 183/200，本次 200/200 顺利完成，验证有效。
4. 与同拓扑的历史数据对比显示，多数格子的 TTFT/E2E 读数在修复后显著降低，这是聚合准入口径被修正、请求"真正开始计时"的时机更准确导致的**结果修正**，而非性能回退；该现象也解释了本轮调查最初的触发疑问（长输出长度下 TTFT 异常暴涨）。
5. 修复代码目前仍为**工作区未提交状态**（见第 5 节文件清单），是否提交及提交信息由用户决定，本次测试未擅自提交。
6. 本次测试未覆盖 PD 合并（TP=2/DP=2）场景，按用户明确要求跳过。

---

## 附：测试产出文件位置

- 全量矩阵驱动脚本：`/mnt/nfs02/users/tjiang/pd_fix_validation/run_full_matrix_crash_test.sh`
- 全量矩阵原始指标：`/mnt/nfs02/users/tjiang/pd_fix_validation/cases_topology/1P2D_full_matrix_both_fixes/metrics_*.jsonl`（24 个文件）
- 服务端/客户端日志：`/mnt/nfs02/users/tjiang/pd_fix_validation/server_logs/{server,client}_1P2D_full_matrix_both_fixes.log`
- 3.4 节死锁问题的独立 200 请求复现脚本（修复前 183/200 卡死，修复后 200/200）：`/mnt/nfs02/users/tjiang/pd_fix_validation/hang_repro/run_hang_repro.sh`
