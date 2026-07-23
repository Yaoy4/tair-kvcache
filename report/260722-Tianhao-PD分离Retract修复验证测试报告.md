# Hisim PD 分离（single_process / BackendA）Retract 修复验证测试报告

- 测试日期：2026-07-22（服务端进程存活时段 01:13:08 ~ 01:42:11，持续约 29 分 3 秒，全程未重启）
- 测试目标：验证 `0721-Retract错误.md` 中记录的 PD Backend A 崩溃问题（`ValueError: cannot schedule prefill for rid=... in phase=running_decode`）在当前（尚未提交的）修复代码下，按照 `260721-实验测试统一参数与指令-蒋天颢.md` 规定的**真实 server-client 参数矩阵**跑一遍是否还会复现。
- 测试范围：**仅 PD 分离（disagg.enabled=true, backend="single_process"，对应 BackendA）**，按用户要求未测试 PD 合并（TP=2/DP=2）场景。
- 结论先行：**24/24 组参数全部通过，服务端全程无崩溃、无异常堆栈，期间共发生 140 次 SGLang 侧真实 KV-cache-pool-full retract 事件，全部被新的 `reset_for_retract` 逻辑正确吸收，未再触发 `ValueError`。修复有效。**

---

## 1. 测试环境

| 项目 | 内容 |
|---|---|
| 仓库/分支 | `tair-kvcache`，分支 `Thjiang_Dev`（工作区含未提交改动，见第 6 节） |
| Python 环境 | `/mnt/nfs02/users/tjiang/Gitrepo/tair-kvcache/hisim/hisim`（venv，**不是**仓库根目录的 `.venv`） |
| hisim 安装方式 | editable install，`hisim.__file__` 直接指向 `.../tair-kvcache/hisim/src/...`，确认运行的是工作区中**未提交**的修复代码 |
| 关键依赖版本 | `sglang==0.5.6.post2`、`torch==2.9.1`、`aiconfigurator==0.5.0`（pip 包，仅作为占位，真正生效的是下面本地 clone） |
| aiconfigurator 数据源 | 本地 clone `/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator`，commit `e0735ccca08c790b37c511a20f484a62a4127118`（与部署指南要求的 commit 一致） |
| 模型 | `Qwen/Qwen3-8B`（仅缓存了 config/tokenizer，约 6.9MB，无需权重文件，因为 Hisim 只做性能模拟不做真实推理） |
| 运行模式 | `SGLANG_USE_CPU_ENGINE=1`，`device=cpu`；机器上的 2×RTX PRO 6000 GPU 未使用（模拟器不需要 GPU 算力） |
| 网络 | `HF_ENDPOINT=https://hf-mirror.com`；发现系统级 `http_proxy/https_proxy=http://127.0.0.1:29831` 会拦截对 `127.0.0.1:30003` 的本地请求（见第 5 节问题记录），已通过 `no_proxy=127.0.0.1,localhost` 规避 |

---

## 2. 测试配置文件（严格照抄 260721 文档的 PD 分离 JSON，仅修正 `database_path`）

保存路径：会话工作区 `hisim_pd_test_260721/pd_disagg_260721.json`（未修改仓库内 `hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json`，因为该文件 `database_path` 指向他人目录且字段不全，为避免误改仓库文件，另建此文件）。

```json
{
  "platform": {
    "accelerator": { "name": "rtx_pro_6000_server" },
    "disk_read_bandwidth_gb": 4,
    "disk_write_bandwidth_gb": 4,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64
  },
  "predictor": {
    "name": "aiconfigurator",
    "database_path": "/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator/src/aiconfigurator/systems",
    "device_name": "rtx_pro_6000_server"
  },
  "scheduler": {
    "tp_size": 1, "ep_size": 1, "dp_size": 1,
    "data_type": "FP16", "kv_cache_data_type": "FP16",
    "backend_name": "sglang", "backend_version": "0.5.10"
  },
  "disagg": {
    "enabled": true,
    "backend": "single_process",
    "decode_queue_mode": "single_replica",
    "prefill": {
      "device_name": "rtx_pro_6000_server",
      "tp_size": 1, "ep_size": 1, "dp_size": 1, "pp_size": 1,
      "replicas": 1, "max_running_per_replica": 64,
      "data_type": "FP16", "kv_cache_data_type": "FP16",
      "prefill_scale_factor": 1.0, "decode_scale_factor": 1.0,
      "database_path": "/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator/src/aiconfigurator/systems",
      "backend_version": "0.5.10"
    },
    "decode": {
      "device_name": "rtx_pro_6000_server",
      "tp_size": 1, "ep_size": 1, "dp_size": 1, "pp_size": 1,
      "replicas": 1, "max_running_per_replica": 64,
      "data_type": "FP16", "kv_cache_data_type": "FP16",
      "prefill_scale_factor": 1.0, "decode_scale_factor": 1.0,
      "database_path": "/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator/src/aiconfigurator/systems",
      "backend_version": "0.5.10"
    },
    "kv_transfer": { "bw_gbps": 128, "latency_us": 20 }
  }
}
```

`"backend": "single_process"` 已在源码 `pd_runtime.py`/`sglang_hook.py` 中确认映射到 **BackendA**（`"two_process"` 才是 BackendB），因此本次测试确实覆盖了含 Retract 修复的目标组件。

---

## 3. 服务端启动命令（与 260721 文档一致）

```bash
export PYTHONPATH="/mnt/nfs02/users/tjiang/Gitrepo/tair-kvcache/hisim/src:/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator/src"
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HISIM_OUTPUT_DIR=".../hisim_pd_test_260721/hisim_output"
export MODEL_PATH="Qwen/Qwen3-8B"
export HF_ENDPOINT=https://hf-mirror.com

hisim/hisim/bin/python3 -m hisim.simulation.sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 30003 \
  --sim-config-path .../pd_disagg_260721.json \
  --chunked-prefill-size 4096 \
  --skip-server-warmup
```

服务端日志确认：`PD disaggregation enabled (backend=single_process, prefill_replicas=1, decode_replicas=1)`，`chunked_prefill_size=4096`，`max_prefill_tokens=16384`，`max_running_requests=64`（对应各 replica 的 `max_running_per_replica`）。**全程只启动一次服务端进程（PID 1650681），24 组参数全部对着同一个存活的服务端跑**，符合 260721 文档"服务端常驻、客户端逐组切换参数"的设计。

## 4. 客户端测试矩阵（严格按 260721 文档循环）

```bash
for rr in 1 8 64; do
  for il in 1024 4096 8192 16384; do
    for ol in 1024 4096; do
      python3 -m hisim.simulation.bench_serving \
        --bench-mode simulation --backend sglang \
        --host 127.0.0.1 --port 30003 --model Qwen/Qwen3-8B \
        --dataset-name random --num-prompts 200 --warmup-requests 0 \
        --request-rate "$rr" \
        --random-input-len "$il" --random-output-len "$ol" \
        --random-range-ratio 1 --seed 1 --flush-cache \
        --output-file ".../metrics_rr_${rr}_il_${il}_ol_${ol}.jsonl"
    done
  done
done
```

共 3 × 4 × 2 = **24 组**参数组合，每组 `--num-prompts 200`，`--seed 1` 固定（与文档一致，保证可复现）。**未测试**文档中的 PD 合并（TP=2/DP=2）配置，按用户要求跳过。

---

## 5. 测试中发现并处理的问题：`/flush_cache` 403

- 现象：`curl -X POST http://127.0.0.1:30003/flush_cache` 返回 HTTP 403，响应体是企业代理返回的 "IE friendly error message" 占位页面，并非 SGLang 自身的错误。
- 根因：容器/主机环境全局设置了 `http_proxy=https_proxy=http://127.0.0.1:29831`，而 `no_proxy` 未包含 `127.0.0.1/localhost`，导致对本机 `127.0.0.1:30003` 的请求被错误地转发到代理，代理对回环地址请求返回 403。
- 影响面：仅影响使用 `requests` 库同步调用的少数端点（如 `flush_cache`），主 benchmark 的批量请求（`aiohttp`）未受影响（冒烟测试期间 5/5 请求全部成功），但若不修复，**`--flush-cache` 会静默失败**，每组用例之间的 KV cache 状态无法真正复位，可能影响结果可比性。
- 修复：在运行测试矩阵前设置 `export no_proxy="127.0.0.1,localhost,$no_proxy"` 及大写版本 `NO_PROXY`。修复后复测确认 `flush_cache response: Cache flushed.`，问题解决，**正式矩阵测试全程 `--flush-cache` 均生效**。

---

## 6. 待验证的修复代码状态（Git）

以下文件相对上一次提交（`597a0df` "v2.1 fix 3 bugs about PD Disagg"）仍处于**未提交**状态，本次测试运行的正是这些工作区改动：

```
 M hisim/src/hisim/simulation/pd_backend_a.py
 M hisim/src/hisim/simulation/pd_backend_b.py
 M hisim/src/hisim/simulation/pd_backend_protocol.py
 M hisim/src/hisim/simulation/pd_controller.py
 M hisim/src/hisim/simulation/pd_sglang_lifecycle.py
 M hisim/src/hisim/simulation/sglang/sglang_hook.py
 M hisim/tests/unit/simulation/test_pd_backend_a.py
 M hisim/tests/unit/simulation/test_pd_controller.py
 M hisim/tests/unit/simulation/test_pd_sglang_lifecycle.py
```

核心修复：`PDController.reset_for_retract(req, now)`，在检测到 SGLang 真实发生 `retract_decode`（KV cache pool 满）后，把对应请求的 PD 阶段从 `RUNNING_DECODE` 安全回退到 `WAITING_PREFILL`，同时保留已生成的 token 进度，避免重新进入 prefill 时状态机校验报错。

---

## 7. 测试结果总表（24/24 全部 PASS）

`retracts` 列为该组用例期间服务端日志中实际发生的 SGLang KV-cache-pool-full retract 次数（按时间窗口精确归属，24 组之和 = 140，与日志中 retract 事件总数完全一致，无遗漏无重复计数）。

| rr | il | ol | 结果 | 完成数 | retract次数 | 请求吞吐(req/s) | 输入吞吐(tok/s) | 输出吞吐(tok/s) | 平均TTFT(ms) | 平均TPOT(ms) | 平均E2E(ms) |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | PASS | 200/200 | 0 | 0.96 | 981.9 | 982.1 | 59.8 | 15.0 | 15408 |
| 1 | 1024 | 4096 | PASS | 200/200 | 0 | 0.40 | 408.0 | 1632.3 | 75855.6 | 33.4 | 212655 |
| 1 | 4096 | 1024 | PASS | 200/200 | 0 | 0.91 | 3703.2 | 926.0 | 257.1 | 37.4 | 38557 |
| 1 | 4096 | 4096 | PASS | 200/200 | 0 | 0.25 | 1011.1 | 1011.4 | 182155.6 | 56.1 | 411868 |
| 1 | 8192 | 1024 | PASS | 200/200 | **4** | 0.61 | 4938.0 | 617.4 | 34582.5 | 85.9 | 122467 |
| 1 | 8192 | 4096 | PASS | 200/200 | **41** | 0.16 | 1309.5 | 654.9 | 315066.9 | 96.4 | 709763 |
| 1 | 16384 | 1024 | PASS | 200/200 | **5** | 0.31 | 5088.6 | 318.1 | 182187.1 | 90.9 | 275155 |
| **1** | **16384** | **4096** | **PASS** | **200/200** | **16** | 0.09 | 1441.1 | 360.4 | 850169.5 | 92.2 | 1227810 |
| 8 | 1024 | 1024 | PASS | 200/200 | 0 | 2.05 | 2091.3 | 2091.6 | 20723.0 | 25.6 | 46922 |
| 8 | 1024 | 4096 | PASS | 200/200 | 0 | 0.40 | 410.4 | 1642.0 | 147954.9 | 34.5 | 289351 |
| 8 | 4096 | 1024 | PASS | 200/200 | 0 | 1.02 | 4168.5 | 1042.4 | 58268.7 | 50.3 | 109708 |
| 8 | 4096 | 4096 | PASS | 200/200 | 0 | 0.25 | 1014.8 | 1015.0 | 258335.3 | 56.5 | 489499 |
| 8 | 8192 | 1024 | PASS | 200/200 | **7** | 0.61 | 4967.3 | 621.1 | 112483.7 | 80.9 | 195266 |
| 8 | 8192 | 4096 | PASS | 200/200 | **17** | 0.16 | 1277.7 | 639.0 | 462911.8 | 76.8 | 777480 |
| 8 | 16384 | 1024 | PASS | 200/200 | 0 | 0.32 | 5136.8 | 321.1 | 268242.4 | 80.3 | 350396 |
| **8** | **16384** | **4096** | **PASS** | **200/200** | **13** | 0.09 | 1443.3 | 360.9 | 946036.6 | 85.1 | 1294615 |
| 64 | 1024 | 1024 | PASS | 200/200 | 0 | 2.08 | 2119.6 | 2119.9 | 29371.8 | 24.7 | 54628 |
| 64 | 1024 | 4096 | PASS | 200/200 | 0 | 0.40 | 411.4 | 1645.7 | 157380.7 | 34.4 | 298122 |
| 64 | 4096 | 1024 | PASS | 200/200 | 0 | 1.02 | 4168.5 | 1042.4 | 68681.2 | 50.3 | 120121 |
| 64 | 4096 | 4096 | PASS | 200/200 | 0 | 0.25 | 1014.8 | 1015.0 | 268747.8 | 56.5 | 499911 |
| 64 | 8192 | 1024 | PASS | 200/200 | **7** | 0.61 | 4967.3 | 621.1 | 122896.2 | 80.9 | 205678 |
| 64 | 8192 | 4096 | PASS | 200/200 | **17** | 0.16 | 1277.7 | 639.0 | 473324.3 | 76.8 | 787892 |
| **64** | **16384** | **1024** | **PASS** | **200/200** | 0 | 0.32 | 5136.8 | 321.1 | 278654.9 | 80.3 | 360809 |
| **64** | **16384** | **4096** | **PASS** | **200/200** | **13** | 0.09 | 1443.3 | 360.9 | 956449.1 | 85.1 | 1305027 |

**关键规律**（与根因分析一致）：只有 `il ≥ 8192`（即输入长度超过 `chunked_prefill_size=4096`）的用例才会触发真实 retract；`il=1024/4096` 的 8 组用例 retract 均为 0，符合预期。共 140 次 retract 全部被正确处理，**没有一次演变成 `ValueError`/服务崩溃**。

---

## 8. 重点验证：原崩溃场景（il=16384, 高并发）

原始 bug（`0721-Retract错误.md`）复现条件为 `il=16384 > chunked-prefill-size=4096` 且高负载。本次测试中三组 `il=16384, ol=4096` 用例（rr=1/8/64）**均实际触发了 retract（分别 16/13/13 次）**，服务端日志确认新逻辑正常工作：

```
[2026-07-22 01:22:40] KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.1427 -> 0.3414
2026-07-22 01:22:40 INFO [hisim] SGLang retracted 1 request(s) for KV-cache pressure; resetting PD state to WAITING_PREFILL: ['6ecbf309dc734b85a1432f8cdcdd5b3b']
```

对比修复前的报错（`0721-Retract错误.md` 记录）：
```
ValueError: cannot schedule prefill for rid=... in phase=running_decode
  at pd_backend_a.py:134 (_validate_prefill_phase)
```

**本次测试全程未再出现该 ValueError 或任何服务端崩溃/进程退出**，服务端进程（PID 1650681）从 01:13:08 到 01:42:11 连续存活 29 分 3 秒，跑完全部 24 组用例后才被主动 SIGTERM 优雅关闭（日志：`Gracefully exiting... Remaining number of requests 0`）。

---

## 9. 附带观察（非缺陷，供参考）

- `rr=8` 与 `rr=64` 在 `il≥4096` 时指标几乎完全一致：这是因为单请求端到端延迟（几十秒到上千秒不等）远大于两种到达速率下 200 个请求的到达时间窗口（200/8=25s vs 200/64=3.1s），请求早早全部到达排队，`max_running_per_replica=64` 成为唯一瓶颈，故 rr 差异被"抹平"，属预期的排队论现象，非 Bug。
- 未发现除已处理的 `flush_cache` 代理问题外的其它异常/警告（仅有预期内的 "CPU device enabled, low performance expected"、"不支持 Intel AMX" 等模拟模式提示）。

---

## 10. 结论

1. **PD 分离（single_process / BackendA）场景下，按 260721 文档规定的全部 24 组参数矩阵测试，服务端全程零崩溃、零异常堆栈。**
2. 原 bug 的触发条件（`il=16384`、高并发）在本次测试中被真实复现（共 140 次 retract 事件，其中 il≥8192 的用例贡献全部触发次数），且**全部被新增的 `reset_for_retract` 逻辑正确吸收**，验证了修复的有效性。
3. 测试过程中发现并规避了一个与本 Bug 无关的环境问题（本地代理拦截 `127.0.0.1` 请求导致 `flush_cache` 403），已通过 `no_proxy` 配置解决，不影响本次结论。
4. 修复代码目前仍为**工作区未提交状态**（9 个文件），建议在确认本报告后执行 `git commit` 固化此次修复；是否提交及提交信息由用户决定，本次未擅自提交。
5. 本次测试未覆盖 PD 合并（TP=2/DP=2）场景，按用户明确要求跳过，如需要可另行安排。

---

## 附：测试产出文件位置

- 测试脚本与全部原始日志/指标：会话工作区 `hisim_pd_test_260721/`
  - `pd_disagg_260721.json`：本次使用的 PD 分离配置
  - `run_matrix.sh`：24 组用例驱动脚本
  - `matrix_results.log`：逐用例 PASS/FAIL 与耗时汇总
  - `server_logs/server.log`：服务端完整日志（含全部 140 条 retract 记录）
  - `cases_same_seed_flush_cache_32kv_GBps/`：每组用例的独立日志（`log_*.txt`）、指标文件（`metrics_*.jsonl`）及 Hisim 输出目录快照
