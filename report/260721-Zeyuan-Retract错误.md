

- P1D1的情况下跑到 `il=16384、ol=4096` 的情况下遇到了报错

  ```bash
  [2026-07-21 19:49:50] Scheduler hit an exception: Traceback (most recent call last):
    File "/mnt/nfs02/users/zeyuanwa/project/.venv/lib/python3.13/site-packages/sglang/srt/managers/scheduler.py", line 2706, in run_scheduler_process
      scheduler.event_loop_normal()
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    File "/mnt/nfs02/users/zeyuanwa/project/.venv/lib/python3.13/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
      return func(*args, **kwargs)
    File "/mnt/nfs02/users/zeyuanwa/project/.venv/lib/python3.13/site-packages/sglang/srt/managers/scheduler.py", line 993, in event_loop_normal
      result = self.run_batch(batch)
    File "/mnt/nfs02/users/zeyuanwa/project/tair-kvcache/hisim/src/hisim/simulation/sglang/sglang_hook.py", line 1187, in wrapped_run_batch
      pd_latency = admit_prefill_batch_latency(
          C_SchedulerHook.PD_BACKEND, states, now_clock
      )
    File "/mnt/nfs02/users/zeyuanwa/project/tair-kvcache/hisim/src/hisim/simulation/pd_runtime.py", line 262, in admit_prefill_batch_latency
      _, end_t = backend.try_admit_prefill_batch(states, now)
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
    File "/mnt/nfs02/users/zeyuanwa/project/tair-kvcache/hisim/src/hisim/simulation/pd_backend_a.py", line 352, in try_admit_prefill_batch
      self._validate_prefill_phase(req)
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^
    File "/mnt/nfs02/users/zeyuanwa/project/tair-kvcache/hisim/src/hisim/simulation/pd_backend_a.py", line 134, in _validate_prefill_phase
      raise ValueError(
      ...<2 lines>...
      )
  ValueError: cannot schedule prefill for rid='0019aeb5c76b4bc3b7d0becd3fd07128' in phase=running_decode
  
  [2026-07-21 19:49:50] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
  Killed
  ```

  - 可以稳定复现，只要 il 超过 chunked-prefill-size 就会触发。并且更早的日志中存在SGLang的retract操作：

    ```
    [2026-07-21 20:23:46] KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.0980 -> 0.8353
    ```

    > **Retract操作：**
    >
    > 请求完成 prefill 后，其 KV cache 会持续保留，并且每生成一个 token 还需要继续扩展 KV。SGLang 在准备下一轮 decode 时会检查“当前空闲 KV token 数是否足够支持 running batch 的下一步 decode”，如果不足就会从 running decode batch 中撤回请求。因此撤回的请求需要重新Prefill

  - **触发原因：**KV Pool 耗尽导致请求Retract 引发状态回退。

    - 高并发长序列使 SGLang 的 KV Cache 池耗尽，SGLang 因而将一个正在 Decode 的请求 retract，释放其 KV Cache，并用相同 rid 将请求放回队列重新 Prefill 以重建 KV；但当前 HiSim 的单进程 PD 模拟没有同步这一状态回退，仍将该请求记录为 RUNNING_DECODE，所以它再次进入 Prefill 时触发状态异常错误。
    - 与 `max_running_per_replica`的配置无关，即使不设置此参数也仍然会触发。SGLang的 KV pool 似乎是自动计算的。

