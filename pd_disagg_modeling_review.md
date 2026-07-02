# HiSim PD Disaggregation — Modeling Review

**Scope:** Correctness review of the prefill/decode (PD) disaggregation feature built on top of HiSim (single-process **Backend A** path, integrated through the SGLang hook).

**Question answered:** Does the implementation faithfully model real PD disaggregation, and where does it diverge?

**TL;DR:** The per-role predictor wiring, decode batching, and KV-byte derivation are sound. However, the single shared virtual clock means **concurrent prefill↔decode overlap is not modeled**, which undercuts the core disaggregation benefit. There is also a **correctness bug under chunked prefill** and a **misleading `prefill_queue_wait` stage metric**.

---

## Background (how the model works)

- HiSim is a **virtual-time** simulator. `run_batch` does not execute the model; the hook calls a latency predictor for a virtual duration and advances a global clock.
- For PD, a "virtual-time accounting layer" adds two replica pools (prefill, decode), each tracked as a `busy_until` vector. A request is scheduled onto the earliest-free replica.
- Batch **composition** always comes from SGLang's native scheduler — HiSim only re-prices the latency of the batch SGLang formed (extend = prefill, decode = decode).

---

## Critical findings

### 1. Concurrent prefill↔decode overlap is NOT modeled (shared global clock serializes the two roles)

The defining benefit of PD disaggregation is that prefill and decode run on **separate hardware concurrently**, so a long prefill does not stall in-flight decode.

In the SGLang hook's extend branch, the single global clock is advanced by the prefill batch latency **plus** the KV transfer time:

```python
predicted_latency = pd_latency + kv_transfer_extra
```

SGLang's loop runs **either** an extend batch **or** a decode batch per iteration — never both. So whenever a prefill batch runs, the global clock jumps forward by `prefill_dur + kv_transfer`, and every in-flight decode request is frozen in virtual time for that window.

The `busy_until` replica pools only model **intra-role** parallelism (multiple prefill replicas, multiple decode replicas). They do **not** decouple the two roles on the shared clock.

**Impact:**
- TPOT / ITL inflated whenever prefill and decode interleave.
- Decode throughput understated.
- The core disagg benefit (decode isolation from prefill bursts) is largely invisible — an aggregated run and a disagg run look more similar than they should.

**Nature:** Structural limitation of single-process Backend A. Only the two-process **Backend B** can represent true overlap. Should be stated explicitly in any published results.

### 2. KV transfer time also freezes decode (compounds #1)

`kv_transfer_extra` adds the **slowest** request's full KV transfer time to the global clock. During a real KV transfer, decode of *other* requests keeps running; here the entire simulation is frozen for the transfer window. This was needed to make decode start after KV arrives, but it does so by stalling everything. The distortion is worst under low KV bandwidth.

### 3. Chunked prefill double-counts prefill and corrupts request state

The extend branch builds PD request states with **no phase guard**, and the prefill admission path unconditionally re-arrives the request on every appearance:

```python
self._controller.on_request_arrival(req, now)   # resets phase=WAITING_PREFILL, re-enqueues
admitted = self._controller.admit_prefill(capacity=1, now=start)
```

`finalize_prefill_batch` also runs after **every** extend batch, pushing the request into KV_TRANSIT. If SGLang **chunked prefill** is enabled, a single request appears across multiple extend batches and will:

- be re-scheduled onto a prefill replica each chunk (prefill time over-counted),
- be moved to KV_TRANSIT on the first chunk (premature),
- be reset to WAITING_PREFILL on the next chunk (state corruption).

Additionally, `PDRequestState.input_length` is captured once from the **first** chunk's `extend_input_len`, so a chunked request records one chunk's length instead of the full prompt — which also undercounts KV transfer size (#5) and the decode KV base (#8).

**Impact:** If chunked prefill is on, prefill accounting is wrong. If runs are always unchunked (one extend per request), this is latent but safe.

**Fix direction:** Guard already-tracked requests from re-arrival; only finalize prefill on the final chunk; capture full prompt length.

---

## Moderate findings

### 4. Prefill is priced per-request, not as a fused batch

Prefill latency is computed by **looping** over requests and calling the predictor once per request with a **single-request** batch. Decode, by contrast, makes a single fused predictor call (`batch_size = N` with per-request KV lengths). So the "prefill batch latency" is a scheduling envelope (`max(end_times) - now`) over independently-priced solo prefills spread across replicas — fused/chunked multi-request prefill interference is not represented.

### 5. No KV-transfer bandwidth contention

The transfer model is purely `latency + bytes / bw` **per request**, independent of how many transfers are in flight. N requests finishing prefill simultaneously each get full bandwidth with no sharing. Consistent with the documented "NIC congestion out of scope," but it means the KV-bandwidth knob is an idealized per-request ceiling, not a shared link.

### 6. `prefill_queue_wait` metric is structurally ~0 (wrong arrival baseline)

`arrival_time` is set when the request **first appears in an extend batch**, not at true request arrival / SGLang admission. Since `prefill_queue_wait = prefill_start_time - arrival_time` and both are essentially `now` when a prefill replica is free, this stage metric collapses to ~0 and misses the time a request actually spends in SGLang's `waiting_queue`. (End-to-end TTFT measured by the benchmark client is still correct; only the PD stage breakdown is misleading.)

### 7. `max_running_per_replica` is not enforced in the live hook decode path

The offline A/B harness caps decode admission (`idle_replicas * max_running_per_replica`), but the live hook admits **every** rid in the SGLang decode batch regardless of the per-role limit. Decode batch size is implicitly bounded by SGLang's own scheduler, but the configured per-replica decode batch-slot limit is silently ignored in the hook. Harness and live-hook runs therefore use different decode admission semantics.

---

## Minor findings

### 8. Decode KV base ignores prefix-cache reuse

The decode KV length base is set to `input_length`, ignoring any reused/cached tokens from prefix caching. Small effect unless prefix caching is heavy.

### 9. PD termination can diverge from SGLang's stop condition

The controller marks a request FINISHED when `decode_step_count >= output_length`. Once finished, it drops out of the priced decode set; if SGLang continues decoding (EOS vs `max_new_tokens` mismatch), those extra steps are not priced. Aligned when `output_length == max_new_tokens`.

### 10. Wasted aggregated prediction in PD mode

In both branches the aggregated predictor result is computed and then overwritten by the PD value. Harmless, but it is dead compute, and the decode aggregated batch uses SGLang's `past_kv` while the actual priced value uses the PD state's — easy to confuse when debugging.

---

## What is modeled correctly

- **Decode is genuinely batched** — a single fused predictor call with `batch_size` and per-request KV lengths, capturing intra-batch interference.
- **Per-role independent predictors** — device, TP/EP/DP/PP, data type, and KV dtype are independent per role.
- **`kv_bytes_per_token` derived canonically** via the shared utils helper, not hand-picked.
- **Intra-role replica parallelism** via earliest-free `busy_until` selection.
- **Per-request `prefill_end_time`** so a fast request is not penalized by the slowest member of its batch.
- **Clean request state machine** and KV-ready gating in the controller.

---

## Priority recommendation

| Priority | Finding | Action |
|---|---|---|
| P0 | #1, #2 — no P/D overlap on shared clock | Document as a known Backend A limitation; validate overlap-sensitive conclusions on Backend B (two-process). |
| P0 | #3 — chunked prefill correctness | Either assert/guard against chunked prefill, or handle multi-chunk requests properly. |
| P1 | #6 — misleading `prefill_queue_wait` | Thread the true arrival/queue-start timestamp into `arrival_time`. |
| P2 | #4, #5, #7 | Note as modeling simplifications; fix if the target workload is sensitive. |
| P3 | #8, #9, #10 | Low impact; address opportunistically. |

**Bottom line:** Backend A is a sound *relative* what-if tool for intra-role scaling (replica counts, KV bandwidth, per-role hardware), but it cannot represent prefill/decode temporal overlap, so absolute disagg-vs-aggregated comparisons and decode-tail claims should be validated on the two-process backend.
