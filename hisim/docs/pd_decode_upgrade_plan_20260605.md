# PD Decode Upgrade Plan (2026-06-05)

This note summarizes why we should upgrade the current PD decode simulation,
how to baseline the existing model, and how to run an A/B evaluation after
implementing per-replica decode queues.

Scope: single-node simulation in HiSim; no claim of exact production replay.

---

## 1. Why Upgrade Is Needed

Current PD path captures stage semantics well, but decode scheduling in the
SGLang hook is still driven as one central decode batch per loop tick. In
decode-heavy workloads, this can under-represent multi-replica parallelism.

Observed structural gap:

- Prefill admission is request-granular and naturally fans out across replicas.
- Decode admission is batch-granular and often lands as a single batch decision
  per tick.
- Real engines commonly run multiple decode workers with independent local
  queues, which increases overlap opportunities.

Impact on metrics:

- Typical risk is conservative (higher) E2E in decode-bottleneck regimes.
- Tail metrics (P95/P99) are more sensitive than mean under mixed arrivals.

---

## 2. Baseline Definition (Current Model)

Use the current implementation as Baseline A and freeze it before any decode
queue upgrade.

Baseline A characteristics:

- Existing PD controller/state transitions unchanged.
- Existing hook glue unchanged.
- Existing decode batch admission behavior unchanged.
- Existing metric definitions unchanged (`TTFT`, `ITL`, `E2E`, stage durations).

Recommended baseline capture:

1. Pin commit/tag for reproducibility.
2. Run a fixed workload matrix (same seeds, same request traces).
3. Store per-run artifacts:
   - `metrics.json`
   - `request.jsonl`
   - `iteration.jsonl`
   - sweep summary table (if using sweep runner)

---

## 3. Upgrade Target (Candidate B)

Implement decode scheduling as independent per-replica queues in simulation
runtime, while preserving output schema and metric semantics.

Candidate B goals:

- Each decode replica has its own runnable queue and availability clock.
- A decode-ready request is routed to one replica queue (policy configurable,
  default earliest-available).
- Decode step execution is advanced per replica timeline, not only by a single
  global batch decision.

Non-goals in this phase:

- No cross-node/NIC contention modeling.
- No hard GPU placement/NUMA/NVLink topology replay.
- No change to user-facing metrics definition.

---

## 4. A/B Experiment Design

Compare Baseline A vs Candidate B on identical workloads and predictor inputs.

### 4.1 Workload matrix

Use at least these workload families:

- W-low: low concurrency, short outputs (sanity/consistency)
- W-mid: medium concurrency, mixed input/output lengths
- W-high-decode: high concurrency, long outputs (decode bottleneck)
- W-bursty: burst arrivals, uneven request lengths

Keep fixed across A/B:

- request trace (arrival times and token lengths)
- model/backend/version/device settings
- disagg config except decode scheduling implementation

### 4.2 Metrics to compare

Primary:

- `mean_e2e_latency_ms`
- `p50_e2e_latency_ms`
- `p95_e2e_latency_ms`
- `p99_e2e_latency_ms`

Secondary:

- `mean_ttft_ms`, `p95_ttft_ms`, `p99_ttft_ms`
- `mean_itl_ms`, `p95_itl_ms`, `p99_itl_ms`
- throughput (`output_throughput`, `request_throughput`)
- stage metrics:
  - `mean_prefill_queue_ms`
  - `mean_kv_transfer_ms`
  - `mean_decode_queue_ms`

### 4.3 Comparison output

For each workload, report:

- absolute deltas: `B - A`
- relative deltas: `(B - A) / A`
- directionality summary (improved/worse/noisy)

Also report aggregate across workloads:

- median relative delta for E2E P50/P95/P99
- worst-case relative delta

---

## 5. Success Criteria for This Upgrade

Candidate B is considered useful if:

1. No regression in non-disagg path behavior.
2. Metric schema stays backward-compatible.
3. In decode-heavy workloads, B demonstrates expected directionality:
   - lower decode queue pressure or better throughput
   - E2E tails improve or become more realistic relative to known behavior.
4. No structural instability (no deadlocks, no unbounded queue growth,
   deterministic replay still possible).

---

## 6. Known Limits After This Upgrade

Even with per-replica decode queues, simulation still differs from production:

- runtime jitter and kernel launch variance are abstracted
- queueing/policy details in real engines are simplified
- overlap is simulated, not measured from real kernels

Therefore, treat this as a better approximation for ranking and what-if
analysis, not a strict replacement for production tail validation.

---

## 7. Next Step Beyond Candidate B

Planned advanced modeling direction: true P/D overlap in simulator timeline.

Potential phase goals:

- represent prefill and decode as concurrently advancing pipelines
- model overlap windows explicitly under shared constraints
- keep compatibility with current per-request stats output

Expected benefit:

- further reduce structural error in mixed high-load workloads
- improve confidence of E2E tail trend prediction

---

## 8. Execution Checklist

1. Freeze Baseline A commit and run workload matrix.
2. Implement Candidate B decode queue model behind a switch.
3. Re-run exactly same matrix.
4. Produce A/B delta report (E2E P50/P95/P99 first).
5. Decide rollout policy:
   - default on for decode-heavy planning, or
   - keep both modes and select per scenario.
