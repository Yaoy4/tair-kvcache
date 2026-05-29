# Phase 6a — A/B Equivalence Tolerance Spec

Status: SPEC (no harness code).
Owner: PD disaggregation track (phases 5→6 bridge).
Consumers: Phase 6 harness (`pd_ab_harness.py`), Phase 6 CI gate.

## Purpose

Define the numeric and structural tolerances under which BackendA
(single-process, virtual-time) and BackendB (two-process, spawned workers)
must be considered **equivalent** for the same `SimulationArgs` input.

This spec is written BEFORE the harness so the harness implements a target,
not the other way around. If the harness's measured numbers contradict this
spec, the spec wins — investigate the backend, do not loosen the spec
unilaterally.

## Scope

In-scope:
- Single-node, single workload, identical `SimulationArgs` (only
  `disagg.backend` differs: `single_process` vs `two_process`).
- Same model, same per-role predictor config (device, tp/ep/dp/pp, dtype,
  KV dtype, replicas, scale factors).
- Same KV transfer model (`bw_gbps`, `latency_us`).
- Same workload (request arrival sequence, input/output lengths).
- Backends compared on the metrics already produced by
  `pd_metrics.populate_request_stats` and `calc_metrics`.

Out-of-scope (do NOT use this spec to gate):
- BackendB worker startup / shutdown wall-clock cost.
- Cross-node, NIC contention, GPU placement (per ground rules: out of scope
  for the whole PD track).
- Real-runtime (non-HiSim) absolute latency validation.
- Cross-model or cross-hardware regressions.

## Invariants (MUST hold exactly)

These are structural identities. Any violation is a bug, not a tolerance
miss.

1. **Request-count parity.** Number of FINISHED requests in BackendA
   equals number in BackendB.
2. **RID set parity.** The set of FINISHED request IDs is identical.
3. **Output length parity.** For every rid, `output_length` is identical
   (the workload, not the backend, controls token counts).
4. **Phase progression.** Every FINISHED request in both backends passes
   through the canonical state sequence
   `QUEUED → RUNNING_PREFILL → KV_TRANSFER → QUEUED_DECODE →
   RUNNING_DECODE → FINISHED`, with monotonic non-decreasing timestamps:
   `arrival_time ≤ prefill_start_time ≤ prefill_end_time ≤ kv_ready_time
   ≤ decode_start_time ≤ decode_end_time`.
5. **Non-negative stage durations.** `prefill_queue_wait`,
   `kv_transfer_time`, `decode_queue_wait` are all `≥ 0` in both backends.
6. **Schema parity.** `calc_metrics` returns the same key set for A and B
   (including the zero-valued PD keys from phase 4b).

## Numeric tolerances (per-request)

For each rid, compare BackendA value `a` and BackendB value `b`. Pass iff
`abs(a - b) ≤ max(abs_tol, rel_tol * max(abs(a), abs(b)))`.

| Metric | abs_tol | rel_tol | Rationale |
|---|---|---|---|
| `prefill_end_time - arrival_time` | 1e-6 s | 1e-9 | Pure analytic predict. |
| `kv_ready_time - prefill_end_time` (KV xfer) | 1e-6 s | 1e-9 | `BandwidthTransferModel` is closed-form. |
| `decode_end_time - decode_start_time` | 1e-6 s | 1e-9 | Pure analytic predict. |
| `prefill_queue_wait` | 1e-6 s | 1e-9 | Function of replica pool ordering only. |
| `decode_queue_wait` | 1e-6 s | 1e-9 | Same. |
| `e2e_latency` (decode_end - arrival) | 1e-6 s | 1e-9 | Sum of the above. |

Rationale for the tight bounds: virtual-time HiSim has **no source of
non-determinism** that should appear across A/B for the same inputs.
The predictors are deterministic functions; the replica pool is a
deterministic earliest-free pick with stable tie-breaking; the transfer
model is closed-form. The 1e-6 s floor only exists to absorb IEEE-754
rounding (`float64 add` over O(10⁴) requests has worst-case error well
under 1e-9 s per stage; 1e-6 is three orders of magnitude of safety).

If a metric exceeds these bounds, the failure mode is one of:
- Replica-pool tie-breaking differs across backends (real bug).
- BackendB worker received a different scheduler ordering than BackendA
  (real bug — IPC reordered batches).
- A predictor cached state that didn't survive process boundary
  (real bug — predictor is not pure).

## Aggregate tolerances (per-run)

These guard against systematic drift even if individual rids pass.

| Metric | abs_tol | rel_tol |
|---|---|---|
| mean TTFT (= mean of `prefill_end - arrival`) | 1e-6 s | 1e-9 |
| p50 TTFT | 1e-6 s | 1e-9 |
| p95 TTFT | 1e-6 s | 1e-9 |
| p99 TTFT | 1e-6 s | 1e-9 |
| mean E2E | 1e-6 s | 1e-9 |
| p50 / p95 / p99 E2E | 1e-6 s | 1e-9 |
| mean / p95 / p99 ITL | 1e-6 s | 1e-9 |
| total throughput (req/s) | 1e-6 | 1e-9 |
| total output tokens/s | 1e-6 | 1e-9 |

## Allowed asymmetries (NOT tolerance violations)

The harness MUST ignore these — they are intentional and not part of the
A/B contract:

1. Wall-clock simulation runtime (BackendB pays IPC + spawn overhead).
2. `BackendBWorkerError` traceback strings (BackendA cannot produce them).
3. Per-replica `busy_until` internal state at run end (private to backend).
4. `atexit` registration count (BackendB registers one, BackendA none).
5. Memory-usage figures of the harness process.

## Failure-mode classification

When the harness reports a mismatch, classify by the first matching rule:

1. **STRUCT** — any invariant 1–6 above fails. Block CI immediately.
2. **PER_REQ** — at least one rid exceeds per-request tolerance but
   all aggregates pass. Block CI; likely scheduling/order bug.
3. **AGGREGATE** — aggregates exceed tolerance even if per-rid mostly
   passes. Block CI; likely systematic predictor or KV-transfer drift.
4. **WARN** — only allowed-asymmetry items differ. Do not block.

## Workload matrix the harness MUST cover

(Defined here so 6 can implement against it; this is part of 6a so the
acceptance bar is fixed before code is written.)

| Case | Requests | Input len | Output len | Replicas (P/D) | Purpose |
|---|---|---|---|---|---|
| W1: single | 1 | 64 | 16 | 1 / 1 | Smoke. |
| W2: serial | 8 | 64 | 16 | 1 / 1 | Queue ordering. |
| W3: concurrent | 32 | 128 | 64 | 2 / 2 | Replica-pool parity. |
| W4: long-decode | 8 | 32 | 256 | 1 / 1 | Decode-bound. |
| W5: long-prefill | 8 | 1024 | 16 | 1 / 1 | Prefill-bound. |
| W6: KV-bw stress | 16 | 512 | 32 | 2 / 2, bw=10 Gbps | Transfer model parity. |

All six must pass STRUCT + PER_REQ + AGGREGATE tolerances for A/B
equivalence to be declared.

## Non-goals for the harness

- Validating absolute latency against any real runtime.
- Performance benchmarking BackendA vs BackendB.
- Testing real AIC DB models (the real-AIC path is covered by the
  existing 5b.1 smoke test and is allowed to skip when the DB lacks the
  device/version triple).

## Acceptance for Phase 6a

This document, committed and tagged `pd-phase-6a`. No code, no tests.
The next slice (Phase 6 proper) implements the harness against this spec.
