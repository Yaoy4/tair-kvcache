# PD-vs-AGG comparison runbook (single-node, N GPUs)

Reproducible 6-step procedure to compare aggregated vs prefill/decode-disaggregated
deployments inside HiSim, using AIConfigurator (AIC) as the perf-DB front end and
HiSim BackendA (`single_process`) as the simulator. Validated on 8×H100. The same
flow generalizes to other GPUs by swapping `--predictor-device-name`, the AIC
`--system`, and the framework `--backend`/`--backend-version`.

---

## 0. Environment (mandatory exports)

```bash
export PYTHONPATH=/home/cjia/projects/tair-kvcache/hisim/src:/home/cjia/projects/aiconfigurator/src
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
PY=/home/cjia/projects/tair-kvcache/venv/bin/python   # system `python` is not on PATH
```

Per-system knobs you will substitute (record before starting):

| Variable | 8×H100 value | meaning |
|---|---|---|
| `SYSTEM`            | `h100_sxm` | AIC `--system`, also HiSim `--predictor-device-name` |
| `BACKEND`           | `sglang`   | AIC framework backend with perf-DB coverage on SYSTEM |
| `BACKEND_VERSION`   | `0.5.10`   | a version directory present under `aiconfigurator/systems/data/<SYSTEM>/<BACKEND>/` |
| `TOTAL_GPUS`        | `8`        | node size |
| `MODEL`             | `Qwen/Qwen3-30B-A3B-FP8` | HF id |
| `DB_MODE`           | `HYBRID`   | use `SILICON` only if every kernel shape is covered |
| `WORK_ROOT`         | `hisim/output/walkthrough_8xh100` | output dir |

Check coverage before step 1:

```bash
ls aiconfigurator/src/aiconfigurator/systems/data/$SYSTEM/$BACKEND/    # must list a version dir
ls aiconfigurator/src/aiconfigurator/systems/data/$SYSTEM/$BACKEND/$BACKEND_VERSION/
# expect: gemm_perf.txt, moe_perf.txt, context_attention_perf.txt,
#         generation_attention_perf.txt, plus collective perf.
```
If no `sglang` directory exists for that GPU (e.g. B60 has `vllm` only), set
`BACKEND=vllm` and pick its version. If MoE shapes are sparse, force `DB_MODE=HYBRID`.

---

## Step 1 — AIC pareto sweep (AGG configurations)

Generates the AGG Pareto frontier on the target GPU.

```bash
cd /home/cjia/projects/aiconfigurator
$PY -m aiconfigurator.cli.main \
  --system $SYSTEM \
  --backend $BACKEND --backend-version $BACKEND_VERSION \
  --model $MODEL \
  --total-gpus $TOTAL_GPUS \
  --database-mode $DB_MODE \
  --output-dir $WORK_ROOT/aic
# produces: $WORK_ROOT/aic/pareto_agg.csv  (+ pareto_disagg.csv if AIC ran disagg)
```

Sanity:
```bash
head -2 $WORK_ROOT/aic/pareto_agg.csv
wc -l $WORK_ROOT/aic/pareto_agg.csv
```

---

## Step 2 — Pick AGG rows of interest

The top-N rows of `pareto_agg.csv` (by AIC's objective) become the AGG cells we
will replicate in HiSim. Note their row indices; the bridge will write per-row
JSON in step 3.

```bash
$PY hisim/tools/inspect_pareto.py --csv $WORK_ROOT/aic/pareto_agg.csv --top 5
```

---

## Step 3 — Bridge: AIC row → HiSim `sim_config.json` (AGG + synthesized DISAGG)

`aic_topn_to_hisim_sweep.py` calls `aic_to_hisim_bridge.py` under the hood for
each selected row, emits a HiSim `sim_config.json`, and (with
`--synth-disagg-from-agg`) also fabricates disagg variants by splitting the
same `TOTAL_GPUS` between prefill and decode pools.

```bash
cd /home/cjia/projects/tair-kvcache
$PY hisim/tools/aic_topn_to_hisim_sweep.py \
  --aic-csv $WORK_ROOT/aic/pareto_agg.csv \
  --top-n 5 \
  --output-dir $WORK_ROOT/sweep \
  --model-path $MODEL \
  --predictor-device-name $SYSTEM \
  --backend-version $BACKEND_VERSION \
  --database-mode $DB_MODE \
  --cache-scenario no_cache \
  --request-rate 4 \
  --synth-disagg-from-agg \
  --synth-disagg-profiles 1:7,2:6,4:4,6:2,7:1 \
  --disagg-backend single_process
```

This produces, per AIC row, a tree like:
```
$WORK_ROOT/sweep/top01_row<R>/             # AGG cell
$WORK_ROOT/sweep/top02_row<R>_syn_p1_d7/   # synthesized PD 1P+7D
$WORK_ROOT/sweep/top03_row<R>_syn_p2_d6/   # synthesized PD 2P+6D
$WORK_ROOT/sweep/top04_row<R>_syn_p4_d4/   # synthesized PD 4P+4D
$WORK_ROOT/sweep/top05_row<R>_syn_p6_d2/   # synthesized PD 6P+2D
$WORK_ROOT/sweep/top06_row<R>_syn_p7_d1/   # synthesized PD 7P+1D
```
Each cell directory contains a `sim_config.json` (HiSim's `SimulationArgs` JSON
form) ready for step 5.

**Knobs to remember:**
- `--synth-disagg-profiles` is `prefillGPUs:decodeGPUs` pairs that sum to
  `$TOTAL_GPUS`.
- `--kv-transfer-bw-gbps` defaults are NVLink-class. For PCIe-only fabrics (e.g.
  B60) override to ~256 (PCIe 5.0 x8 ≈ 32 GB/s).
- `--database-mode HYBRID` is mandatory whenever any kernel shape misses the DB.

---

## Step 4 — Run HiSim sweep

The sweep driver loads each `sim_config.json`, runs SGLang under
`SGLANG_USE_CPU_ENGINE=1` with HiSim's `sglang_hook` virtual-time predictor,
collects per-request metrics, and aggregates into a summary.

```bash
$PY hisim/tools/run_sweep_dir.py \
  --sweep-dir $WORK_ROOT/sweep \
  --summary-csv $WORK_ROOT/sweep/summary.csv
```

Per-cell artifacts written under each `top0*` directory:
- `result.json` — aggregates (TTFT mean/p99, TPOT, output throughput, E2E, etc).
- `request_stats.csv` — per-request raw timings.
- `run.log` — hook + scheduler log.

Validation (on 8×H100 row 48, AGG, TP=8): HiSim TPOT 4.90 ms vs AIC's predicted
4.86 ms (<1 % delta), output throughput 1 815.8 tok/s (~7 % vs AIC).

---

## Step 5 — Aggregate and rank AGG vs PD

```bash
$PY hisim/tools/summarize_sweep.py \
  --summary-csv $WORK_ROOT/sweep/summary.csv \
  --group-by mode \
  > $WORK_ROOT/sweep/comparison.txt
cat $WORK_ROOT/sweep/comparison.txt
```

Key columns: `mode` (`agg` / `syn_p<P>_d<D>`), `tpot_ms`, `ttft_p99_ms`,
`e2e_mean_ms`, `output_tok_per_s`, `completed`. Inspect interactively via the
Streamlit dashboard:

```bash
$PY -m streamlit run hisim/tools/sweep_dashboard.py \
  --server.headless true --server.port 8765 \
  --browser.gatherUsageStats false \
  -- --summary-csv $WORK_ROOT/sweep/summary.csv
```

---

## Step 6 — Closed-loop PD-vs-AGG smoke (optional, BackendA-only)

`pd_vs_agg_compare.py` is a self-contained event-loop check that exercises the
HiSim PD backend directly (no AIC DB needed, no SGLang) and prints a one-table
verdict. Useful for confirming the PD machinery wins on identical workloads
before trusting the full sweep numbers.

```bash
cd /home/cjia/projects/tair-kvcache/hisim/tools
$PY pd_vs_agg_compare.py --requests 32
```

Caveat: the predictor coefficients inside `pd_vs_agg_compare.py` are H100-tuned.
For a non-H100 system, either (a) re-derive analytic coefficients from
`aiconfigurator/systems/<SYSTEM>.yaml` (mem_bw, *_tc_flops) or (b) wire
`AICPredictorAdapter` against the target perf-DB.

---

## Per-GPU substitution cheat-sheet

| GPU | SYSTEM | BACKEND | BACKEND_VERSION | DB_MODE | KV bw default | Notes |
|---|---|---|---|---|---|---|
| H100 SXM | `h100_sxm` | `sglang` | `0.5.10` | `SILICON` (or HYBRID) | NVLink default | reference target |
| B60      | `b60`      | `vllm`   | `0.12.0`         | **HYBRID** required | `--kv-transfer-bw-gbps 256` | 24 GiB → TP ≥ 2 likely; no sglang DB; sparse collectives |
| H200     | `h200_sxm` | `sglang` (if present) | check tree | HYBRID for MoE | NVLink default | same flow |
| B200     | `b200_sxm` | `sglang` (if present) | check tree | HYBRID for MoE | NVLink default | same flow |

Always re-run the coverage check from step 0 first; the table is only a hint.

---

## Failure modes and quick triage

- `PerfDataNotAvailableError: Failed to query moe data ... Consider using HYBRID mode`
  → re-run step 1 (and step 4) with `--database-mode HYBRID`.
- `python: command not found` → use `$PY = venv/bin/python`, never bare `python`.
- HiSim sweep cell exits 0 but `completed=0` in `result.json` → arrivals never
  drained; check `--request-rate` vs the workload size in `sim_config.json`.
- AIC sweep returns no AGG rows → model + GPU memory constraint is infeasible
  at any TP/EP within `--total-gpus`. Increase `--total-gpus` or pick a smaller
  model.
- `TypeError: admit_prefill_batch_latency() takes 3 positional arguments but 4`
  → wrong `pd_runtime` signature in custom driver; match `pd_demo.py`'s loop
  exactly (`backend, batch, now`).

---

## Provenance

- Validated end-to-end on 8×H100 / Qwen3-30B-A3B-FP8 / sglang 0.5.10 on
  2026-06-02.
- HiSim PD backend: BackendA (`single_process`). BackendB is bit-identical by
  Phase 6 spec but ~order-of-magnitude slower; not used for replication.
- Ground rules: `/memories/hisim-aic-ground-rules.md` and
  `/memories/repo/pd-disagg-plan.md`.
