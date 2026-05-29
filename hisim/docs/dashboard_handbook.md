# HiSim Sweep Dashboard — First-Run Handbook

A guide for someone running the HiSim visualization stack (`sweep_dashboard.py`)
for the first time.

---

## 1. What this dashboard is

`sweep_dashboard.py` is a Streamlit app for exploring HiSim PD-disagg / aggregated
sweep results. It has two data sources:

1. **Sweep CSV** — `summary.csv` produced by [`hisim/tools/aic_topn_to_hisim_sweep.py`](../tools/aic_topn_to_hisim_sweep.py).
   One row per simulated configuration.
2. **Single Run** — an in-app form that runs ONE BackendA PD sim (analytic
   predictor, no AIC perf DB) and appends a row to session state.

Tabs: **Pareto · Stage breakdown · PD heatmap · Table · Comparison** (multi-select
overlay across sweep rows + single runs).

Backend: **BackendA only** (single-process virtual-time, no `mp.spawn`).

---

## 2. Environment

One-time setup:

```bash
cd /home/cjia/projects/tair-kvcache
source venv/bin/activate
pip install -e "hisim[dashboard]"   # pulls plotly, jinja2, streamlit
```

Required env vars for every run (HiSim hooks SGLang in CPU mode):

```bash
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/cjia/projects/tair-kvcache/hisim/src:/home/cjia/projects/aiconfigurator/src
```

---

## 3. Launch the dashboard

```bash
streamlit run hisim/tools/sweep_dashboard.py \
  --server.headless true --server.port 8501
```

Open http://localhost:8501. You can also pass a default CSV after `--`:

```bash
streamlit run hisim/tools/sweep_dashboard.py -- \
  --summary-csv hisim/output/my_aic_bridge_sweep_disagg_top3/summary.csv
```

---

## 4. Using existing `summary.csv` files

Already present in this workspace (good for a first look):

| Path | Rows |
| --- | --- |
| [hisim/output/my_aic_bridge_sweep_disagg_top3/summary.csv](../output/my_aic_bridge_sweep_disagg_top3/summary.csv) | 3 |
| [hisim/output/my_aic_bridge_sweep_disagg_top3_l2/summary.csv](../output/my_aic_bridge_sweep_disagg_top3_l2/summary.csv) | 3 |
| [hisim/output/my_aic_bridge_sweep_disagg_top3_no_cache/summary.csv](../output/my_aic_bridge_sweep_disagg_top3_no_cache/summary.csv) | 3 |
| [hisim/output/my_aic_bridge_sweep_disagg_top1/summary.csv](../output/my_aic_bridge_sweep_disagg_top1/summary.csv) | 1 |
| [hisim/output/aic_bridge_sweep_smoke/summary.csv](../output/aic_bridge_sweep_smoke/summary.csv) | 1 |

Paste the absolute path into the sidebar **"Summary CSV path"** input. The
dashboard reads it through `sweep_data.load_sweep()`, which auto-enriches each
row from its `result_file` (bench JSON) and `sim_config` JSON sitting in the
same per-run subfolder (`topNN_rowM/`).

---

## 5. Producing a NEW `summary.csv` from scratch

End-to-end flow: **AIConfigurator → CSV → sweep runner → summary.csv**.

### 5a. Get an AIC top-N CSV

Run AIConfigurator's `default` mode to produce the top-N Pareto configurations
(one row per config, columns include `tp`, `dp`, `ep`, `system`, stage cost
estimates, etc.). General form:

```bash
export PYTHONPATH=/home/cjia/projects/aiconfigurator/src
python -m aiconfigurator.cli.main default \
  --model        <hf_model_or_path> \
  --total-gpus   <N> \
  --system       <device_name>         # e.g. h200_sxm, b60
  --backend      <trtllm|sglang|vllm|auto> \
  --backend-version <ver> \
  --database-mode HYBRID \
  --top-n        5 \
  --isl 4000 --osl 500 --ttft 600 --tpot 50 \
  --save-dir     <out_dir>
```

Output (use `pareto.csv` as the AIC top-N CSV for §5b):
```
<out_dir>/
├── pareto.csv          ← merged agg+disagg (best for HiSim)
├── pareto_agg.csv
├── pareto_disagg.csv
└── …
```

#### Example: 8× B60, Qwen3-30B-A3B-FP8, SGLang

> Prerequisite: `aiconfigurator/src/aiconfigurator/systems/data/b60/sglang`
> symlink → `b60/vllm` (B60 ships with vLLM perf data only; reusing it for
> SGLang is acceptable for relative ranking).

```bash
cd /home/cjia/projects && source tair-kvcache/venv/bin/activate
export PYTHONPATH=/home/cjia/projects/aiconfigurator/src

python -m aiconfigurator.cli.main default \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --total-gpus 8 \
  --system b60 \
  --backend sglang \
  --backend-version 0.12.0 \
  --database-mode HYBRID \
  --top-n 5 \
  --isl 4000 --osl 500 --ttft 600 --tpot 50 \
  --save-dir /home/cjia/projects/tair-kvcache/hisim/output/aic_b60_qwen30b
```

If you don't have any AIC CSV yet, you can smoke-test §5b with the AIC CSV
that produced one of the existing sweeps (see `generator_config.yaml` /
`bench_run.sh` next to it).

### 5b. Run the sweep

```bash
cd /home/cjia/projects/tair-kvcache
python hisim/tools/aic_topn_to_hisim_sweep.py \
  --aic-csv   /path/to/pareto.csv \
  --top-n     3 \
  --output-dir hisim/output/my_first_sweep \
  --model-path /path/to/model \
  --emit-disagg \
  --cache-scenario no_cache \
  --request-rate  4
```

#### Example continued: B60 Qwen3-30B-A3B-FP8

```bash
cd /home/cjia/projects/tair-kvcache
export SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/cjia/projects/tair-kvcache/hisim/src:/home/cjia/projects/aiconfigurator/src

python hisim/tools/aic_topn_to_hisim_sweep.py \
  --aic-csv   hisim/output/aic_b60_qwen30b/pareto.csv \
  --top-n     5 \
  --output-dir hisim/output/my_b60_sweep \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --predictor-device-name b60 \
  --backend-version 0.12.0 \
  --data-type fp8 \
  --kv-cache-data-type fp8 \
  --emit-disagg \
  --cache-scenario no_cache \
  --request-rate 4
```

Useful flags:

- `--top-n N` / `--row-offset K` — slice AIC rows `[K, K+N)`.
- `--emit-disagg` — bridge writes the disagg block (prefill + decode predictors)
  consumed by `sim_args._disagg_from_dict`. Without it, the run is aggregated.
- `--kv-transfer-bw-gbps`, `--kv-transfer-latency-us` — KV-transfer model.
- `--cache-scenario {no_cache,l1,l2}` and `--request-rate` — workload knobs.
- `--dry-run` — parse + plan, no sims.
- `--no-plot` — skip the HTML auto-report (default it writes
  `<output-dir>/sweep_report.html`).

> **B60 caveat**: the B60 perf db is vLLM-only. Reusing it for SGLang via the
> `b60/sglang → b60/vllm` symlink works for **relative ranking and what-if**
> but not for absolute SLO certification. Pick `--data-type fp8` for
> Qwen3-30B-A3B-FP8; B60 MoE coverage is only `(2048,768,topk=8,128 experts)`
> and `(2048,1408,topk=4,60 experts)`, so other MoE models will hit
> extrapolation.

### 5c. Inspect output

```
hisim/output/my_first_sweep/
├── summary.csv          ← feed this to the dashboard
├── summary.json
├── sweep_report.html    ← static report (also viewable in browser)
└── topNN_rowM/
    ├── sim_config.json  ← SchedulerConfig used
    ├── result.json      ← bench-serving metrics
    ├── server.log
    └── bench.log
```

`summary.csv` columns include: `row_index`, `run_rank`, `status`,
`cache_scenario`, `request_rate`, `aic_tp/dp/ep/system`, `sim_config`,
`result_file`, `request_throughput`, `output_throughput`, `mean_ttft_ms`,
`mean_tpot_ms`, etc. The dashboard derives `total_gpus`, `topology`,
`pd_ratio`, and `is_pareto_optimal` on the fly.

---

## 6. Dashboard walkthrough

### Sidebar

- **Summary CSV path** — file path text box; blank = no sweep loaded.
- Filters: **Only Pareto**, **Only disagg**, **Only agg**, plus multi-selects
  over `cache_scenario` / `request_rate` / `topology`.
- **Single Run** expander — pick devices, TP, replicas, KV BW/latency, prompts,
  burst; click **Run single sim** to append one row to session state.

### Top metrics

`Rows · Pareto · Disagg · Aggregated` — counts after filters.

### Tabs

1. **Pareto** — scatter of `mean_ttft_ms` vs `output_throughput`; color = topology,
   size = `total_gpus`; dashed frontier overlay.
2. **Stage breakdown** — stacked bar of prefill_queue / kv_transfer / decode_queue
   for top-K rows by throughput (slider).
3. **PD heatmap** — pivot of `prefill_replicas × decode_replicas` on
   `p99_e2e_latency_ms` (disagg only).
4. **Table** — full filtered DataFrame.
5. **Comparison** — multi-select across sweep rows + single runs; renders
   side-by-side Pareto + stage-breakdown on the pinned subset only.

### Single-run results

Above the tabs, a collapsible table shows everything you ran this session +
a **Clear** button.

---

## 7. Static HTML report (no Streamlit needed)

If you just want a shareable artifact:

```bash
python hisim/tools/sweep_plot.py \
  --summary-csv hisim/output/my_first_sweep/summary.csv \
  --output      /tmp/report.html \
  --title       "My first sweep"
```

This uses the same `sweep_data` + `sweep_figures` layers and writes a single
self-contained HTML.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: streamlit` | `pip install -e "hisim[dashboard]"` |
| `ModuleNotFoundError: sglang` / hooks fail | export `SGLANG_USE_CPU_ENGINE=1` and `PYTHONPATH` (see §2) |
| Dashboard shows "No rows match the current filters" | clear sidebar filters; verify CSV has rows |
| Heatmap empty | only disagg rows have `prefill_replicas`/`decode_replicas` — uncheck "Only agg" |
| Single Run extremely slow | reduce `Num prompts` / `Burst`; use BackendA defaults |
| Stale charts after editing CSV | sidebar caches via `@st.cache_data`; reload the browser tab |

---

## 9. Where the code lives

- Data layer: [hisim/src/hisim/visualization/sweep_data.py](../src/hisim/visualization/sweep_data.py)
- Figures: [hisim/src/hisim/visualization/sweep_figures.py](../src/hisim/visualization/sweep_figures.py)
- HTML report: [hisim/src/hisim/visualization/sweep_report.py](../src/hisim/visualization/sweep_report.py)
- Single-run helper: [hisim/src/hisim/visualization/single_run.py](../src/hisim/visualization/single_run.py)
- Streamlit entry: [hisim/tools/sweep_dashboard.py](../tools/sweep_dashboard.py)
- Sweep runner: [hisim/tools/aic_topn_to_hisim_sweep.py](../tools/aic_topn_to_hisim_sweep.py)
- Static CLI: [hisim/tools/sweep_plot.py](../tools/sweep_plot.py)

Layering rule (don't violate): `sweep_data` (pandas only) → `sweep_figures`
(plotly Figures, no IO) → `sweep_report` (jinja2 HTML) / `sweep_dashboard`
(streamlit). Dashboard never reimplements logic from the lower layers.
