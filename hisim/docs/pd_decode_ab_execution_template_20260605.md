# PD Decode A/B Execution Template (2026-06-05)

This template is for running Baseline A (current model) vs Candidate B
(decode per-replica queue model) with reproducible inputs.

Use with:

- [pd_decode_upgrade_plan_20260605.md](hisim/docs/pd_decode_upgrade_plan_20260605.md)
- existing sweep/runbook conventions in [pd_vs_agg_runbook.md](hisim/docs/pd_vs_agg_runbook.md)

---

## 0. Environment

```bash
export PYTHONPATH=/home/cjia/projects/tair-kvcache/hisim/src:/home/cjia/projects/aiconfigurator/src
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
PY=/home/cjia/projects/tair-kvcache/venv/bin/python
```

Set experiment root:

```bash
EXP_ROOT=/home/cjia/projects/tair-kvcache/hisim/output/pd_decode_ab_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP_ROOT"/{baseline_a,candidate_b,reports,inputs}
echo "$EXP_ROOT"
```

---

## 1. Prepare Fixed Inputs (shared by A/B)

Goal: exact same request traces/config for both sides.

Place all frozen inputs under:

```text
$EXP_ROOT/inputs/
```

Suggested contents:

- `workload_matrix.json` (W-low/W-mid/W-high-decode/W-bursty)
- `sim_config_template.json`
- optional seed file (`seed.txt`)

Record source commit and branch:

```bash
cd /home/cjia/projects/tair-kvcache
git rev-parse --abbrev-ref HEAD | tee "$EXP_ROOT/inputs/git_branch.txt"
git rev-parse HEAD | tee "$EXP_ROOT/inputs/git_commit.txt"
```

---

## 2. Run Baseline A

Freeze Baseline A (current PD decode implementation) and run all workloads.

Directory convention:

```text
$EXP_ROOT/baseline_a/<workload_id>/
```

Concrete baseline runner (available today): `hisim/tools/aic_topn_to_hisim_sweep.py`.

Inputs to set once:

```bash
# Required: existing AIC pareto CSV (agg rows are fine)
AIC_CSV=/path/to/pareto_agg.csv

# Required: model and hardware knobs for predictor/perf-db lookup
MODEL=Qwen/Qwen3-8B
PREDICTOR_DEVICE=h100_sxm
BACKEND_VERSION=0.5.10
DB_MODE=HYBRID

# Optional: run only top-1 row for fast baseline smoke
TOP_N=1

# For agg AIC CSV, synthesize a fixed disagg profile for both A/B arms.
# Keep this identical between baseline and candidate.
DISAGG_PROFILE=4:4
```

Run four fixed workload buckets (same commands to be reused later for Candidate B):

```bash
declare -a WORKLOADS=(
    "w_low 32 256 32 2.0"
    "w_mid 64 512 128 4.0"
    "w_high_decode 64 256 512 4.0"
    "w_bursty 128 512 256 inf"
)

for item in "${WORKLOADS[@]}"; do
    read -r W NUM_PROMPTS RIN ROUT RR <<<"$item"
    OUT="$EXP_ROOT/baseline_a/${W}"
  mkdir -p "$OUT"

    "$PY" /home/cjia/projects/tair-kvcache/hisim/tools/aic_topn_to_hisim_sweep.py \
        --aic-csv "$AIC_CSV" \
        --top-n "$TOP_N" \
        --output-dir "$OUT" \
        --model-path "$MODEL" \
        --predictor-device-name "$PREDICTOR_DEVICE" \
        --backend-version "$BACKEND_VERSION" \
        --database-mode "$DB_MODE" \
        --cache-scenario no_cache \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$RIN" \
        --random-output-len "$ROUT" \
        --request-rate "$RR" \
        --synth-disagg-from-agg \
        --synth-disagg-profiles "$DISAGG_PROFILE"
done
```

Expected artifacts per workload (minimum):

- `<workload_id>/summary.csv`
- `<workload_id>/summary.json`
- `<workload_id>/top*/result.json`
- `<workload_id>/top*/server.log` and `<workload_id>/top*/bench.log`

Note: this step is fully runnable now and defines Baseline A.

---

## 3. Run Candidate B

Switch to upgraded decode queue mode and rerun the same workload matrix.

Directory convention:

```text
$EXP_ROOT/candidate_b/<workload_id>/
```

What to do:

- keep exactly the same workload matrix and command family as Section 2
- keep the same synthetic disagg profile (`$DISAGG_PROFILE`)
- add one switch for candidate mode:
    `--disagg-decode-queue-mode per_replica_queue`
- write outputs to `$EXP_ROOT/candidate_b/<workload_id>/`

Template command:

```bash
for item in "${WORKLOADS[@]}"; do
    read -r W NUM_PROMPTS RIN ROUT RR <<<"$item"
    OUT="$EXP_ROOT/candidate_b/${W}"
  mkdir -p "$OUT"

    "$PY" /home/cjia/projects/tair-kvcache/hisim/tools/aic_topn_to_hisim_sweep.py \
        --aic-csv "$AIC_CSV" \
        --top-n "$TOP_N" \
        --output-dir "$OUT" \
        --model-path "$MODEL" \
        --predictor-device-name "$PREDICTOR_DEVICE" \
        --backend-version "$BACKEND_VERSION" \
        --database-mode "$DB_MODE" \
        --cache-scenario no_cache \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$RIN" \
        --random-output-len "$ROUT" \
        --request-rate "$RR" \
        --synth-disagg-from-agg \
        --synth-disagg-profiles "$DISAGG_PROFILE" \
        --disagg-decode-queue-mode per_replica_queue
done
```

Important:

- keep same traces/arrival/token lengths
- keep same predictor inputs
- change only decode scheduling model

---

## 4. Build A/B Comparison Table

Create a single CSV for deltas:

```text
$EXP_ROOT/reports/ab_delta.csv
```

Required columns:

- `workload_id`
- `metric`
- `baseline_a`
- `candidate_b`
- `delta_abs` (`b - a`)
- `delta_rel` (`(b - a)/a`)

Primary metrics to include:

- `mean_e2e_latency_ms`
- `median_e2e_latency_ms` (acts as P50 E2E)
- `p95_e2e_latency_ms`
- `p99_e2e_latency_ms`

Secondary metrics:

- `mean_ttft_ms`, `p95_ttft_ms`, `p99_ttft_ms`
- `mean_itl_ms`, `p95_itl_ms`, `p99_itl_ms`
- throughput and stage metrics

---

## 5. Quick Aggregation Script (optional)

Use this minimal helper to aggregate selected metrics from two summary files.

```bash
cat > "$EXP_ROOT/reports/ab_extract.py" <<'PY'
import json
import sys
from pathlib import Path

if len(sys.argv) != 5:
    print("Usage: ab_extract.py <baseline_result_json> <candidate_result_json> <workload_id> <out_csv>")
    sys.exit(1)

base_json, cand_json, workload_id, out_csv = sys.argv[1:5]
metrics = [
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

base = load_json(base_json)
cand = load_json(cand_json)

write_header = not Path(out_csv).exists()
with open(out_csv, "a", newline="") as f:
    import csv
    w = csv.writer(f)
    if write_header:
        w.writerow(["workload_id", "metric", "baseline_a", "candidate_b", "delta_abs", "delta_rel"])
    for m in metrics:
        if m not in base or m not in cand:
            continue
        a = float(base[m])
        b = float(cand[m])
        d = b - a
        r = 0.0 if a == 0 else d / a
        w.writerow([workload_id, m, a, b, d, r])
PY
```

Invoke per workload:

```bash
for W in w_low w_mid w_high_decode w_bursty; do
    BASE_JSON=$(ls "$EXP_ROOT/baseline_a/$W"/top*/result.json | head -n1)
    CAND_JSON=$(ls "$EXP_ROOT/candidate_b/$W"/top*/result.json | head -n1)
  $PY "$EXP_ROOT/reports/ab_extract.py" \
        "$BASE_JSON" \
        "$CAND_JSON" \
    "$W" \
    "$EXP_ROOT/reports/ab_delta.csv"
done
```

---

## 6. Decision Summary Template

Create:

```text
$EXP_ROOT/reports/decision.md
```

Suggested structure:

1. setup and commits
2. workload matrix
3. headline deltas (E2E P50/P95/P99)
4. where Candidate B helps most
5. regressions/noise notes
6. recommendation:
   - default to B for decode-heavy what-if
   - or keep A/B selectable by scenario

---

## 7. Next Upgrade Hook

After this A/B, if needed, continue to advanced modeling:

- true prefill/decode overlap timeline
- more explicit concurrency constraints
- keep metrics schema backward-compatible

Track this phase against:

- [pd_decode_upgrade_plan_20260605.md](hisim/docs/pd_decode_upgrade_plan_20260605.md)
