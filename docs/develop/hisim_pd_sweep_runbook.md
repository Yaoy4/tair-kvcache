# HiSim PD Sweep Runbook (same seed + flush cache)

This document explains how to run a reproducible HiSim benchmark sweep for PD-disagg with sglang 0.5.10.

## Prerequisites

- Repo: `/mnt/nfs02/users/yaoyao1/projects/tair-kvcache`
- Model path: `$HOME/projects/models/Qwen3-8B`
- Aiconfigurator source: `$HOME/projects/aiconfigurator/src`
- Python virtual env exists at `venv/`
- Config file exists: `hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json`

## Terminal 1: launch simulation server

```bash
cd /mnt/nfs02/users/yaoyao1/projects/tair-kvcache
source venv/bin/activate

export PYTHONPATH="$HOME/projects/tair-kvcache/hisim/src:$HOME/projects/aiconfigurator/src"
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HISIM_OUTPUT_DIR="$HOME/projects/hisim/output"
export MODEL_PATH="$HOME/projects/models/Qwen3-8B"
export HISIM_PORT=30003

python3 -m hisim.simulation.sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$HISIM_PORT" \
  --sim-config-path hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
  --skip-server-warmup
```

Keep this terminal running.

## Terminal 2: run sweep

```bash
cd /mnt/nfs02/users/yaoyao1/projects/tair-kvcache
source venv/bin/activate

export PYTHONPATH="$HOME/projects/tair-kvcache/hisim/src:$HOME/projects/aiconfigurator/src"
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HISIM_OUTPUT_DIR="$HOME/projects/hisim/output"
export MODEL_PATH="$HOME/projects/models/Qwen3-8B"
export HISIM_PORT=30003

mkdir -p "$HOME/projects/hisim/cases_same_seed_flush_cache_32kv_GBps"
mkdir -p "$HISIM_OUTPUT_DIR"

for rr in 1 2 4 8 12; do
  for il in 256 1024 2048; do
    for ol in 128 512; do
      echo "RUN rr=$rr il=$il ol=$ol seed=1"

      python3 -m hisim.simulation.bench_serving \
        --bench-mode simulation \
        --backend sglang \
        --host 127.0.0.1 \
        --port "$HISIM_PORT" \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --num-prompts 200 \
        --warmup-requests 0 \
        --request-rate "$rr" \
        --random-input-len "$il" \
        --random-output-len "$ol" \
        --random-range-ratio 0 \
        --seed 1 \
        --flush-cache \
        --output-file "$HOME/projects/hisim/cases_same_seed_flush_cache_32kv_GBps/metrics_rr${rr}_il${il}_ol${ol}.jsonl"

      case_dir="$HOME/projects/hisim/cases_same_seed_flush_cache_32kv_GBps/rr_${rr}_il_${il}_ol_${ol}"
      rm -rf "$case_dir"
      if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
        mv "$HISIM_OUTPUT_DIR" "$case_dir"
      fi
      mkdir -p "$HISIM_OUTPUT_DIR"
    done
  done
done
```

## Outputs

- Summary files:
  - `$HOME/projects/hisim/cases_same_seed_flush_cache_32kv_GBps/metrics_rr*_il*_ol*.jsonl`
- Per-case server output directories:
  - `$HOME/projects/hisim/cases_same_seed_flush_cache_32kv_GBps/rr_<rr>_il_<il>_ol_<ol>/`

## Notes

- If `/flush_cache` is slow on your server, use `--flush-cache-timeout <seconds>` in the benchmark command.
- If a run is interrupted, you can re-run only selected `(rr, il, ol)` combinations by editing the loop ranges.
