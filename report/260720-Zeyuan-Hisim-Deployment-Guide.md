# Hisim-Deployment-Guide

> 更新于：2026.7.20

- 整体结构：

  ```bash
  ~/project/
  ├── .venv/                                      # Python 虚拟环境
  ├── aiconfigurator-e0735cc/                     # AIC
  │   └── src/aiconfigurator/systems/             # AIC 读取的性能表
  └── tair-kvcache/
      └── hisim/                                  # HiSim
          └── tools/
              └── pd_disagg_rtx6000_sglang_0_5_10.json  # HiSim 配置文件
  ```

- clone需要的代码repo

  ```bash
  # Hisim
  git clone --branch Thjiang_Dev --single-branch https://github.com/Yaoy4/tair-kvcache.git
  # AIC（e0735cc）
  git clone https://github.com/ai-dynamo/aiconfigurator.git aiconfigurator-e0735cc
  cd aiconfigurator-e0735cc
  git checkout e0735ccca08c790b37c511a20f484a62a4127118
  ```

- 创建python虚拟环境

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- 安装Hisim依赖

  ```bash
  cd tair-kvcache/hisim
  pip install .
  ```

- 安装 SGLang

  > Hisim 目前目前对SGLang的支持停留在0.5.6.post2，此版本与性能表中使用的SGLang版本独立

  ```bash
  # 0.5.6.post2
  python3 -m pip install "sglang==0.5.6.post2"
  
  # CPU 版本
  python3 -m pip install "sglang[srt]==0.5.6.post2"
  ```

- “sim-config-path” 配置文件的路径问题修正：

  Hisim-config 文件中的“database_path”为硬编码，需要根据实际文件路径进行调整，见整体结构。需要将 database_path 指向 clone 到本地的 AIC 仓库下

  ```bash
  # 例如
  /mnt/nfs02/users/XXX/project/aiconfigurator-e0735cc/src/aiconfigurator/systems
  ```

- 测试指令（注意修改路径）：

  ```bash
  # ~/project/
  source .venv/bin/activate
  
  export HF_ENDPOINT=https://hf-mirror.com
  
  # server
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  python3 -m hisim.simulation.sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$HISIM_PORT" \
    --sim-config-path tair-kvcache/hisim/tools/pd_disagg_rtx6000_sglang_0_5_10.json \
    --skip-server-warmup \
    # --attention-backend triton
  ```

  ```bash
  # ~/project/
  source .venv/bin/activate
  
  export HF_ENDPOINT=https://hf-mirror.com
  
  # client
  export PYTHONPATH="$HOME/project/tair-kvcache/hisim/src:$HOME/project/aiconfigurator-e0735cc/src"
  export SGLANG_USE_CPU_ENGINE=1
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  export HISIM_OUTPUT_DIR="$HOME/project/hisim/output"
  export MODEL_PATH="Qwen/Qwen3-8B"
  export HISIM_PORT=30003
  
  mkdir -p "$HOME/project/cases_same_seed_flush_cache_32kv_GBps"
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
          --output-file "$HOME/project/cases_same_seed_flush_cache_32kv_GBps/metrics_rr${rr}_il${il}_ol${ol}.jsonl"
  
        case_dir="$HOME/project/cases_same_seed_flush_cache_32kv_GBps/rr_${rr}_il_${il}_ol_${ol}"
        rm -rf "$case_dir"
        if [[ -d "$HISIM_OUTPUT_DIR" ]]; then
          mv "$HISIM_OUTPUT_DIR" "$case_dir"
        fi
        mkdir -p "$HISIM_OUTPUT_DIR"
      done
    done
  done
  ```

