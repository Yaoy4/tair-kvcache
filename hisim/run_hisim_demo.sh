#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python}"
AICONFIG_ROOT="${AICONFIG_ROOT:-/home/cjia/projects/aiconfigurator}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
SIM_CONFIG_PATH="${SIM_CONFIG_PATH:-test/assets/mock/config.json}"
REQUEST_RATE="${REQUEST_RATE:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-256}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${AICONFIG_ROOT}/src" ]]; then
  echo "aiconfigurator source path not found: ${AICONFIG_ROOT}/src" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/src:${AICONFIG_ROOT}/src"
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

SERVER_LOG="${SCRIPT_DIR}/.hisim_server.log"
rm -f "${SERVER_LOG}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[demo] starting server..."
"${PYTHON_BIN}" -m hisim.simulation.sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --sim-config-path "${SIM_CONFIG_PATH}" \
  --skip-server-warmup >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[demo] waiting for server on 127.0.0.1:30000..."
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[demo] server exited early, tail logs:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi

  if (echo >/dev/tcp/127.0.0.1/30000) >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! (echo >/dev/tcp/127.0.0.1/30000) >/dev/null 2>&1; then
  echo "[demo] server did not become ready in time, tail logs:" >&2
  tail -n 80 "${SERVER_LOG}" >&2 || true
  exit 1
fi

echo "[demo] running benchmark..."
"${PYTHON_BIN}" -m hisim.simulation.bench_serving \
  --warmup-requests 0 \
  --model "${MODEL_PATH}" \
  --bench-mode simulation \
  --dataset-name random \
  --request-rate "${REQUEST_RATE}" \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --random-output-len "${RANDOM_OUTPUT_LEN}" \
  --random-range-ratio 1 \
  --num-prompts "${NUM_PROMPTS}"

echo "[demo] done"