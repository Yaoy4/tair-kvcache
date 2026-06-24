#!/usr/bin/env bash
# Phase 6e — CI gate for PD A/B equivalence across W1-W6.
# Exits non-zero on any blocking (STRUCT/PER_REQ/AGGREGATE) classification.
#
# Usage:
#   tools/pd_ab_check.sh                # run all workloads
#   tools/pd_ab_check.sh --verbose      # dump first mismatches
#   tools/pd_ab_check.sh --workload W3-concurrent
#
# Env it sets (idempotent):
#   PYTHONPATH includes hisim/src + aiconfigurator/src
#   SGLANG_USE_CPU_ENGINE=1
#   FLASHINFER_DISABLE_VERSION_CHECK=1

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# Prefer the project's venv if present, otherwise system python.
PY="${PY:-${repo_root}/venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

aic_src="${repo_root%/tair-kvcache}/aiconfigurator/src"
export PYTHONPATH="${repo_root}/hisim/src:${aic_src}:${PYTHONPATH:-}"
export SGLANG_USE_CPU_ENGINE="${SGLANG_USE_CPU_ENGINE:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

exec "$PY" -m hisim.simulation.pd_ab_check "$@"
