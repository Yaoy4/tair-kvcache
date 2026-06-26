#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HISIM_ROOT="$ROOT/hisim"
DEFAULT_PY="$HISIM_ROOT/hisim/bin/python3"
PY="${HISIM_PYTHON:-$DEFAULT_PY}"

if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

if [[ -z "${PY:-}" || ! -x "$PY" ]]; then
  echo "error: no usable python3 interpreter found" >&2
  exit 1
fi

export PYTHONPATH="$HISIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Some tests spawn `python` explicitly. Provide a stable alias even on hosts
# that only expose `python3`.
TMP_BIN="$(mktemp -d)"
trap 'rm -rf "$TMP_BIN"' EXIT
ln -sf "$PY" "$TMP_BIN/python"
export PATH="$TMP_BIN:$PATH"

echo "==> Using Python: $PY"
echo "==> 1/4 Syntax check"
"$PY" -m compileall \
  "$HISIM_ROOT/src/hisim/simulation/pd_controller.py" \
  "$HISIM_ROOT/src/hisim/simulation/pd_runtime.py" \
  "$HISIM_ROOT/src/hisim/simulation/sglang/sglang_hook.py" >/dev/null

echo "==> 2/4 Config JSON check"
"$PY" -m json.tool "$HISIM_ROOT/tools/pd_disagg_rtx6000_sglang_0_5_10.json" >/dev/null

echo "==> 3/4 Targeted PD regression tests"
cd "$HISIM_ROOT"
"$PY" -m pytest \
  tests/unit/simulation/test_pd_controller.py \
  tests/unit/simulation/test_pd_hook_glue.py \
  tests/unit/simulation/test_pd_hook_glue_decode.py \
  -q

echo "==> 4/4 Full simulation unit suite"
"$PY" -m pytest tests/unit/simulation -q

echo "PD disaggregation validation passed."
