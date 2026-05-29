#!/usr/bin/env bash
# End-to-end PD demo:  synthetic AIC CSV  →  bridge --emit-disagg
#                                          →  Hisim config JSON
#                                          →  pd_demo.py --config
#
# Run from tair-kvcache repo root:
#   bash hisim/tools/pd_demo_e2e.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/venv/bin/python"
export PYTHONPATH="$ROOT/hisim/src:$ROOT/../aiconfigurator/src"
export SGLANG_USE_CPU_ENGINE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

TMP="$(mktemp -d)"
CSV="$TMP/pareto.csv"
CFG="$TMP/hisim.json"

cat > "$CSV" <<'EOF'
(p)tp,(p)pp,(p)dp,(p)moe_ep,(p)gemm,(p)kvcache,(p)version,(p)system,(p)workers,(d)tp,(d)pp,(d)dp,(d)moe_ep,(d)gemm,(d)kvcache,(d)version,(d)system,(d)workers
4,1,1,1,fp16,fp16,0.5.6.post2,h100_sxm,2,2,1,2,1,fp16,fp16,0.5.6.post2,h20,4
EOF

echo "==> Step 1: synthetic AIC CSV"
cat "$CSV"
echo

echo "==> Step 2: run bridge --emit-disagg"
"$PY" "$ROOT/hisim/tools/aic_to_hisim_bridge.py" \
  --aic-csv "$CSV" --row-index 0 --output "$CFG" \
  --database-path /aic/db \
  --emit-disagg --kv-transfer-bw-gbps 50 --kv-transfer-latency-us 20

echo
echo "==> Step 3: generated Hisim JSON"
"$PY" -m json.tool "$CFG"
echo

echo "==> Step 4: run pd_demo.py against the generated JSON"
"$PY" "$ROOT/hisim/tools/pd_demo.py" --config "$CFG" --requests 6
