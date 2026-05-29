#!/usr/bin/env bash
# Checkpoint helper for PD disagg sub-slices.
# Usage: tools/pd_checkpoint.sh <slice-id> "<short message>"
#   e.g. tools/pd_checkpoint.sh 0b "TransferModel interface + unit tests"
#
# Behavior:
#   - Stages PD-related paths only.
#   - Commits with a conventional message.
#   - Creates annotated tag pd-phase-<slice-id>.
#   - Refuses to run if the tag already exists or nothing is staged.
#
# Revert examples:
#   git reset --hard pd-phase-0a            # full revert
#   git checkout pd-phase-0a -- <path>      # partial revert
#   git diff pd-phase-0a..HEAD              # inspect a slice

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <slice-id> \"<short message>\"" >&2
  exit 2
fi

slice_id="$1"
message="$2"
tag="pd-phase-${slice_id}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  echo "error: tag ${tag} already exists. Use a new slice id or delete the tag." >&2
  exit 1
fi

# PD-related paths (extend as later phases add files).
paths=(
  "hisim/src/hisim/simulation/pd_types.py"
  "hisim/src/hisim/simulation/pd_controller.py"
  "hisim/src/hisim/simulation/pd_transfer.py"
  "hisim/src/hisim/simulation/pd_config.py"
  "hisim/src/hisim/simulation/pd_factory.py"
  "hisim/src/hisim/simulation/pd_backend_a.py"
  "hisim/src/hisim/simulation/pd_backend_b.py"
  "hisim/src/hisim/simulation/pd_backend_protocol.py"
  "hisim/src/hisim/simulation/pd_aic_adapter.py"
  "hisim/src/hisim/simulation/pd_aic_factory.py"
  "hisim/src/hisim/simulation/pd_runtime.py"
  "hisim/src/hisim/simulation/pd_metrics.py"
  "hisim/src/hisim/simulation/pd_ab_harness.py"
  "hisim/src/hisim/simulation/manager/config.py"
  "hisim/src/hisim/simulation/types.py"
  "hisim/tools/pd_demo.py"
  "hisim/tools/pd_demo_e2e.sh"
  "hisim/src/hisim/simulation/sim_args.py"
  "hisim/src/hisim/simulation/utils.py"
  "hisim/src/hisim/simulation/manager/config.py"
  "hisim/src/hisim/simulation/sglang/sglang_hook.py"
  "hisim/src/hisim/simulation/sglang/pd_backend_single.py"
  "hisim/src/hisim/simulation/sglang/pd_backend_two_process.py"
  "hisim/tests/unit/simulation"
  "hisim/tools/aic_to_hisim_bridge.py"
  "hisim/tools/aic_topn_to_hisim_sweep.py"
  "hisim/docs/pd_ab_equivalence_spec.md"
)

existing=()
for p in "${paths[@]}"; do
  [[ -e "$p" ]] && existing+=("$p")
done

if [[ ${#existing[@]} -eq 0 ]]; then
  echo "error: none of the tracked PD paths exist yet. Nothing to stage." >&2
  exit 1
fi

git add -- "${existing[@]}"

if git diff --cached --quiet; then
  echo "error: no staged changes under PD paths. Nothing to commit." >&2
  exit 1
fi

commit_msg="hisim(pd): phase ${slice_id} — ${message}"
git commit -m "${commit_msg}"
git tag -a "${tag}" -m "PD disagg sub-slice ${slice_id} checkpoint"

echo
echo "checkpoint created: ${tag}"
git --no-pager log --oneline -1
