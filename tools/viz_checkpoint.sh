#!/usr/bin/env bash
# Checkpoint helper for HiSim sweep visualization sub-slices.
# Usage: tools/viz_checkpoint.sh <slice-id> "<short message>"
#   e.g. tools/viz_checkpoint.sh v0 "scaffold + viz extras + import smoke test"
#
# Behavior:
#   - Stages viz-related paths only.
#   - Commits with a conventional message.
#   - Creates annotated tag viz-phase-<slice-id>.
#   - Refuses to run if the tag already exists or nothing is staged.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <slice-id> \"<short message>\"" >&2
  exit 2
fi

slice_id="$1"
message="$2"
tag="viz-phase-${slice_id}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  echo "error: tag ${tag} already exists. Use a new slice id or delete the tag." >&2
  exit 1
fi

# Viz-related paths (extend as later phases add files).
paths=(
  "hisim/pyproject.toml"
  "hisim/src/hisim/visualization"
  "hisim/tools/sweep_plot.py"
  "hisim/tools/sweep_dashboard.py"
  "hisim/tests/unit/visualization"
  "hisim/tests/integration/test_sweep_plot_cli.py"
  "hisim/tests/integration/test_sweep_tool_plot_wiring.py"
  "hisim/tools/aic_topn_to_hisim_sweep.py"
)

existing=()
for p in "${paths[@]}"; do
  [[ -e "$p" ]] && existing+=("$p")
done

if [[ ${#existing[@]} -eq 0 ]]; then
  echo "error: none of the viz paths exist yet. Nothing to stage." >&2
  exit 1
fi

git add -- "${existing[@]}"

if git diff --cached --quiet; then
  echo "error: no staged changes under viz paths. Nothing to commit." >&2
  exit 1
fi

git commit -m "hisim(viz): phase ${slice_id} — ${message}"
git tag -a "${tag}" -m "${message}"

echo
echo "checkpoint created: ${tag}"
git --no-pager log -1 --oneline --decorate
