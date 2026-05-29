"""V4 integration tests for sweep tool plot wiring.

These tests exercise the ``--no-plot`` flag and verify the auto-render
behavior is correctly gated. They do NOT spin up real servers; they
rely on ``--dry-run`` mode for safety.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_HISIM_SRC = _REPO_ROOT / "hisim" / "src"
_AIC_SRC = _REPO_ROOT.parent / "aiconfigurator" / "src"
_TOOL = _REPO_ROOT / "hisim" / "tools" / "aic_topn_to_hisim_sweep.py"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_HISIM_SRC), str(_AIC_SRC), existing]))
    env["SGLANG_USE_CPU_ENGINE"] = "1"
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    return env


def _write_aic_csv(path: Path) -> None:
    # Minimal AIC CSV — only enough columns for _read_rows / _pick_row_value.
    path.write_text(textwrap.dedent("""\
        rank,system,tp,dp,moe_ep
        1,H20,2,1,1
        2,H20,4,1,1
    """))


def _run_tool(extra_args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    aic_csv = tmp_path / "aic.csv"
    _write_aic_csv(aic_csv)
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(_TOOL),
        "--aic-csv", str(aic_csv),
        "--top-n", "1",
        "--output-dir", str(out_dir),
        "--model-path", "/tmp/fake-model",
        "--dry-run",
        *extra_args,
    ]
    return subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=120)


@pytest.mark.integration
def test_dry_run_does_not_render_report(tmp_path):
    """--dry-run skips the plot block even without --no-plot."""
    proc = _run_tool([], tmp_path)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    out_dir = tmp_path / "out"
    assert (out_dir / "summary.csv").exists()
    assert not (out_dir / "sweep_report.html").exists()


@pytest.mark.integration
def test_no_plot_flag_parses(tmp_path):
    """--no-plot is accepted by the CLI."""
    proc = _run_tool(["--no-plot"], tmp_path)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    out_dir = tmp_path / "out"
    assert not (out_dir / "sweep_report.html").exists()


@pytest.mark.integration
def test_help_does_not_crash():
    """--help mentions the new --no-plot flag."""
    proc = subprocess.run(
        [sys.executable, str(_TOOL), "--help"],
        env=_env(), capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert "--no-plot" in proc.stdout
    assert "--plot-output" in proc.stdout
