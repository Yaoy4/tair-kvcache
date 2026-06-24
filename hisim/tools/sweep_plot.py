"""CLI: render a HiSim sweep summary CSV to a self-contained HTML report.

Usage:
    python -m hisim.tools.sweep_plot --summary-csv path/to/summary.csv \
        --output report.html [--title "..."]

This tool is intentionally thin — all logic lives in the
``hisim.visualization`` package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a HiSim sweep summary CSV to a self-contained HTML report.",
    )
    parser.add_argument(
        "--summary-csv",
        required=True,
        type=Path,
        help="Path to a sweep summary CSV (as produced by the sweep tool).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output HTML path. Parent directory is created if missing.",
    )
    parser.add_argument(
        "--title",
        default="HiSim Sweep Report",
        help="Title shown at the top of the report.",
    )
    parser.add_argument(
        "--no-enrich-result-file",
        action="store_true",
        help="Skip merging per-row result_file JSON (stage timings).",
    )
    parser.add_argument(
        "--no-enrich-sim-config",
        action="store_true",
        help="Skip merging per-row sim_config JSON (disagg topology).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.summary_csv.exists():
        print(f"error: summary CSV not found: {args.summary_csv}", file=sys.stderr)
        return 2

    # Imports are inside main so --help works without optional deps.
    from hisim.visualization import sweep_data, sweep_report

    df = sweep_data.load_sweep(
        args.summary_csv,
        enrich_with_result_files=not args.no_enrich_result_file,
        enrich_with_sim_configs=not args.no_enrich_sim_config,
    )
    df = sweep_data.add_derived_columns(df)

    out = sweep_report.render_report(
        df,
        args.output,
        title=args.title,
        source=str(args.summary_csv),
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
