#!/usr/bin/env python3
"""Run top-N AIC rows through Hisim simulation.

Pipeline:
1) Read AIConfigurator CSV rows.
2) Convert each selected row into a Hisim config with aic_to_hisim_bridge.py.
3) Launch Hisim server and run bench_serving.
4) Save per-run logs/results and aggregate a summary CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows


def _wait_for_port(host: str, port: int, timeout_s: int, proc: subprocess.Popen[Any]) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1)
    return False


def _stop_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _cache_flags(cache_scenario: str) -> list[str]:
    if cache_scenario == "no_cache":
        return ["--disable-radix-cache", "--disable-chunked-prefix-cache"]
    if cache_scenario == "l2":
        return [
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2.0",
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "kernel",
            "--hicache-storage-backend",
            "file",
        ]
    return []


def _pick_row_value(row: dict[str, str], agg_key: str, disagg_key: str) -> str:
    if agg_key in row and str(row.get(agg_key, "")).strip():
        return str(row[agg_key]).strip()
    if disagg_key in row and str(row.get(disagg_key, "")).strip():
        return str(row[disagg_key]).strip()
    return ""


def _parse_args() -> argparse.Namespace:
    hisim_root = Path(__file__).resolve().parents[1]
    projects_root = Path(__file__).resolve().parents[3]
    default_aic_root = projects_root / "aiconfigurator"

    parser = argparse.ArgumentParser(description="Top-N AIC -> Hisim sweep runner")
    parser.add_argument("--aic-csv", required=True, help="Path to AIC CSV file")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top rows to run")
    parser.add_argument("--row-offset", type=int, default=0, help="Start row index in CSV")
    parser.add_argument("--output-dir", default=None, help="Output directory for all run artifacts")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable for server/bench/bridge commands",
    )

    parser.add_argument("--model-path", required=True, help="Model path for Hisim server and bench")
    parser.add_argument(
        "--base-config",
        default=str(hisim_root / "test/assets/mock/config.json"),
        help="Base Hisim config template",
    )
    parser.add_argument(
        "--bridge-script",
        default=str(hisim_root / "tools/aic_to_hisim_bridge.py"),
        help="Bridge script path",
    )

    parser.add_argument(
        "--aiconfig-root",
        default=str(default_aic_root),
        help="AIConfigurator repo root path",
    )
    parser.add_argument(
        "--database-path",
        default=None,
        help="AIConfigurator systems path. Defaults to <aiconfig-root>/src/aiconfigurator/systems",
    )
    parser.add_argument("--database-mode", default="SILICON")
    parser.add_argument("--predictor-device-name", default=None)
    parser.add_argument("--backend-version", default=None)
    parser.add_argument("--disagg-role", choices=["decode", "prefill"], default="decode")

    parser.add_argument("--prefill-scale-factor", type=float, default=None)
    parser.add_argument("--decode-scale-factor", type=float, default=None)
    parser.add_argument("--xgb-model-path", default=None)
    parser.add_argument("--data-type", default=None)
    parser.add_argument("--kv-cache-data-type", default=None)

    parser.add_argument("--platform-accelerator", default="H20")
    parser.add_argument("--disk-read-bandwidth-gb", type=float, default=8)
    parser.add_argument("--disk-write-bandwidth-gb", type=float, default=8)
    parser.add_argument("--memory-read-bandwidth-gb", type=float, default=16)
    parser.add_argument("--memory-write-bandwidth-gb", type=float, default=16)
    parser.add_argument("--num-device-per-node", type=int, default=8)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-ready-timeout", type=int, default=120)
    parser.add_argument("--cache-scenario", choices=["no_cache", "l1", "l2"], default="l1")

    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--random-input-len", type=int, default=512)
    parser.add_argument("--random-output-len", type=int, default=256)
    parser.add_argument("--random-range-ratio", type=float, default=1.0)

    parser.add_argument("--dry-run", action="store_true", help="Generate commands/configs only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    hisim_root = Path(__file__).resolve().parents[1]

    csv_path = Path(args.aic_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"AIC CSV not found: {csv_path}")

    rows = _read_rows(csv_path)
    start = max(0, args.row_offset)
    end = min(len(rows), start + max(0, args.top_n))
    if start >= len(rows) or start == end:
        raise ValueError(f"No rows selected: row-offset={start}, top-n={args.top_n}, total={len(rows)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (hisim_root / f"output/aic_bridge_sweep_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    aic_root = Path(args.aiconfig_root)
    database_path = (
        Path(args.database_path)
        if args.database_path is not None
        else aic_root / "src/aiconfigurator/systems"
    )

    env = os.environ.copy()
    env["SGLANG_USE_CPU_ENGINE"] = "1"
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    py_path_parts = [str(hisim_root / "src"), str(aic_root / "src")]
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_path_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

    summary_rows: list[dict[str, Any]] = []
    bridge_script = Path(args.bridge_script)

    for row_idx in range(start, end):
        run_rank = row_idx - start + 1
        run_dir = output_dir / f"top{run_rank:02d}_row{row_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)

        sim_config_path = run_dir / "sim_config.json"
        result_path = run_dir / "result.json"
        server_log = run_dir / "server.log"
        bench_log = run_dir / "bench.log"

        bridge_cmd = [
            args.python_bin,
            str(bridge_script),
            "--aic-csv",
            str(csv_path),
            "--row-index",
            str(row_idx),
            "--output",
            str(sim_config_path),
            "--base-config",
            str(args.base_config),
            "--disagg-role",
            args.disagg_role,
            "--database-path",
            str(database_path),
            "--database-mode",
            args.database_mode,
            "--platform-accelerator",
            args.platform_accelerator,
            "--disk-read-bandwidth-gb",
            str(args.disk_read_bandwidth_gb),
            "--disk-write-bandwidth-gb",
            str(args.disk_write_bandwidth_gb),
            "--memory-read-bandwidth-gb",
            str(args.memory_read_bandwidth_gb),
            "--memory-write-bandwidth-gb",
            str(args.memory_write_bandwidth_gb),
            "--num-device-per-node",
            str(args.num_device_per_node),
        ]
        if args.predictor_device_name:
            bridge_cmd.extend(["--predictor-device-name", args.predictor_device_name])
        if args.backend_version:
            bridge_cmd.extend(["--backend-version", args.backend_version])
        if args.prefill_scale_factor is not None:
            bridge_cmd.extend(["--prefill-scale-factor", str(args.prefill_scale_factor)])
        if args.decode_scale_factor is not None:
            bridge_cmd.extend(["--decode-scale-factor", str(args.decode_scale_factor)])
        if args.xgb_model_path is not None:
            bridge_cmd.extend(["--xgb-model-path", args.xgb_model_path])
        if args.data_type is not None:
            bridge_cmd.extend(["--data-type", args.data_type])
        if args.kv_cache_data_type is not None:
            bridge_cmd.extend(["--kv-cache-data-type", args.kv_cache_data_type])

        subprocess.run(bridge_cmd, check=True, env=env)

        server_cmd = [
            args.python_bin,
            "-m",
            "hisim.simulation.sglang.launch_server",
            "--model-path",
            args.model_path,
            "--sim-config-path",
            str(sim_config_path),
            "--skip-server-warmup",
        ]
        server_cmd.extend(_cache_flags(args.cache_scenario))

        bench_cmd = [
            args.python_bin,
            "-m",
            "hisim.simulation.bench_serving",
            "--host",
            args.host,
            "--warmup-requests",
            "0",
            "--bench-mode",
            "simulation",
            "--backend",
            "sglang",
            "--model",
            args.model_path,
            "--dataset-name",
            "random",
            "--request-rate",
            str(args.request_rate),
            "--random-input-len",
            str(args.random_input_len),
            "--random-output-len",
            str(args.random_output_len),
            "--random-range-ratio",
            str(args.random_range_ratio),
            "--num-prompts",
            str(args.num_prompts),
            "--output-file",
            str(result_path),
        ]

        if args.dry_run:
            summary_rows.append(
                {
                    "row_index": row_idx,
                    "run_rank": run_rank,
                    "status": "dry_run",
                    "sim_config": str(sim_config_path),
                    "result_file": str(result_path),
                    "server_cmd": " ".join(server_cmd),
                    "bench_cmd": " ".join(bench_cmd),
                }
            )
            continue

        server_proc: subprocess.Popen[Any] | None = None
        bench_rc = -1
        status = "unknown"
        error = ""
        metrics: dict[str, Any] = {}
        started_at = time.time()

        try:
            with server_log.open("w", encoding="utf-8") as slog:
                server_proc = subprocess.Popen(
                    server_cmd,
                    stdout=slog,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(hisim_root),
                )

            ready = _wait_for_port(args.host, args.port, args.server_ready_timeout, server_proc)
            if not ready:
                status = "server_not_ready"
                error = f"Server not ready on {args.host}:{args.port}"
            else:
                with bench_log.open("w", encoding="utf-8") as blog:
                    bench_rc = subprocess.run(
                        bench_cmd,
                        stdout=blog,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=str(hisim_root),
                        check=False,
                    ).returncode
                if bench_rc != 0:
                    status = "bench_failed"
                    error = f"bench_serving exited with rc={bench_rc}"
                else:
                    status = "ok"
                    if result_path.exists():
                        with result_path.open("r", encoding="utf-8") as rf:
                            result = json.load(rf)
                        metrics = {
                            "request_throughput": result.get("request_throughput"),
                            "output_throughput": result.get("output_throughput"),
                            "mean_ttft_ms": result.get("mean_ttft_ms"),
                            "mean_tpot_ms": result.get("mean_tpot_ms"),
                            "p99_e2e_latency_ms": result.get("p99_e2e_latency_ms"),
                            "prefix_cache_reused_ratio": result.get("prefix_cache_reused_ratio"),
                            "disk_prefetch_ratio": result.get("disk_prefetch_ratio"),
                        }
        finally:
            _stop_process(server_proc)

        elapsed_s = time.time() - started_at
        row = rows[row_idx]
        summary_rows.append(
            {
                "row_index": row_idx,
                "run_rank": run_rank,
                "status": status,
                "error": error,
                "elapsed_s": round(elapsed_s, 3),
                "cache_scenario": args.cache_scenario,
                "request_rate": args.request_rate,
                "num_prompts": args.num_prompts,
                "aic_tp": _pick_row_value(row, "tp", f"({args.disagg_role[0]})tp"),
                "aic_dp": _pick_row_value(row, "dp", f"({args.disagg_role[0]})dp"),
                "aic_ep": _pick_row_value(row, "moe_ep", f"({args.disagg_role[0]})moe_ep"),
                "aic_system": _pick_row_value(row, "system", f"({args.disagg_role[0]})system"),
                "sim_config": str(sim_config_path),
                "server_log": str(server_log),
                "bench_log": str(bench_log),
                "result_file": str(result_path),
                **metrics,
            }
        )

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
        f.write("\n")

    if summary_rows:
        fieldnames: list[str] = []
        for item in summary_rows:
            for key in item.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    print(f"[sweep] output_dir={output_dir}")
    print(f"[sweep] rows_selected={len(summary_rows)} ok={ok_count}")
    print(f"[sweep] summary_csv={summary_csv}")
    print(f"[sweep] summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
