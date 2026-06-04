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


DTYPE_MAP = {
    "float16": "FP16",
    "bfloat16": "BF16",
    "fp8": "FP8",
    "fp8_block": "FP8",
    "int8": "INT8",
    "int8_wo": "INT8",
    "nvfp4": "FP4",
    "int4_wo": "INT4",
}


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


def _normalize_dtype(value: str | None, default: str | None = None) -> str | None:
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    upper_raw = raw.upper()
    if upper_raw in {"FP16", "BF16", "FP8", "INT8", "FP4", "INT4"}:
        return upper_raw
    return DTYPE_MAP.get(raw.lower(), default)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def _is_disagg_row(row: dict[str, str]) -> bool:
    return "(p)tp" in row and "(d)tp" in row


def _parse_synth_disagg_profiles(raw: str) -> list[tuple[int, int]]:
    profiles: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --synth-disagg-profiles entry '{item}'. Expected 'prefill:decode'."
            )
        prefill_raw, decode_raw = item.split(":", 1)
        prefill_replicas = _to_int(prefill_raw, -1)
        decode_replicas = _to_int(decode_raw, -1)
        if prefill_replicas <= 0 or decode_replicas <= 0:
            raise ValueError(
                f"Invalid --synth-disagg-profiles entry '{item}'. Replica counts must be > 0."
            )
        pair = (prefill_replicas, decode_replicas)
        if pair not in seen:
            seen.add(pair)
            profiles.append(pair)
    if not profiles:
        raise ValueError("--synth-disagg-profiles produced no valid replica pairs.")
    return profiles


def _build_synth_disagg_block(
    row: dict[str, str],
    args: argparse.Namespace,
    *,
    database_path: Path,
    prefill_replicas: int,
    decode_replicas: int,
) -> dict[str, Any]:
    device_name = args.predictor_device_name or _pick_row_value(row, "system", "(d)system")
    if not device_name:
        import warnings
        device_name = "h100_sxm"
        warnings.warn(
            "Synth disagg block: no `system` column in AIC row and no"
            f" --predictor-device-name provided; defaulting to {device_name!r}."
            " Pass --predictor-device-name <aic_device_name> to override.",
            stacklevel=2,
        )
    tp_size = _to_int(row.get("tp"), 1)
    ep_size = _to_int(row.get("moe_ep"), 1)
    dp_size = _to_int(row.get("dp"), 1)
    pp_size = _to_int(row.get("pp"), 1)

    data_type = _normalize_dtype(args.data_type, _normalize_dtype(row.get("gemm")))
    kv_cache_data_type = _normalize_dtype(
        args.kv_cache_data_type,
        _normalize_dtype(row.get("kvcache"), data_type),
    )
    backend_version = args.backend_version or row.get("version")

    row_moe_tp = _to_int(row.get("moe_tp"), tp_size)
    width = tp_size * dp_size

    synth_moe_tp = getattr(args, "synth_moe_tp_size", None)
    synth_moe_ep = getattr(args, "synth_ep_size", None)
    if synth_moe_tp is None and synth_moe_ep is None:
        moe_tp_size = row_moe_tp
        role_ep_size = ep_size
    elif synth_moe_tp is not None and synth_moe_ep is not None:
        if synth_moe_tp <= 0 or synth_moe_ep <= 0:
            raise ValueError("--synth-moe-tp-size/--synth-ep-size must be > 0")
        if synth_moe_tp * synth_moe_ep != width:
            raise ValueError(
                f"Invalid synthetic MoE parallelism: {synth_moe_tp}*{synth_moe_ep}!={width} (tp*dp)"
            )
        moe_tp_size = synth_moe_tp
        role_ep_size = synth_moe_ep
    elif synth_moe_tp is not None:
        if synth_moe_tp <= 0 or width % synth_moe_tp != 0:
            raise ValueError(
                f"--synth-moe-tp-size={synth_moe_tp} is invalid for tp*dp={width}"
            )
        moe_tp_size = synth_moe_tp
        role_ep_size = width // synth_moe_tp
    else:
        if synth_moe_ep is None or synth_moe_ep <= 0 or width % synth_moe_ep != 0:
            raise ValueError(
                f"--synth-ep-size={synth_moe_ep} is invalid for tp*dp={width}"
            )
        role_ep_size = synth_moe_ep
        moe_tp_size = width // synth_moe_ep

    def _role(replicas: int) -> dict[str, Any]:
        role: dict[str, Any] = {
            "device_name": device_name,
            "tp_size": tp_size,
            "ep_size": role_ep_size,
            "dp_size": dp_size,
            "pp_size": pp_size,
            "replicas": replicas,
            "max_running_per_replica": args.synth_max_running_per_replica,
            "database_path": str(database_path),
        }
        if data_type is not None:
            role["data_type"] = data_type
        if kv_cache_data_type is not None:
            role["kv_cache_data_type"] = kv_cache_data_type
        if backend_version is not None and str(backend_version).strip():
            role["backend_version"] = str(backend_version).strip()
        return role

    return {
        "enabled": True,
        "backend": args.disagg_backend,
        "prefill": _role(prefill_replicas),
        "decode": _role(decode_replicas),
        "_resolved_moe_tp_size": moe_tp_size,
        "kv_transfer": {
            "bw_gbps": float(args.kv_transfer_bw_gbps),
            "latency_us": float(args.kv_transfer_latency_us),
        },
    }


def _build_bridge_cmd(
    args: argparse.Namespace,
    *,
    bridge_script: Path,
    csv_path: Path,
    row_idx: int,
    sim_config_path: Path,
    database_path: Path,
    emit_disagg: bool,
) -> list[str]:
    """Build the aic_to_hisim_bridge subprocess argv for a single run.

    Phase 3c: when --emit-disagg is set on the sweep, propagate the
    disagg JSON-emission flags (--emit-disagg, --kv-transfer-bw-gbps,
    --kv-transfer-latency-us, --disagg-backend) so the produced sim
    config carries a `disagg` block consumable by Backend A.
    """
    cmd = [
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
        cmd.extend(["--predictor-device-name", args.predictor_device_name])
    if args.backend_version:
        cmd.extend(["--backend-version", args.backend_version])
    if args.prefill_scale_factor is not None:
        cmd.extend(["--prefill-scale-factor", str(args.prefill_scale_factor)])
    if args.decode_scale_factor is not None:
        cmd.extend(["--decode-scale-factor", str(args.decode_scale_factor)])
    if args.xgb_model_path is not None:
        cmd.extend(["--xgb-model-path", args.xgb_model_path])
    if args.data_type is not None:
        cmd.extend(["--data-type", args.data_type])
    if args.kv_cache_data_type is not None:
        cmd.extend(["--kv-cache-data-type", args.kv_cache_data_type])
    if emit_disagg:
        cmd.extend(
            [
                "--emit-disagg",
                "--kv-transfer-bw-gbps",
                str(args.kv_transfer_bw_gbps),
                "--kv-transfer-latency-us",
                str(args.kv_transfer_latency_us),
                "--disagg-backend",
                args.disagg_backend,
            ]
        )
    return cmd


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

    # Phase 3c: optional disagg JSON emission propagated to the bridge.
    parser.add_argument(
        "--emit-disagg",
        action="store_true",
        help="Ask the bridge to emit a DisaggConfig block in each sim config.",
    )
    parser.add_argument(
        "--kv-transfer-bw-gbps",
        type=float,
        default=200.0,
        help="KV transfer bandwidth (GB/s) used when --emit-disagg is set.",
    )
    parser.add_argument(
        "--kv-transfer-latency-us",
        type=float,
        default=10.0,
        help="KV transfer fixed latency (microseconds) used when --emit-disagg is set.",
    )
    parser.add_argument(
        "--disagg-backend",
        choices=["single_process", "two_process"],
        default="single_process",
        help="DisaggConfig.backend value propagated to the bridge.",
    )
    parser.add_argument(
        "--synth-disagg-from-agg",
        action="store_true",
        help=(
            "When selected AIC rows are agg-only, synthesize additional disagg runs "
            "from each row so agg vs disagg can still be compared in HiSim."
        ),
    )
    parser.add_argument(
        "--synth-disagg-profiles",
        default="2:6,4:4,6:2",
        help=(
            "Replica splits for synthetic disagg runs in 'prefill:decode' format, "
            "comma-separated (for example: 2:6,4:4,6:2)."
        ),
    )
    parser.add_argument(
        "--synth-max-running-per-replica",
        type=int,
        default=8,
        help="max_running_per_replica for each synthetic disagg role.",
    )
    parser.add_argument(
        "--synth-moe-tp-size",
        type=int,
        default=None,
        help=(
            "Optional synthetic MoE tp-size override for agg rows. If set with "
            "--synth-ep-size, product must equal tp*dp."
        ),
    )
    parser.add_argument(
        "--synth-ep-size",
        type=int,
        default=None,
        help=(
            "Optional synthetic MoE ep-size override for agg rows. If set with "
            "--synth-moe-tp-size, product must equal tp*dp."
        ),
    )

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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip rendering the HTML sweep report after summary.csv is written.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="HTML report path. Defaults to <output-dir>/sweep_report.html.",
    )
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

    synth_profiles = (
        _parse_synth_disagg_profiles(args.synth_disagg_profiles)
        if args.synth_disagg_from_agg
        else []
    )

    run_plans: list[dict[str, Any]] = []
    for row_idx in range(start, end):
        row = rows[row_idx]
        row_is_disagg = _is_disagg_row(row)

        if args.emit_disagg and not row_is_disagg and not args.synth_disagg_from_agg:
            raise ValueError(
                "--emit-disagg requires disagg-shaped CSV rows ((p)/(d) columns). "
                "For agg-only rows, pass --synth-disagg-from-agg."
            )

        run_plans.append(
            {
                "row_idx": row_idx,
                "variant": "agg",
                "run_mode": "agg",
                "emit_disagg": bool(args.emit_disagg and row_is_disagg),
            }
        )

        if args.synth_disagg_from_agg and not row_is_disagg:
            for prefill_replicas, decode_replicas in synth_profiles:
                run_plans.append(
                    {
                        "row_idx": row_idx,
                        "variant": f"syn_p{prefill_replicas}_d{decode_replicas}",
                        "run_mode": "disagg_synth",
                        "emit_disagg": False,
                        "prefill_replicas": prefill_replicas,
                        "decode_replicas": decode_replicas,
                    }
                )

    for run_rank, plan in enumerate(run_plans, start=1):
        row_idx = int(plan["row_idx"])
        variant = str(plan["variant"])
        run_dir_name = f"top{run_rank:02d}_row{row_idx}"
        if variant != "agg":
            run_dir_name += f"_{variant}"
        run_dir = output_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        sim_config_path = run_dir / "sim_config.json"
        result_path = run_dir / "result.json"
        server_log = run_dir / "server.log"
        bench_log = run_dir / "bench.log"

        bridge_cmd = _build_bridge_cmd(
            args,
            bridge_script=bridge_script,
            csv_path=csv_path,
            row_idx=row_idx,
            sim_config_path=sim_config_path,
            database_path=database_path,
            emit_disagg=bool(plan.get("emit_disagg", False)),
        )

        subprocess.run(bridge_cmd, check=True, env=env)

        if plan.get("run_mode") == "disagg_synth":
            row = rows[row_idx]
            prefill_replicas = int(plan["prefill_replicas"])
            decode_replicas = int(plan["decode_replicas"])
            with sim_config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["disagg"] = _build_synth_disagg_block(
                row,
                args,
                database_path=database_path,
                prefill_replicas=prefill_replicas,
                decode_replicas=decode_replicas,
            )
            resolved_moe_tp = cfg["disagg"].pop("_resolved_moe_tp_size", None)
            if resolved_moe_tp is not None:
                cfg.setdefault("scheduler", {})["moe_tp_size"] = int(resolved_moe_tp)
            with sim_config_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")

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
                    "run_mode": plan.get("run_mode"),
                    "variant": variant,
                    "synthetic_profile": (
                        f"p{plan['prefill_replicas']}:d{plan['decode_replicas']}"
                        if plan.get("run_mode") == "disagg_synth"
                        else ""
                    ),
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
                "run_mode": plan.get("run_mode"),
                "variant": variant,
                "synthetic_profile": (
                    f"p{plan['prefill_replicas']}:d{plan['decode_replicas']}"
                    if plan.get("run_mode") == "disagg_synth"
                    else ""
                ),
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

    if not args.no_plot and summary_rows and not args.dry_run:
        try:
            from hisim.visualization import sweep_data, sweep_report
        except ImportError as exc:
            print(
                f"[sweep] skip plot: visualization extras missing ({exc}). "
                f"Install with: pip install -e '.[viz]'"
            )
        else:
            plot_path = Path(args.plot_output) if args.plot_output else (output_dir / "sweep_report.html")
            try:
                df = sweep_data.load_sweep(summary_csv)
                df = sweep_data.add_derived_columns(df)
                sweep_report.render_report(
                    df,
                    plot_path,
                    title=f"HiSim Sweep — {output_dir.name}",
                    source=str(summary_csv),
                )
                print(f"[sweep] report={plot_path}")
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[sweep] skip plot: render failed ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
