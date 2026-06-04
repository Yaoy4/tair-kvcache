#!/usr/bin/env python3
"""Bridge AIConfigurator search output to Hisim simulation config.

This script reads one row from AIConfigurator CSV output (for example
`best_config_topn.csv` or `pareto.csv`) and generates a Hisim-compatible
simulation config JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


# NOTE on configurability:
#   platform.accelerator.name -> HiSim's internal hw identity (registered in
#       hisim/spec/accelerator/info.py). Defaults to "H20" because that is the
#       only profile currently registered; override via --platform-accelerator.
#   predictor.device_name     -> AIConfigurator perf-database device, e.g.
#       "h100_sxm", "h200_sxm", "b60", or any future accelerator with a perf
#       db under aiconfigurator/systems/data/<name>/. This MUST be supplied
#       either via the AIC CSV `system` column or --predictor-device-name; the
#       bridge no longer falls back to a hardcoded device.
DEFAULT_CONFIG: dict[str, Any] = {
    "platform": {
        "accelerator": {"name": "H20"},
        "disk_read_bandwidth_gb": 8,
        "disk_write_bandwidth_gb": 8,
        "memory_read_bandwidth_gb": 16,
        "memory_write_bandwidth_gb": 16,
        "num_device_per_node": 8,
    },
    "predictor": {
        "name": "aiconfigurator",
    },
    "scheduler": {
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
    },
}


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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_get(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return str(row[key]).strip()
    return None


def _normalize_dtype(value: str | None, default: str | None = None) -> str | None:
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    upper_raw = raw.upper()
    if upper_raw in {"FP16", "BF16", "FP8", "INT8", "FP4", "INT4"}:
        return upper_raw

    normalized = DTYPE_MAP.get(raw.lower())
    if normalized is not None:
        return normalized
    return default


def _load_csv_row(csv_path: Path, row_index: int) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(
            f"row-index {row_index} out of range for {csv_path} (rows={len(rows)})"
        )
    return rows[row_index]


def _select_mode_prefix(row: dict[str, str], disagg_role: str) -> str:
    if "tp" in row:
        return ""
    if "(p)tp" in row or "(d)tp" in row:
        return "(p)" if disagg_role == "prefill" else "(d)"
    raise ValueError(
        "Cannot find AIC parallelism columns. Expected agg columns like 'tp' or disagg columns like '(d)tp'."
    )


def _build_config_from_row(
    row: dict[str, str],
    disagg_role: str,
    base_config: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = deepcopy(base_config if base_config is not None else DEFAULT_CONFIG)

    mode_prefix = _select_mode_prefix(row, disagg_role)
    tp = _to_int(_row_get(row, f"{mode_prefix}tp"), 1)
    dp = _to_int(_row_get(row, f"{mode_prefix}dp"), 1)
    ep = _to_int(_row_get(row, f"{mode_prefix}moe_ep"), 1)
    # AIC pareto rows carry both `moe_tp` and `moe_ep`. We forward both so
    # HiSim's predictor builds a valid ModelConfig
    # (tp_size * dp_size == moe_tp_size * ep_size). Fall back to tp for
    # dense-style legacy rows that lack `moe_tp`.
    moe_tp = _to_int(_row_get(row, f"{mode_prefix}moe_tp"), tp)

    row_system = _row_get(row, f"{mode_prefix}system", "system")
    row_backend_version = _row_get(row, f"{mode_prefix}version", "version")
    row_gemm = _row_get(row, f"{mode_prefix}gemm", "gemm")
    row_kvcache = _row_get(row, f"{mode_prefix}kvcache", "kvcache")

    platform = cfg.setdefault("platform", {})
    accel = platform.setdefault("accelerator", {})
    accel["name"] = args.platform_accelerator
    platform["disk_read_bandwidth_gb"] = args.disk_read_bandwidth_gb
    platform["disk_write_bandwidth_gb"] = args.disk_write_bandwidth_gb
    platform["memory_read_bandwidth_gb"] = args.memory_read_bandwidth_gb
    platform["memory_write_bandwidth_gb"] = args.memory_write_bandwidth_gb
    platform["num_device_per_node"] = args.num_device_per_node

    predictor = cfg.setdefault("predictor", {})
    predictor["name"] = "aiconfigurator"
    predictor["database_path"] = args.database_path
    device_name = args.predictor_device_name or row_system
    if not device_name:
        import warnings
        device_name = "h100_sxm"
        warnings.warn(
            "No predictor device_name resolved from CSV or --predictor-device-name;"
            f" defaulting to {device_name!r}. Pass --predictor-device-name to use"
            " a different AIC perf-db device.",
            stacklevel=2,
        )
    predictor["device_name"] = device_name
    predictor["database_mode"] = args.database_mode

    if args.prefill_scale_factor is not None:
        predictor["prefill_scale_factor"] = args.prefill_scale_factor
    if args.decode_scale_factor is not None:
        predictor["decode_scale_factor"] = args.decode_scale_factor
    if args.xgb_model_path is not None:
        predictor["xgb_model_path"] = args.xgb_model_path

    scheduler = cfg.setdefault("scheduler", {})
    scheduler["tp_size"] = tp
    scheduler["dp_size"] = dp
    scheduler["ep_size"] = ep
    scheduler["moe_tp_size"] = moe_tp

    dtype = _normalize_dtype(args.data_type, _normalize_dtype(row_gemm))
    if dtype is not None:
        scheduler["data_type"] = dtype

    kv_dtype = _normalize_dtype(args.kv_cache_data_type, _normalize_dtype(row_kvcache, dtype))
    if kv_dtype is not None:
        scheduler["kv_cache_data_type"] = kv_dtype

    scheduler["backend_version"] = args.backend_version or row_backend_version or scheduler.get(
        "backend_version", "0.5.6.post2"
    )

    return cfg


def _build_disagg_role_dict(
    row: dict[str, str], prefix: str, args: argparse.Namespace
) -> dict[str, Any]:
    """Build a RolePredictorConfig-shaped dict from a disagg AIC row.

    prefix is '(p)' for prefill, '(d)' for decode. Keys produced must match
    fields accepted by hisim.simulation.pd_config.RolePredictorConfig (i.e.
    consumable by sim_args._disagg_from_dict).
    """
    role_device = (
        _row_get(row, f"{prefix}system") or args.predictor_device_name
    )
    if not role_device:
        import warnings
        role_device = "h100_sxm"
        warnings.warn(
            f"No device_name for disagg role with prefix {prefix!r}; defaulting"
            f" to {role_device!r}. Provide `{prefix}system` in the AIC CSV or"
            " pass --predictor-device-name to override.",
            stacklevel=2,
        )
    role: dict[str, Any] = {
        "device_name": role_device,
        "tp_size": _to_int(_row_get(row, f"{prefix}tp"), 1),
        "ep_size": _to_int(_row_get(row, f"{prefix}moe_ep"), 1),
        "dp_size": _to_int(_row_get(row, f"{prefix}dp"), 1),
        "pp_size": _to_int(_row_get(row, f"{prefix}pp"), 1),
        "replicas": _to_int(_row_get(row, f"{prefix}workers"), 1),
    }
    data_type = _normalize_dtype(_row_get(row, f"{prefix}gemm"))
    if data_type is not None:
        role["data_type"] = data_type
    kv_dtype = _normalize_dtype(_row_get(row, f"{prefix}kvcache"), data_type)
    if kv_dtype is not None:
        role["kv_cache_data_type"] = kv_dtype
    backend_version = _row_get(row, f"{prefix}version") or args.backend_version
    if backend_version is not None:
        role["backend_version"] = backend_version
    if args.database_path is not None:
        role["database_path"] = args.database_path
    return role


def _build_disagg_block(
    row: dict[str, str], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "enabled": True,
        "backend": args.disagg_backend,
        "prefill": _build_disagg_role_dict(row, "(p)", args),
        "decode": _build_disagg_role_dict(row, "(d)", args),
        "kv_transfer": {
            "bw_gbps": float(args.kv_transfer_bw_gbps),
            "latency_us": float(args.kv_transfer_latency_us),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one AIConfigurator CSV row to a Hisim simulation config JSON"
    )
    parser.add_argument("--aic-csv", required=True, help="Path to AIC CSV (best_config_topn.csv or pareto.csv)")
    parser.add_argument("--row-index", type=int, default=0, help="Row index to convert (0-based)")
    parser.add_argument("--output", required=True, help="Output Hisim config JSON path")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional existing Hisim config JSON used as template before overrides",
    )
    parser.add_argument(
        "--disagg-role",
        choices=["decode", "prefill"],
        default="decode",
        help="When CSV is disagg schema, choose which role's parallelism/system to map",
    )

    parser.add_argument("--database-path", required=True, help="Path to aiconfigurator systems directory")
    parser.add_argument("--database-mode", default="SILICON", help="AIC database mode for predictor")
    parser.add_argument(
        "--predictor-device-name",
        default=None,
        help="Override predictor.device_name; by default read from CSV system column",
    )
    parser.add_argument("--backend-version", default=None, help="Override scheduler.backend_version")

    parser.add_argument("--prefill-scale-factor", type=float, default=None)
    parser.add_argument("--decode-scale-factor", type=float, default=None)
    parser.add_argument("--xgb-model-path", default=None)

    parser.add_argument("--data-type", default=None, help="Override scheduler.data_type (FP16/BF16/FP8/INT8/FP4/INT4)")
    parser.add_argument(
        "--kv-cache-data-type",
        default=None,
        help="Override scheduler.kv_cache_data_type (FP16/BF16/FP8/INT8/FP4/INT4)",
    )

    parser.add_argument("--platform-accelerator", default="H20", help="platform.accelerator.name")
    parser.add_argument("--disk-read-bandwidth-gb", type=float, default=8)
    parser.add_argument("--disk-write-bandwidth-gb", type=float, default=8)
    parser.add_argument("--memory-read-bandwidth-gb", type=float, default=16)
    parser.add_argument("--memory-write-bandwidth-gb", type=float, default=16)
    parser.add_argument("--num-device-per-node", type=int, default=8)

    # --- Phase 1d: disagg emission ---
    parser.add_argument(
        "--emit-disagg",
        action="store_true",
        help="Emit a top-level `disagg` block (requires a disagg-shaped AIC CSV "
             "with (p)tp / (d)tp columns and --kv-transfer-* flags).",
    )
    parser.add_argument(
        "--disagg-backend",
        choices=["single_process", "two_process"],
        default="single_process",
        help="DisaggConfig.backend value when --emit-disagg is set.",
    )
    parser.add_argument(
        "--kv-transfer-bw-gbps",
        type=float,
        default=None,
        help="KV-transfer bandwidth in GB/s. Required with --emit-disagg.",
    )
    parser.add_argument(
        "--kv-transfer-latency-us",
        type=float,
        default=None,
        help="KV-transfer fixed latency in microseconds. Required with --emit-disagg.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    aic_csv = Path(args.aic_csv)
    if not aic_csv.exists():
        raise FileNotFoundError(f"AIC CSV not found: {aic_csv}")

    base_config = None
    if args.base_config is not None:
        base_path = Path(args.base_config)
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        with base_path.open("r", encoding="utf-8") as f:
            base_config = json.load(f)

    row = _load_csv_row(aic_csv, args.row_index)
    out_cfg = _build_config_from_row(row, args.disagg_role, base_config, args)

    if args.emit_disagg:
        if "(p)tp" not in row or "(d)tp" not in row:
            raise SystemExit(
                "--emit-disagg requires a disagg-shaped CSV with (p)tp and (d)tp columns"
            )
        if args.kv_transfer_bw_gbps is None or args.kv_transfer_latency_us is None:
            raise SystemExit(
                "--emit-disagg requires --kv-transfer-bw-gbps and --kv-transfer-latency-us"
            )
        out_cfg["disagg"] = _build_disagg_block(row, args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
        f.write("\n")

    selected_mode = "agg" if "tp" in row else f"disagg/{args.disagg_role}"
    print(f"[bridge] source={aic_csv} row={args.row_index} mode={selected_mode}")
    print(f"[bridge] wrote hisim config: {out_path}")
    print(
        "[bridge] mapped scheduler: "
        f"tp={out_cfg['scheduler'].get('tp_size')} "
        f"dp={out_cfg['scheduler'].get('dp_size')} "
        f"ep={out_cfg['scheduler'].get('ep_size')} "
        f"moe_tp={out_cfg['scheduler'].get('moe_tp_size')} "
        f"dtype={out_cfg['scheduler'].get('data_type')} "
        f"kv_dtype={out_cfg['scheduler'].get('kv_cache_data_type')}"
    )
    print(
        "[bridge] mapped predictor: "
        f"device={out_cfg['predictor'].get('device_name')} "
        f"db={out_cfg['predictor'].get('database_path')}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
