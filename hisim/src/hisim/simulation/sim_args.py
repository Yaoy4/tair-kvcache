import argparse
import dataclasses
import json
from typing import Optional

from hisim.simulation.pd_config import (
    BandwidthTransferConfig,
    DisaggConfig,
    RolePredictorConfig,
)


@dataclasses.dataclass
class AcceleratorConfig:
    name: str = "H20"


@dataclasses.dataclass
class PlatformConfig:
    accelerator: AcceleratorConfig = dataclasses.field(
        default_factory=AcceleratorConfig
    )
    disk_read_bandwidth_gb: float = 4.0
    disk_write_bandwidth_gb: float = 4.0
    memory_read_bandwidth_gb: float = 64.0
    memory_write_bandwidth_gb: float = 64.0


@dataclasses.dataclass
class PredictorConfig:
    name: str = "aiconfigurator"
    database_path: Optional[str] = None
    device_name: Optional[str] = None
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0


@dataclasses.dataclass
class SchedulerConfig:
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    data_type: str = "FP16"
    kv_cache_data_type: str = "FP16"
    backend_name: str = "sglang"
    backend_version: Optional[str] = None


@dataclasses.dataclass
class SimulationArgs:
    config_path: Optional[str] = None

    platform: PlatformConfig = dataclasses.field(default_factory=PlatformConfig)
    predictor: PredictorConfig = dataclasses.field(default_factory=PredictorConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    disagg: DisaggConfig = dataclasses.field(default_factory=DisaggConfig)

    def add_cli_args(parser: argparse.ArgumentParser):
        prefix = "sim-"

        parser.add_argument(
            f"--{prefix}config-path",
            dest="sim_config_path",
            type=str,
            default=None,
            help="Path to simulation JSON config (same as HISIM_CONFIG_PATH).",
        )

        parser.add_argument(
            f"--{prefix}accelerator-name",
            dest="sim_accelerator_name",
            type=str,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}disk-read-bandwidth-gb",
            dest="sim_disk_read_bandwidth_gb",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}disk-write-bandwidth-gb",
            dest="sim_disk_write_bandwidth_gb",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}memory-read-bandwidth-gb",
            dest="sim_memory_read_bandwidth_gb",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}memory-write-bandwidth-gb",
            dest="sim_memory_write_bandwidth_gb",
            type=float,
            default=None,
        )

        parser.add_argument(
            f"--{prefix}predictor-name",
            dest="sim_predictor_name",
            type=str,
            default=None,
            choices=["aiconfigurator"],
        )
        parser.add_argument(
            f"--{prefix}database-path", dest="sim_database_path", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}device-name", dest="sim_device_name", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}prefill-scale-factor",
            dest="sim_prefill_scale_factor",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}decode-scale-factor",
            dest="sim_decode_scale_factor",
            type=float,
            default=None,
        )

        parser.add_argument(
            f"--{prefix}tp-size", dest="sim_tp_size", type=int, default=None
        )
        parser.add_argument(
            f"--{prefix}ep-size", dest="sim_ep_size", type=int, default=None
        )
        parser.add_argument(
            f"--{prefix}data-type", dest="sim_data_type", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}kv-cache-data-type",
            dest="sim_kv_cache_data_type",
            type=str,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}backend-name", dest="sim_backend_name", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}backend-version",
            dest="sim_backend_version",
            type=str,
            default=None,
        )

        # ----- disagg flags -----
        parser.add_argument(
            f"--{prefix}disagg-enable",
            dest="sim_disagg_enable",
            action="store_true",
        )
        parser.add_argument(
            f"--{prefix}disagg-backend",
            dest="sim_disagg_backend",
            type=str,
            default=None,
            choices=["single_process", "two_process"],
        )
        parser.add_argument(
            f"--{prefix}disagg-decode-queue-mode",
            dest="sim_disagg_decode_queue_mode",
            type=str,
            default=None,
            choices=["single_replica", "per_replica_queue"],
        )
        for role in ("prefill", "decode"):
            parser.add_argument(
                f"--{prefix}disagg-{role}-device-name",
                dest=f"sim_disagg_{role}_device_name",
                type=str,
                default=None,
            )
            for short, full in (("tp", "tp_size"), ("ep", "ep_size"),
                                ("dp", "dp_size"), ("pp", "pp_size")):
                parser.add_argument(
                    f"--{prefix}disagg-{role}-{short}",
                    dest=f"sim_disagg_{role}_{full}",
                    type=int,
                    default=None,
                )
            parser.add_argument(
                f"--{prefix}disagg-{role}-replicas",
                dest=f"sim_disagg_{role}_replicas",
                type=int,
                default=None,
            )
            parser.add_argument(
                f"--{prefix}disagg-{role}-max-running",
                dest=f"sim_disagg_{role}_max_running_per_replica",
                type=int,
                default=None,
            )
            parser.add_argument(
                f"--{prefix}disagg-{role}-data-type",
                dest=f"sim_disagg_{role}_data_type",
                type=str,
                default=None,
            )
            parser.add_argument(
                f"--{prefix}disagg-{role}-kv-cache-data-type",
                dest=f"sim_disagg_{role}_kv_cache_data_type",
                type=str,
                default=None,
            )
            parser.add_argument(
                f"--{prefix}disagg-{role}-database-path",
                dest=f"sim_disagg_{role}_database_path",
                type=str,
                default=None,
            )
            parser.add_argument(
                f"--{prefix}disagg-{role}-backend-version",
                dest=f"sim_disagg_{role}_backend_version",
                type=str,
                default=None,
            )
        parser.add_argument(
            f"--{prefix}kv-transfer-bw-gbps",
            dest="sim_kv_transfer_bw_gbps",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}kv-transfer-latency-us",
            dest="sim_kv_transfer_latency_us",
            type=float,
            default=None,
        )

    @staticmethod
    def from_json(path: str) -> "SimulationArgs":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        platform = cfg.get("platform", {})
        accelerator = platform.get("accelerator", {})
        predictor = cfg.get("predictor", {})
        scheduler = cfg.get("scheduler", {})

        return SimulationArgs(
            config_path=path,
            platform=PlatformConfig(
                accelerator=AcceleratorConfig(name=accelerator.get("name", "H20")),
                disk_read_bandwidth_gb=platform.get("disk_read_bandwidth_gb", 4.0),
                disk_write_bandwidth_gb=platform.get("disk_write_bandwidth_gb", 4.0),
                memory_read_bandwidth_gb=platform.get("memory_read_bandwidth_gb", 64.0),
                memory_write_bandwidth_gb=platform.get(
                    "memory_write_bandwidth_gb", 64.0
                ),
            ),
            predictor=PredictorConfig(
                name=predictor.get("name", "aiconfigurator"),
                database_path=predictor.get("database_path"),
                device_name=predictor.get("device_name"),
                prefill_scale_factor=predictor.get("prefill_scale_factor", 1.0),
                decode_scale_factor=predictor.get("decode_scale_factor", 1.0),
            ),
            scheduler=SchedulerConfig(
                tp_size=scheduler.get("tp_size", 1),
                ep_size=scheduler.get("ep_size", 1),
                data_type=scheduler.get("data_type", "FP16"),
                kv_cache_data_type=scheduler.get("kv_cache_data_type", "FP16"),
                backend_name=scheduler.get("backend_name", "sglang"),
                backend_version=scheduler.get("backend_version"),
            ),
            disagg=_disagg_from_dict(cfg.get("disagg", {})),
        )

    def to_dict(self, indent=2, ensure_ascii: bool = False) -> dict:
        data = dataclasses.asdict(self)
        data.pop("config_path", None)
        return data

    @classmethod
    def from_cli_args(cls, ns: argparse.Namespace) -> "SimulationArgs":
        # config_path
        if getattr(ns, "sim_config_path", None) is not None:
            return SimulationArgs.from_json(ns.sim_config_path)

        args = SimulationArgs()

        # platform
        if getattr(ns, "sim_accelerator_name", None) is not None:
            args.platform.accelerator.name = ns.sim_accelerator_name
        for arg, field in [
            ("sim_disk_read_bandwidth_gb", "disk_read_bandwidth_gb"),
            ("sim_disk_write_bandwidth_gb", "disk_write_bandwidth_gb"),
            ("sim_memory_read_bandwidth_gb", "memory_read_bandwidth_gb"),
            ("sim_memory_write_bandwidth_gb", "memory_write_bandwidth_gb"),
        ]:
            v = getattr(ns, arg, None)
            if v is not None:
                setattr(args.platform, field, v)

        # predictor
        if getattr(ns, "sim_predictor_name", None) is not None:
            args.predictor.name = ns.sim_predictor_name
        for arg, field in [
            ("sim_database_path", "database_path"),
            ("sim_device_name", "device_name"),
            ("sim_prefill_scale_factor", "prefill_scale_factor"),
            ("sim_decode_scale_factor", "decode_scale_factor"),
        ]:
            v = getattr(ns, arg, None)
            if v is not None:
                setattr(args.predictor, field, v)

        # scheduler
        for arg, field in [
            ("sim_tp_size", "tp_size"),
            ("sim_ep_size", "ep_size"),
            ("sim_data_type", "data_type"),
            ("sim_kv_cache_data_type", "kv_cache_data_type"),
            ("sim_backend_name", "backend_name"),
            ("sim_backend_version", "backend_version"),
        ]:
            v = getattr(ns, arg, None)
            if v is not None:
                setattr(args.scheduler, field, v)

        # disagg
        args.disagg = _disagg_from_cli(ns)

        return args


_ROLE_FIELDS = (
    "device_name",
    "tp_size",
    "ep_size",
    "dp_size",
    "pp_size",
    "replicas",
    "max_running_per_replica",
    "data_type",
    "kv_cache_data_type",
    "database_path",
    "backend_version",
)


def _role_from_dict(d: dict) -> RolePredictorConfig:
    kwargs = {k: d[k] for k in _ROLE_FIELDS if k in d and d[k] is not None}
    if "prefill_scale_factor" in d:
        kwargs["prefill_scale_factor"] = d["prefill_scale_factor"]
    if "decode_scale_factor" in d:
        kwargs["decode_scale_factor"] = d["decode_scale_factor"]
    return RolePredictorConfig(**kwargs)


def _disagg_from_dict(d: dict) -> DisaggConfig:
    if not d.get("enabled", False):
        return DisaggConfig()
    prefill = _role_from_dict(d["prefill"]) if "prefill" in d else None
    decode = _role_from_dict(d["decode"]) if "decode" in d else None
    transfer = None
    if "kv_transfer" in d:
        kv = d["kv_transfer"]
        transfer = BandwidthTransferConfig(
            bw_gbps=kv["bw_gbps"], latency_us=kv["latency_us"]
        )
    return DisaggConfig(
        enabled=True,
        backend=d.get("backend", "single_process"),
        decode_queue_mode=d.get("decode_queue_mode", "single_replica"),
        prefill=prefill,
        decode=decode,
        kv_transfer=transfer,
    )


def _role_from_cli(ns: argparse.Namespace, role: str) -> Optional[RolePredictorConfig]:
    device_name = getattr(ns, f"sim_disagg_{role}_device_name", None)
    if device_name is None:
        return None
    kwargs = {"device_name": device_name}
    for field in (
        "tp_size",
        "ep_size",
        "dp_size",
        "pp_size",
        "replicas",
        "max_running_per_replica",
        "data_type",
        "kv_cache_data_type",
        "database_path",
        "backend_version",
    ):
        v = getattr(ns, f"sim_disagg_{role}_{field}", None)
        if v is not None:
            kwargs[field] = v
    return RolePredictorConfig(**kwargs)


def _disagg_from_cli(ns: argparse.Namespace) -> DisaggConfig:
    if not getattr(ns, "sim_disagg_enable", False):
        return DisaggConfig()
    prefill = _role_from_cli(ns, "prefill")
    decode = _role_from_cli(ns, "decode")
    bw = getattr(ns, "sim_kv_transfer_bw_gbps", None)
    lat = getattr(ns, "sim_kv_transfer_latency_us", None)
    transfer = (
        BandwidthTransferConfig(bw_gbps=bw, latency_us=lat)
        if bw is not None and lat is not None
        else None
    )
    backend = getattr(ns, "sim_disagg_backend", None) or "single_process"
    return DisaggConfig(
        enabled=True,
        backend=backend,
        decode_queue_mode=getattr(ns, "sim_disagg_decode_queue_mode", None)
        or "single_replica",
        prefill=prefill,
        decode=decode,
        kv_transfer=transfer,
    )
