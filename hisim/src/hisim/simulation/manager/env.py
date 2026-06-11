import os
from pathlib import Path
from hisim.utils.logger import get_logger

logger = get_logger("hisim")


class Envs:
    @classmethod
    def config_path(cls) -> str:
        HISIM_CONFIG_PATH = os.getenv("HISIM_CONFIG_PATH")
        if not HISIM_CONFIG_PATH or not os.path.exists(HISIM_CONFIG_PATH):
            raise RuntimeError(
                f"The mock configuration path is not set or does not exist({HISIM_CONFIG_PATH}). Please set it using the system variable HISIM_CONFIG_PATH"
            )
        return HISIM_CONFIG_PATH

    @classmethod
    def output_dir(cls) -> str:
        configured = os.getenv("HISIM_OUTPUT_DIR")
        candidates = []
        if configured:
            candidates.append(Path(configured))
        candidates.extend(
            [
                Path("/tmp/hisim/output"),
                Path.home() / ".cache" / "hisim" / "output",
                Path.cwd() / ".hisim" / "output",
            ]
        )

        for candidate in candidates:
            try:
                path = candidate.resolve()
                if path.exists() and path.is_file():
                    logger.error(
                        f"The metrics output path, {path}, exists and is a file."
                    )
                    continue
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".hisim_write_test"
                with test_file.open("w") as f:
                    f.write("ok")
                test_file.unlink(missing_ok=True)
                os.environ["HISIM_OUTPUT_DIR"] = str(path)
                return str(path)
            except OSError as e:
                logger.warning(f"Failed to create metrics output path {candidate}: {e}")

        raise RuntimeError(
            "Could not resolve a writable HISIM_OUTPUT_DIR. "
            "Set HISIM_OUTPUT_DIR to a writable directory and retry."
        )

    @classmethod
    def simulation_mode(cls) -> str:
        HISIM_SIMULATION_MODE = os.getenv("HISIM_SIMULATION_MODE", "OFFLINE").upper()
        assert HISIM_SIMULATION_MODE in ("BLOCKING", "OFFLINE")
        return HISIM_SIMULATION_MODE

    @classmethod
    def num_warmup(cls) -> int:
        # The number of warmup requests.
        return int(os.getenv("HISIM_NUM_WARMUP", "0"))

    @classmethod
    def reset_hicache_storage(cls) -> bool:
        return os.getenv("HISIM_RESET_HICACHE_STORAGE") == "1"
