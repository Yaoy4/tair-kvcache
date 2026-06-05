"""Run W1..W6 AGG workloads against a hisim mock sglang server.

Prereq: launch_server already running on --port (default 30099) with the
        h100_sxm + sglang 0.5.10 AIC config.

Run:
    PYTHONPATH=hisim/src:/home/cjia/projects/aiconfigurator/src \
        venv/bin/python hisim/tools/agg_h100_w1_w6.py \
        --model Qwen/Qwen3-8B --port 30099 \
        --out hisim/output/agg_h100_w1_w6.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Workload:
    name: str
    num: int
    isl: int
    osl: int
    max_concurrency: Optional[int] = None  # None = unlimited


# NOTE: --max-concurrency is incompatible with hisim's --bench-mode simulation
# (mock server doesn't get sim params and hangs). We let AGG batch naturally —
# that IS the AGG vs PD contrast (PD's W2 was serial because P=1/D=1 replicas).
WORKLOADS: List[Workload] = [
    Workload("W1-single", 1, 64, 16),
    Workload("W2-serial", 8, 64, 16),
    Workload("W3-concurrent", 32, 128, 64),
    Workload("W4-long-decode", 8, 32, 256),
    Workload("W5-long-prefill", 8, 1024, 16),
    Workload("W6-kv-bw-stress", 16, 512, 32),
]


def run_one(wl: Workload, model: str, host: str, port: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        out_path = tf.name
    try:
        cmd = [
            sys.executable, "-m", "hisim.simulation.bench_serving",
            "--warmup-requests", "0",
            "--bench-mode", "simulation",
            "--model", model,
            "--backend", "sglang",
            "--host", host,
            "--port", str(port),
            "--dataset-name", "random",
            "--random-input-len", str(wl.isl),
            "--random-output-len", str(wl.osl),
            "--random-range-ratio", "1",
            "--num-prompts", str(wl.num),
            "--request-rate", "inf",
            "--output-file", out_path,
        ]
        if wl.max_concurrency is not None:
            cmd += ["--max-concurrency", str(wl.max_concurrency)]
        print(f"[{wl.name}] running: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(f"[{wl.name}] FAILED rc={proc.returncode}")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            return {"name": wl.name, "error": f"rc={proc.returncode}", "stderr": proc.stderr[-1000:]}
        # Read last JSON line
        last = None
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if not last:
            return {"name": wl.name, "error": "no output line"}
        rec = json.loads(last)
        # Trim verbose server_info
        rec.pop("server_info", None)
        rec["_workload"] = {
            "name": wl.name, "num": wl.num, "isl": wl.isl, "osl": wl.osl,
            "max_concurrency": wl.max_concurrency,
        }
        return rec
    finally:
        try: os.unlink(out_path)
        except OSError: pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30099)
    p.add_argument("--out", default="hisim/output/agg_h100_w1_w6.json")
    p.add_argument("--only", default=None, help="Comma-separated workload names to run")
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None
    results = []
    for wl in WORKLOADS:
        if only and wl.name not in only:
            continue
        rec = run_one(wl, args.model, args.host, args.port)
        results.append(rec)
        if "error" not in rec:
            print(f"[{wl.name}] e2e={rec.get('mean_e2e_latency_ms', '?'):>8.2f} ms  "
                  f"ttft={rec.get('mean_ttft_ms', '?'):>7.2f} ms  "
                  f"tpot={rec.get('mean_tpot_ms', '?'):>6.2f} ms  "
                  f"duration={rec.get('duration', '?'):>7.3f} s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
