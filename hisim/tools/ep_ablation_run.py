#!/usr/bin/env python3
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/cjia/projects/tair-kvcache/hisim')
BASE_CFG = ROOT / 'output/my_aic_bridge_sweep_disagg_top3/top01_row0/sim_config.json'
OUT_ROOT = ROOT / f"output/ep_ablation_cpu_{time.strftime('%Y%m%d_%H%M%S')}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ENV = os.environ.copy()
py_path = '/home/cjia/projects/tair-kvcache/hisim/src:/home/cjia/projects/aiconfigurator/src'
if ENV.get('PYTHONPATH'):
    py_path = py_path + ':' + ENV['PYTHONPATH']
ENV['PYTHONPATH'] = py_path
ENV['SGLANG_USE_CPU_ENGINE'] = '1'
ENV['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

PY = '/home/cjia/projects/tair-kvcache/venv/bin/python'


def wait_port(host: str = '127.0.0.1', port: int = 30000, timeout: int = 240) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return True
            except Exception:
                time.sleep(1)
    return False


rows = []
for ep in [1, 2, 4]:
    run_dir = OUT_ROOT / f'ep{ep}'
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / 'sim_config.json'
    shutil.copy(BASE_CFG, cfg_path)
    cfg = json.loads(cfg_path.read_text())
    cfg['scheduler']['ep_size'] = ep
    cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')

    server_log_path = run_dir / 'server.log'
    bench_log_path = run_dir / 'bench.log'
    result_path = run_dir / 'result.json'

    with server_log_path.open('w', encoding='utf-8') as server_log, bench_log_path.open('w', encoding='utf-8') as bench_log:
        server_cmd = [
            PY,
            '-m',
            'hisim.simulation.sglang.launch_server',
            '--model-path',
            'Qwen/Qwen3-8B',
            '--sim-config-path',
            str(cfg_path),
            '--skip-server-warmup',
        ]
        proc = subprocess.Popen(
            server_cmd,
            cwd=str(ROOT),
            env=ENV,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            if not wait_port():
                raise RuntimeError(f'server not ready for ep={ep}')

            bench_cmd = [
                PY,
                '-m',
                'hisim.simulation.bench_serving',
                '--host',
                '127.0.0.1',
                '--warmup-requests',
                '0',
                '--bench-mode',
                'simulation',
                '--backend',
                'sglang',
                '--model',
                'Qwen/Qwen3-8B',
                '--dataset-name',
                'random',
                '--request-rate',
                '1.0',
                '--random-input-len',
                '256',
                '--random-output-len',
                '128',
                '--random-range-ratio',
                '0.0',
                '--num-prompts',
                '6',
                '--output-file',
                str(result_path),
            ]
            rc = subprocess.run(
                bench_cmd,
                cwd=str(ROOT),
                env=ENV,
                stdout=bench_log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
            if rc != 0:
                raise RuntimeError(f'bench failed for ep={ep}, rc={rc}')

            d = json.loads(result_path.read_text())
            rows.append(
                (
                    ep,
                    d.get('request_throughput'),
                    d.get('output_throughput'),
                    d.get('mean_ttft_ms'),
                    d.get('mean_tpot_ms'),
                    d.get('p99_e2e_latency_ms'),
                )
            )
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)

print(f'OUT_ROOT={OUT_ROOT}')
print('ep,request_throughput,output_throughput,mean_ttft_ms,mean_tpot_ms,p99_e2e_latency_ms')
for r in rows:
    print(','.join(str(x) for x in r))
