import subprocess
import time
from pathlib import Path

import yaml


def generate_inference_config_yaml(
    output_dir,
    inference_config_filename,
    height,
    width,
    num_frames,
    fps,
    sample_steps,
    seed,
    guidance_scale,
    guidance_scale_2,
    boundary_ratio,
    negative_prompt
):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / inference_config_filename

    data = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps": fps,
        "sample_steps": sample_steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "guidance_scale_2": guidance_scale_2,
        "boundary_ratio": boundary_ratio,
        "negative_prompt": negative_prompt
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True
        )

    return str(path.resolve())

def get_params(params_dict):
    params = []
    for key, value in params_dict.items():
        params.append(f"--{key} {value}")
    return " ".join(params)

def run_experiment(nodes: int, node_idx: int, server_run_cmd: str, mlperf_cmd: str):
    """
    Orchestrates MLPerf experiments with distinct logic for single-node, 
    multi-node master, and multi-node worker scenarios.
    """

    # --- CASE 1: SINGLE NODE ---
    if nodes == 1:
        print("[Single-Node] Starting experiment...")
        sidecar = subprocess.Popen(server_run_cmd, shell=True)
        try:
            workload = subprocess.Popen(mlperf_cmd, shell=True)
            workload.wait()
        finally:
            sidecar.terminate()
            sidecar.wait()
            print("[Single-Node] Cleanup complete.")
        return "success"

    # --- CASE 2: MULTI-NODE (MASTER) ---
    if nodes > 1 and node_idx == 0:
        print("[Multi-Node Master] Initializing Redis and Sidecar...")
        # Start Redis
        subprocess.run("docker run --rm -d --name redis-master -p 6379:6379 redis", shell=True)
        time.sleep(2) 
        
        sidecar = subprocess.Popen(server_run_cmd, shell=True)
        try:
            workload = subprocess.Popen(mlperf_cmd, shell=True)
            workload.wait()
        finally:
            sidecar.terminate()
            sidecar.wait()
            subprocess.run("docker stop redis-master", shell=True)
            print("[Multi-Node Master] Cleanup complete.")
        return "success"

    # --- CASE 3: MULTI-NODE (WORKER) ---
    if nodes > 1 and node_idx > 0:
        print(f"[Multi-Node Worker {node_idx}] Running sidecar and waiting...")
        # Workers simply run the docker_cmd and wait for it to exit
        subprocess.run(server_run_cmd, shell=True)
        print(f"[Multi-Node Worker {node_idx}] Execution finished.")
        
    return "success"