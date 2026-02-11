import argparse
import array
import os
import time
from pathlib import Path

import mlperf_loadgen as lg
import requests
import yaml

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}
NANO_SEC = 1e9
MILLI_SEC = 1000


def parse_args():
    parser = argparse.ArgumentParser(description="MLPerf Loadgen Argument Parser")

    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/vbench_prompts.txt",
        help="Path to dataset file (text prompts, one per line) (default: ./data/prompts.txt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./inference_config.yaml",
        help="Path to inference configuration file (default: ./inference_config.yaml)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=-1, help="Process only first N prompts (for testing, default: all)"
    )  # MLPerf loadgen arguments
    parser.add_argument(
        "--scenario",
        default="SingleStream",
        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument("--audit_conf", default="audit.config", help="config for LoadGen audit settings")
    parser.add_argument(
        "--performance-sample-count",
        type=int,
        help="performance sample count",
        default=5000,
    )
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    # Dont overwrite these for official submission
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--qps", type=float, help="target qps")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--samples-per-query",
        default=8,
        type=int,
        help="mlperf multi-stream samples per query",
    )
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument(
        "--division", type=str, default="closed", choices=["open", "closed"], help="Which division to operate in"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to send requests to the server on")
    parser.add_argument("--hosts", type=str, nargs="+", default=["localhost"], help="Addresses of Kiss-V servers")
    parser.add_argument(
        "--videos_dir",
        default="./videos",
        help="Shared directory where Kiss-V servers will store videos",
    )

    return parser.parse_args()


def load_mlperf_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(dataset_path):
    """Load prompts from dataset file."""
    with open(dataset_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


class KissAdapter:
    def __init__(self, config, prompts, args):
        self._prompts = prompts

        self._negative_prompt = config["negative_prompt"].strip()

        self._request_params = {
            "height": config["height"],
            "width": config["width"],
            "frames": config["num_frames"],
            "guidance_scale": config["guidance_scale"],
            "low_noise_guidance_scale": config["guidance_scale_2"],
            "boundary_ratio": config["boundary_ratio"],
            "seed": config["seed"],
            "max_iteration_steps": config["sample_steps"],
        }

        if args.division == "open":
            self._request_params["cache_level"] = "slow"

        self.hosts = args.hosts
        self.port = args.port
        self.videos_dir = args.videos_dir
        
        for host in self.hosts:
            while True:
                resp = requests.get(f"http://{host}:{self.port}/health/")
                if resp.status_code == 200:
                    break
                time.sleep(10)

    def _send_query(self, req, host):
        resp = requests.post(f"http://{host}:{self.port}/generate_video/", json=req)
        assert resp.status_code == 200, "Request failed"

    def _wait_for_response(self, uid):
        video_path = os.path.join(self.videos_dir, f"{uid}.mp4")
        while not os.path.exists(video_path):
            time.sleep(0.05)

    def issue_queries(self, query_samples):
        idx = [q.index for q in query_samples]
        query_ids = [q.id for q in query_samples]

        for i, q in zip(idx, query_ids):
            req = self._request_params | {"uuid": q, "pos": self._prompts[i], "neg": self._negative_prompt, "media": []}
            host = self.hosts[i % len(self.hosts)]

            self._send_query(req, host)

        response = []
        response_array_refs = []
        for i, q in enumerate(query_ids):
            video_path = os.path.join(self.videos_dir, f"{q}.mp4")
            self._wait_for_response(q)

            with open(video_path, "rb") as f:
                resp = f.read()

            response_array = array.array("B", resp)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(q, bi[0], bi[1]))
            response_array_refs.append(response_array)
            print(f"Sample index {i + 1} complete {time.time()}")

        lg.QuerySamplesComplete(response)

    def flush_queries(self):
        pass

    def warmup(self, prompt: str, steps: int):
        req = self._request_params | {
            "pos": prompt,
            "neg": self._negative_prompt,
            "media": [],
            "max_iteration_steps": steps,
        }

        for i, host in enumerate(self.hosts):
            self._send_query(req | {"uuid": f"warmup_host_{i}"}, host)

        for i in range(len(self.hosts)):
            self._wait_for_response(f"warmup_host_{i}")


def run_mlperf(args, config):
    # Load dataset
    dataset = load_prompts(args.dataset)

    # Generation parameters from config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_lg = str(args.output_dir)

    # Prepare loadgen for run
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir_lg
    log_output_settings.copy_summary_to_stdout = False

    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings

    user_conf = os.path.abspath(args.user_conf)
    settings = lg.TestSettings()
    settings.FromConfig(user_conf, "wan-2.2-t2v-a14b", args.scenario)

    audit_config = os.path.abspath(args.audit_conf)
    if os.path.exists(audit_config):
        settings.FromConfig(audit_config, "wan-2.2-t2v-a14b", args.scenario)
    settings.scenario = SCENARIO_MAP[args.scenario]

    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC
    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    count = args.count
    # count_override = False
    # if count:
    #     count_override = True

    if args.count:
        settings.min_query_count = count
        settings.max_query_count = count
    count = len(dataset)

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(args.max_latency * NANO_SEC)

    performance_sample_count = args.performance_sample_count if args.performance_sample_count else min(count, 500)

    def empty(*args, **kwargs):
        pass

    adapter = KissAdapter(config, dataset, args)

    # Warmup model with 1 query
    adapter.warmup("The quick brown fox jumped over the lazy dog", steps=2)

    sut = lg.ConstructSUT(adapter.issue_queries, adapter.flush_queries)
    qsl = lg.ConstructQSL(count, performance_sample_count, empty, empty)

    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings, audit_config)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


if __name__ == "__main__":
    args = parse_args()

    config = load_mlperf_config(args.config)
    run_mlperf(
        args,
        config,
    )
