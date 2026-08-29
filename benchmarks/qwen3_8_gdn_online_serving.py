# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched online-serving benchmark for Qwen3.8 GDN state storage dtypes.

This runner starts one server process at a time and keeps every serving and
workload setting fixed except ``mamba_ssm_cache_dtype``. It delegates request
generation and latency measurement to ``vllm bench serve``.
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import regex as re

STATE_DTYPES = ("float32", "fp8_e4m3fn")
PERCENTILE_METRICS = ("ttft", "tpot", "itl", "e2el")
PERCENTILES = (50, 90, 99)


def _run_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def _gpu_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {"timestamp": time.time()}
    try:
        gpu_rows = _run_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        snapshot["gpus"] = [
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": int(parts[3]),
                "memory_used_mib": int(parts[4]),
            }
            for line in gpu_rows.splitlines()
            if line.strip()
            for parts in [[part.strip() for part in line.split(",")]]
        ]
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        snapshot["gpu_query_error"] = str(exc)

    try:
        app_rows = _run_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        snapshot["compute_apps"] = [
            {"pid": int(parts[0]), "memory_used_mib": int(parts[1])}
            for line in app_rows.splitlines()
            if line.strip()
            for parts in [[part.strip() for part in line.split(",")]]
        ]
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        snapshot["compute_app_query_error"] = str(exc)
    return snapshot


def _fetch_text(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])


def _wait_for_server(
    process: subprocess.Popen[str],
    base_url: str,
    log_path: Path,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Server exited with code {return_code}:\n{_tail(log_path)}"
            )
        try:
            _fetch_text(f"{base_url}/health")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"Server was not ready after {timeout}s:\n{_tail(log_path)}")


def _stop_process_group(process: subprocess.Popen[str], timeout: float = 120) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _parse_prometheus_labels(label_text: str) -> dict[str, str]:
    labels = {}
    pattern = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"(?:,|$)')
    for match in pattern.finditer(label_text):
        value = bytes(match.group(2), "utf-8").decode("unicode_escape")
        labels[match.group(1)] = value
    return labels


def _parse_cache_config_metrics(metrics_text: str) -> dict[str, str]:
    for line in metrics_text.splitlines():
        if not line.startswith(("vllm:cache_config_info{", "vllm_cache_config_info{")):
            continue
        labels = line.split("{", 1)[1].rsplit("}", 1)[0]
        return _parse_prometheus_labels(labels)
    return {}


def _parse_capacity_log(log_text: str) -> dict[str, object]:
    capacity: dict[str, object] = {}
    patterns = {
        "gpu_kv_cache_size_tokens": r"GPU KV cache size:\s*([\d,]+) tokens",
        "available_mamba_cache_blocks": (
            r"available Mamba cache blocks\s*\(?([\d,]+)\)?"
        ),
        "maximum_concurrency": (
            r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x"
        ),
        "peak_model_weight_memory_gib": (
            r"Peak GPU memory after loading weights:\s*([\d.]+) GiB"
        ),
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
        if not matches:
            continue
        raw = matches[-1].replace(",", "")
        capacity[key] = float(raw) if "." in raw else int(raw)
    return capacity


def _memory_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    observations: list[tuple[int, dict[str, object]]] = []
    for sample in samples:
        for gpu in sample.get("gpus", []):
            if isinstance(gpu, dict) and "memory_used_mib" in gpu:
                observations.append((int(gpu["memory_used_mib"]), sample))
    if not observations:
        return {"num_samples": len(samples)}
    minimum = min(observations, key=lambda item: item[0])
    maximum = max(observations, key=lambda item: item[0])
    return {
        "num_samples": len(samples),
        "min_gpu_memory_used_mib": minimum[0],
        "max_gpu_memory_used_mib": maximum[0],
        "peak_snapshot": maximum[1],
    }


def _run_client(command: list[str], log_path: Path) -> dict[str, object]:
    samples = [_gpu_snapshot()]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            samples.append(_gpu_snapshot())
            time.sleep(0.5)
    if process.returncode:
        raise RuntimeError(
            f"Benchmark exited with code {process.returncode}:\n{_tail(log_path)}"
        )
    samples.append(_gpu_snapshot())
    return _memory_summary(samples)


def _server_command(args: argparse.Namespace, state_dtype: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--mamba-ssm-cache-dtype",
        state_dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(max(args.concurrencies)),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--no-enable-prefix-caching",
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    return command


def _client_command(
    args: argparse.Namespace,
    state_dtype: str,
    concurrency: int,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.prompt_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--num-warmups",
        str(args.num_warmups),
        "--temperature",
        "0",
        "--seed",
        str(args.seed),
        "--ignore-eos",
        "--percentile-metrics",
        ",".join(PERCENTILE_METRICS),
        "--metric-percentiles",
        ",".join(str(value) for value in PERCENTILES),
        "--save-result",
        "--save-detailed",
        "--result-filename",
        str(output_path),
        "--metadata",
        f"state_dtype={state_dtype}",
        "tensor_parallel_size=1",
        "prefix_caching=false",
        f"enforce_eager={str(args.enforce_eager).lower()}",
        "--disable-tqdm",
    ]


def _validate_result(
    result: dict[str, Any], args: argparse.Namespace, concurrency: int
) -> None:
    if result.get("failed") != 0 or result.get("completed") != args.num_prompts:
        raise AssertionError(
            f"B={concurrency}: completed={result.get('completed')}, "
            f"failed={result.get('failed')}"
        )
    if set(result.get("input_lens", [])) != {args.prompt_len}:
        raise AssertionError(f"B={concurrency}: unexpected input lengths")
    if set(result.get("output_lens", [])) != {args.output_len}:
        raise AssertionError(f"B={concurrency}: unexpected output lengths")


def _compact_result(result: dict[str, Any]) -> dict[str, object]:
    fields = [
        "duration",
        "completed",
        "failed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    ]
    fields.extend(
        f"p{percentile}_{metric}_ms"
        for metric in PERCENTILE_METRICS
        for percentile in PERCENTILES
    )
    return {field: result[field] for field in fields if field in result}


def _run_dtype(
    args: argparse.Namespace, state_dtype: str, output_dir: Path
) -> dict[str, object]:
    server_log_path = output_dir / f"server-{state_dtype}.log"
    server_command = _server_command(args, state_dtype)
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            server_command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    base_url = f"http://{args.host}:{args.port}"
    run: dict[str, object] = {
        "state_dtype": state_dtype,
        "server_command": server_command,
        "server_log": str(server_log_path),
        "gpu_before_server": _gpu_snapshot(),
        "benchmarks": [],
    }
    try:
        _wait_for_server(server, base_url, server_log_path, args.server_ready_timeout)
        run["gpu_after_server_ready"] = _gpu_snapshot()
        metrics_text = _fetch_text(f"{base_url}/metrics", timeout=15)
        metrics_path = output_dir / f"metrics-{state_dtype}.prom"
        metrics_path.write_text(metrics_text, encoding="utf-8")
        run["cache_config"] = _parse_cache_config_metrics(metrics_text)
        run["metrics_snapshot"] = str(metrics_path)

        benchmark_results = []
        for concurrency in args.concurrencies:
            result_path = output_dir / f"{state_dtype}-c{concurrency}.json"
            client_log_path = output_dir / f"client-{state_dtype}-c{concurrency}.log"
            client_command = _client_command(
                args, state_dtype, concurrency, result_path
            )
            memory = _run_client(client_command, client_log_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            _validate_result(result, args, concurrency)
            benchmark_results.append(
                {
                    "concurrency": concurrency,
                    "result_file": str(result_path),
                    "client_log": str(client_log_path),
                    "client_command": client_command,
                    "metrics": _compact_result(result),
                    "gpu_memory": memory,
                }
            )
        run["benchmarks"] = benchmark_results
    finally:
        _stop_process_group(server)
        time.sleep(3)
        run["gpu_after_server_stop"] = _gpu_snapshot()

    server_text = server_log_path.read_text(encoding="utf-8")
    run["capacity_from_server_log"] = _parse_capacity_log(server_text)
    return run


def _ratio(numerator: object, denominator: object) -> float | None:
    if not isinstance(numerator, (int, float)):
        return None
    if not isinstance(denominator, (int, float)) or denominator == 0:
        return None
    return float(numerator / denominator)


def _comparison(runs: list[dict[str, object]]) -> dict[str, object]:
    by_dtype = {str(run["state_dtype"]): run for run in runs}
    fp32 = by_dtype["float32"]
    fp8 = by_dtype["fp8_e4m3fn"]
    fp32_cases = {
        int(case["concurrency"]): case
        for case in fp32["benchmarks"]
        if isinstance(case, dict)
    }
    fp8_cases = {
        int(case["concurrency"]): case
        for case in fp8["benchmarks"]
        if isinstance(case, dict)
    }
    rows = []
    for concurrency in sorted(fp32_cases.keys() & fp8_cases.keys()):
        fp32_metrics = fp32_cases[concurrency]["metrics"]
        fp8_metrics = fp8_cases[concurrency]["metrics"]
        assert isinstance(fp32_metrics, dict) and isinstance(fp8_metrics, dict)
        speedups = {
            "request_throughput": _ratio(
                fp8_metrics.get("request_throughput"),
                fp32_metrics.get("request_throughput"),
            ),
            "output_throughput": _ratio(
                fp8_metrics.get("output_throughput"),
                fp32_metrics.get("output_throughput"),
            ),
        }
        for metric in PERCENTILE_METRICS:
            for percentile in PERCENTILES:
                key = f"p{percentile}_{metric}_ms"
                speedups[key] = _ratio(fp32_metrics.get(key), fp8_metrics.get(key))
        rows.append(
            {
                "concurrency": concurrency,
                "float32": fp32_metrics,
                "fp8_e4m3fn": fp8_metrics,
                "fp8_over_float32_throughput_or_float32_over_fp8_latency": speedups,
            }
        )

    fp32_cache = fp32.get("cache_config", {})
    fp8_cache = fp8.get("cache_config", {})
    capacity: dict[str, object] = {}
    if isinstance(fp32_cache, dict) and isinstance(fp8_cache, dict):
        for key in (
            "num_gpu_blocks",
            "kv_cache_size_tokens",
            "kv_cache_max_concurrency",
        ):
            try:
                fp32_value = float(fp32_cache[key])
                fp8_value = float(fp8_cache[key])
            except (KeyError, TypeError, ValueError):
                continue
            capacity[key] = {
                "float32": fp32_value,
                "fp8_e4m3fn": fp8_value,
                "fp8_over_float32": _ratio(fp8_value, fp32_value),
            }
    return {"serving": rows, "capacity": capacity}


def main(args: argparse.Namespace) -> None:
    if args.num_prompts < max(args.concurrencies):
        raise ValueError("--num-prompts must be at least the largest concurrency")
    if len(set(args.concurrencies)) != len(args.concurrencies):
        raise ValueError("--concurrencies must be unique")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "timestamp": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_revision": os.environ.get("QWEN38_BENCH_GIT_REVISION"),
        "gpu_initial": _gpu_snapshot(),
        "vllm_enable_v1_multiprocessing": os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        ),
        "vllm_enable_fla_packed_recurrent_decode": os.environ.get(
            "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"
        ),
    }
    settings = {
        "model": args.model,
        "served_model_name": args.served_model_name,
        "controlled_variable": "mamba_ssm_cache_dtype",
        "state_dtypes": list(STATE_DTYPES),
        "model_dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "num_prompts_per_concurrency": args.num_prompts,
        "concurrencies": args.concurrencies,
        "request_rate": "inf",
        "num_warmups": args.num_warmups,
        "seed": args.seed,
        "ignore_eos": True,
        "temperature": 0,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(args.concurrencies),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prefix_caching": False,
        "enforce_eager": args.enforce_eager,
        "percentile_metrics": list(PERCENTILE_METRICS),
        "percentiles": list(PERCENTILES),
    }

    runs = []
    for state_dtype in STATE_DTYPES:
        print(f"Starting matched serving run: {state_dtype}", flush=True)
        runs.append(_run_dtype(args, state_dtype, output_dir))

    report = {
        "schema_version": 1,
        "environment": environment,
        "settings": settings,
        "runs": runs,
        "comparison": _comparison(runs),
    }
    output_path = output_dir / "comparison.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2), flush=True)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--served-model-name", default="qwen3.8-27b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument(
        "--concurrencies", nargs="+", type=int, default=[1, 32, 64, 128]
    )
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--server-ready-timeout", type=float, default=1800)
    parser.add_argument("--enforce-eager", action="store_true")
    main(parser.parse_args())
