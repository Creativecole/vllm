# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

app = modal.App("qwen38-gdn-online-serving")

MODEL_PATH = "/models/Qwen3.8-27B"
REMOTE_VLLM = "/usr/local/lib/python3.12/site-packages/vllm"
OVERLAY_DIR = "/opt/qwen38-gdn-online-serving"
RESULT_ROOT = "/results/qwen38-gdn-online-serving"

VLLM_FILES = [
    "config/cache.py",
    "utils/torch_utils.py",
    "model_executor/layers/mamba/mamba_utils.py",
    "model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py",
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "model_executor/models/olmo_hybrid.py",
    "model_executor/models/qwen3_5.py",
    "model_executor/models/qwen3_next.py",
    "model_executor/warmup/qwen_triton_warmup.py",
    "third_party/flash_linear_attention/ops/fused_recurrent.py",
]

model_volume = modal.Volume.from_name("qwen-models", create_if_missing=False)
result_volume = modal.Volume.from_name("qwen38-profile-results", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "ninja-build", "cmake")
    .pip_install("vllm==0.27.1")
    .env(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "1",
        }
    )
)

if modal.is_local():
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        cwd=repo,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    image = image.env({"QWEN38_BENCH_GIT_REVISION": revision})

    for rel in VLLM_FILES:
        src = repo / "vllm" / rel
        if not src.is_file():
            raise RuntimeError(f"Missing vLLM overlay: {src}")
        image = image.add_local_file(str(src), f"{OVERLAY_DIR}/vllm/{rel}", copy=True)

    benchmark = repo / "benchmarks/qwen3_8_gdn_online_serving.py"
    if not benchmark.is_file():
        raise RuntimeError(f"Missing serving benchmark: {benchmark}")
    image = image.add_local_file(
        str(benchmark), f"{OVERLAY_DIR}/benchmark.py", copy=True
    )

    copy_commands = [
        f"cp -f '{OVERLAY_DIR}/vllm/{rel}' '{REMOTE_VLLM}/{rel}'" for rel in VLLM_FILES
    ]
    image = image.run_commands(
        *copy_commands,
        f"find '{REMOTE_VLLM}' -name '*.pyc' -delete",
        f"grep -q '_packed_decode_launch_config' "
        f"'{REMOTE_VLLM}/third_party/flash_linear_attention/ops/fused_recurrent.py'",
        f"grep -q '_prepare_fp8_prefill_initial_state' "
        f"'{REMOTE_VLLM}/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'",
    )


@app.function(
    image=image,
    gpu="H100!",
    volumes={"/models": model_volume, "/results": result_volume},
    timeout=14400,
    memory=32768,
)
def run(
    model_path: str = MODEL_PATH,
    num_prompts: int = 128,
    concurrencies: list[int] | None = None,
    enforce_eager: bool = False,
) -> dict[str, object]:
    if concurrencies is None:
        concurrencies = [1, 32, 64, 128]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    result_dir = f"{RESULT_ROOT}/{timestamp}"

    command = [
        sys.executable,
        f"{OVERLAY_DIR}/benchmark.py",
        "--model",
        model_path,
        "--output-dir",
        result_dir,
        "--prompt-len",
        "512",
        "--output-len",
        "128",
        "--num-prompts",
        str(num_prompts),
        "--concurrencies",
        *[str(value) for value in concurrencies],
        "--num-warmups",
        "2",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.9",
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=OVERLAY_DIR)

    comparison_path = f"{result_dir}/comparison.json"
    with open(comparison_path, encoding="utf-8") as report_file:
        report = json.load(report_file)
    result_volume.commit()
    return {
        "result_dir": result_dir,
        "comparison_path": comparison_path,
        "settings": report["settings"],
        "comparison": report["comparison"],
    }


@app.local_entrypoint()
def main(
    model_path: str = MODEL_PATH,
    num_prompts: int = 128,
    concurrencies: str = "1,32,64,128",
    enforce_eager: bool = False,
) -> None:
    parsed_concurrencies = [int(value) for value in concurrencies.split(",")]
    result = run.remote(
        model_path=model_path,
        num_prompts=num_prompts,
        concurrencies=parsed_concurrencies,
        enforce_eager=enforce_eager,
    )
    print(json.dumps(result, indent=2))
