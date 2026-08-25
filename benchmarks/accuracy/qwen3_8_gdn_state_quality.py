# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Qwen3.8 generation with FP32 and FP8 GDN state storage."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

GSM8K_URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl",
}
GSM8K_TEST_SIZE = 1319
PARITY_MAX_TOKENS = 256
GSM8K_MAX_TOKENS = 256
GSM8K_SHOTS = 5
STATE_DTYPES = ("float32", "fp8_e4m3fn")
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

_PARITY_CASES = (
    (
        "arithmetic",
        "Compute the final account balance and explain each operation.",
        "The account starts at 840 dollars, receives 125 dollars, then pays "
        "three invoices of 47 dollars each.",
    ),
    (
        "programming",
        "Write a Python function and briefly justify its time complexity.",
        "The input is a list of signed integers and the result must preserve "
        "the first occurrence order while removing duplicates.",
    ),
    (
        "physics",
        "Derive the requested quantity using the stated assumptions.",
        "A cart begins from rest, accelerates uniformly at two meters per "
        "second squared, and friction is negligible.",
    ),
    (
        "biology",
        "Explain the mechanism in precise but accessible language.",
        "A plant is moved from bright indirect light to a dim room while "
        "temperature, water, and soil nutrients remain constant.",
    ),
    (
        "history",
        "Give a concise causal analysis and distinguish causes from triggers.",
        "A trading city faces a failed harvest, rising food prices, a disputed "
        "succession, and a newly imposed port tax.",
    ),
    (
        "literature",
        "Analyze the narrator's reliability using evidence from the passage.",
        "The narrator insists that every clock is wrong, yet records meetings "
        "at exact minutes and contradicts the diary found later.",
    ),
    (
        "logic",
        "Determine whether the conclusion follows and show the reasoning.",
        "Every copper token is marked, no marked token is transparent, and "
        "some objects in the box are transparent.",
    ),
    (
        "data-analysis",
        "Summarize the trend and identify one conclusion the data cannot support.",
        "Weekly observations are 18, 21, 21, 25, 24, 29, and 31, while the "
        "collection method changes after the fourth observation.",
    ),
    (
        "editing",
        "Rewrite the passage for clarity while retaining every factual claim.",
        "The committee met on Tuesday, reviewed four proposals, deferred two, "
        "and approved the remaining pair subject to legal review.",
    ),
    (
        "planning",
        "Produce a prioritized plan with dependencies and a fallback.",
        "A small team has two days to move a service, verify backups, notify "
        "users, and preserve a four-hour rollback window.",
    ),
    (
        "translation",
        "Translate the quoted sentence into natural English and explain one idiom.",
        "The source sentence says that after the rain the road was difficult, "
        "but the travelers kept their promise and arrived before dusk.",
    ),
    (
        "classification",
        "Assign the report to one category and state the decisive evidence.",
        "The device powers on, passes its memory test, reaches the login screen, "
        "and loses network connectivity only after waking from sleep.",
    ),
    (
        "economics",
        "Explain the likely first-order effects without assuming perfect markets.",
        "A town caps apartment rents while construction costs rise and the "
        "number of households seeking leases grows.",
    ),
    (
        "probability",
        "Calculate the probability and name any independence assumption used.",
        "A bag contains five red, four blue, and three green counters; two are "
        "drawn without replacement.",
    ),
    (
        "systems",
        "Diagnose the most likely bottleneck and propose two measurements.",
        "Request latency rises with concurrency, CPU utilization stays low, GPU "
        "memory is stable, and device memory bandwidth approaches saturation.",
    ),
    (
        "summarization",
        "Return a three-sentence summary separating observations from inference.",
        "A field survey recorded fewer birds near the road, more insects after "
        "rain, and incomplete counts on the two windiest mornings.",
    ),
)
_PARITY_CONTEXT_REPETITIONS = (1, 2, 4, 6, 10, 16)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _git_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        return {
            "revision": revision,
            "dirty": dirty,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "tracked_diff_sha256": None}


def _environment(torch_module: Any, vllm_version: str) -> dict[str, Any]:
    cuda_available = torch_module.cuda.is_available()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "vllm": vllm_version,
        "torch": torch_module.__version__,
        "cuda_runtime": torch_module.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch_module.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": (
            list(torch_module.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "relevant_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": os.environ.get(
                "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"
            ),
        },
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git": _git_revision(),
    }


def build_parity_prompts() -> list[dict[str, str | int]]:
    """Build a fixed 96-prompt set spanning domains and context lengths."""
    prompts: list[dict[str, str | int]] = []
    for case_index, (domain, instruction, context) in enumerate(_PARITY_CASES):
        for length_index, repetitions in enumerate(_PARITY_CONTEXT_REPETITIONS):
            repeated_context = " ".join([context] * repetitions)
            prompt = (
                f"Domain: {domain}.\n"
                f"Context: {repeated_context}\n"
                f"Task: {instruction}\n"
                "Give a self-contained answer."
            )
            prompts.append(
                {
                    "id": f"parity-{case_index:02d}-{length_index:02d}",
                    "domain": domain,
                    "length_variant": length_index,
                    "prompt": prompt,
                }
            )
    if not 64 <= len(prompts) <= 128:
        raise AssertionError(f"Expected 64-128 parity prompts, got {len(prompts)}")
    return prompts


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {path}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            records.append(json.loads(line))
    return records


def load_gsm8k(
    data_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    paths = {split: data_dir / f"{split}.jsonl" for split in GSM8K_URLS}
    for split, url in GSM8K_URLS.items():
        _download_if_missing(url, paths[split])
    hashes = {
        split: hashlib.sha256(path.read_bytes()).hexdigest()
        for split, path in paths.items()
    }
    return (
        _read_jsonl(paths["train"]),
        _read_jsonl(paths["test"]),
        {
            "urls": GSM8K_URLS,
            "sha256": hashes,
        },
    )


def _build_gsm8k_prompts(
    train_data: list[dict[str, str]],
    test_data: list[dict[str, str]],
    limit: int,
) -> list[dict[str, Any]]:
    if len(train_data) < GSM8K_SHOTS:
        raise ValueError("GSM8K train data does not contain five examples")
    if not 1 <= limit <= len(test_data):
        raise ValueError(f"--limit must be between 1 and {len(test_data)}")

    few_shot = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
        for row in train_data[:GSM8K_SHOTS]
    )
    examples = []
    for index, row in enumerate(test_data[:limit]):
        prompt = few_shot + f"Question: {row['question']}\nAnswer:"
        examples.append(
            {
                "id": f"gsm8k-test-{index:04d}",
                "dataset_index": index,
                "question": row["question"],
                "gold_answer_text": row["answer"],
                "prompt": prompt,
            }
        )
    return examples


def extract_numeric_answer(text: str) -> str | None:
    """Extract and normalize a deterministic final numeric answer."""
    answer_region = text.rsplit("####", maxsplit=1)[-1] if "####" in text else text
    matches = _NUMBER_RE.findall(answer_region)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _token_failure_reason(
    token_ids: list[Any], expected_length: int | None
) -> str | None:
    if not token_ids:
        return "empty_generation"
    for token_id in token_ids:
        if isinstance(token_id, float) and not math.isfinite(token_id):
            return "non_finite_token_id"
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            return "non_integer_token_id"
        if token_id < 0:
            return "negative_token_id"
    if expected_length is not None and len(token_ids) != expected_length:
        return "unexpected_token_count"
    return None


def _completion_record(output: Any, expected_length: int | None) -> dict[str, Any]:
    if not output.outputs:
        return {
            "output_text": "",
            "output_token_ids": [],
            "output_token_count": 0,
            "finish_reason": None,
            "invalid_reason": "missing_completion",
        }
    completion = output.outputs[0]
    token_ids = list(completion.token_ids)
    return {
        "output_text": completion.text,
        "output_token_ids": token_ids,
        "output_token_count": len(token_ids),
        "finish_reason": completion.finish_reason,
        "invalid_reason": _token_failure_reason(token_ids, expected_length),
    }


def _run_parity(llm: Any, sampling_params_cls: Any, seed: int) -> dict[str, Any]:
    prompt_records = build_parity_prompts()
    prompts = [str(record["prompt"]) for record in prompt_records]
    tokenizer = llm.get_tokenizer()
    sampling = sampling_params_cls(
        temperature=0.0,
        max_tokens=PARITY_MAX_TOKENS,
        ignore_eos=True,
        seed=seed,
    )
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(prompt_records):
        raise RuntimeError(
            f"Expected {len(prompt_records)} parity outputs, got {len(outputs)}"
        )

    examples = []
    for prompt_record, output in zip(prompt_records, outputs, strict=True):
        completion = _completion_record(output, PARITY_MAX_TOKENS)
        examples.append(
            {
                **prompt_record,
                "prompt_sha256": hashlib.sha256(
                    str(prompt_record["prompt"]).encode("utf-8")
                ).hexdigest(),
                "input_token_count": len(tokenizer.encode(prompt_record["prompt"])),
                **completion,
            }
        )

    invalid = [row for row in examples if row["invalid_reason"] is not None]
    non_finite = [
        row for row in invalid if row["invalid_reason"] == "non_finite_token_id"
    ]
    return {
        "settings": {
            "prompt_count": len(examples),
            "prompt_set_sha256": _sha256_json(
                [(row["id"], row["prompt"]) for row in prompt_records]
            ),
            "temperature": 0.0,
            "max_tokens": PARITY_MAX_TOKENS,
            "ignore_eos": True,
            "seed": seed,
        },
        "examples": examples,
        "aggregate": {
            "num_examples": len(examples),
            "total_generated_tokens": sum(
                int(row["output_token_count"]) for row in examples
            ),
            "obviously_invalid_failure_count": len(invalid),
            "non_finite_failure_count": len(non_finite),
            "elapsed_seconds": elapsed,
        },
    }


def _run_gsm8k(
    llm: Any,
    sampling_params_cls: Any,
    seed: int,
    limit: int,
    train_data: list[dict[str, str]],
    test_data: list[dict[str, str]],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    prompt_records = _build_gsm8k_prompts(train_data, test_data, limit)
    prompts = [row["prompt"] for row in prompt_records]
    tokenizer = llm.get_tokenizer()
    sampling = sampling_params_cls(
        temperature=0.0,
        max_tokens=GSM8K_MAX_TOKENS,
        ignore_eos=False,
        stop=["\n\nQuestion:"],
        seed=seed,
    )
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(prompt_records):
        raise RuntimeError(
            f"Expected {len(prompt_records)} GSM8K outputs, got {len(outputs)}"
        )

    examples = []
    for prompt_record, output in zip(prompt_records, outputs, strict=True):
        completion = _completion_record(output, None)
        gold = extract_numeric_answer(prompt_record["gold_answer_text"])
        prediction = extract_numeric_answer(completion["output_text"])
        if gold is None:
            raise ValueError(f"Could not extract gold answer for {prompt_record['id']}")
        examples.append(
            {
                "id": prompt_record["id"],
                "dataset_index": prompt_record["dataset_index"],
                "question": prompt_record["question"],
                "prompt_sha256": hashlib.sha256(
                    prompt_record["prompt"].encode("utf-8")
                ).hexdigest(),
                "input_token_count": len(tokenizer.encode(prompt_record["prompt"])),
                "gold_answer": gold,
                "extracted_answer": prediction,
                "correct": prediction == gold,
                **completion,
            }
        )

    correct = sum(bool(row["correct"]) for row in examples)
    invalid_answers = sum(row["extracted_answer"] is None for row in examples)
    return {
        "settings": {
            "dataset": "openai/grade-school-math GSM8K test",
            "dataset_files": dataset,
            "selection": "first N test examples in source order",
            "few_shot_selection": "first 5 train examples in source order",
            "num_questions": len(examples),
            "num_shots": GSM8K_SHOTS,
            "temperature": 0.0,
            "max_tokens": GSM8K_MAX_TOKENS,
            "ignore_eos": False,
            "stop": ["\n\nQuestion:"],
            "seed": seed,
            "answer_extraction": (
                "Use text after the final #### marker when present; otherwise "
                "use the final signed integer or decimal; remove thousands "
                "separators and normalize with Decimal."
            ),
            "question_set_sha256": _sha256_json(
                [(row["id"], row["question"]) for row in prompt_records]
            ),
        },
        "examples": examples,
        "aggregate": {
            "num_questions": len(examples),
            "correct": correct,
            "exact_match_accuracy": correct / len(examples),
            "invalid_answer_count": invalid_answers,
            "total_generated_tokens": sum(
                int(row["output_token_count"]) for row in examples
            ),
            "elapsed_seconds": elapsed,
        },
    }


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_parity(
    fp32_task: dict[str, Any], fp8_task: dict[str, Any]
) -> dict[str, Any]:
    if fp32_task["settings"] != fp8_task["settings"]:
        raise ValueError("Parity task settings or prompt sets differ")
    fp32_rows = {row["id"]: row for row in fp32_task["examples"]}
    fp8_rows = {row["id"]: row for row in fp8_task["examples"]}
    if fp32_rows.keys() != fp8_rows.keys():
        raise ValueError("Parity example IDs differ")

    paired = []
    identical = 0
    aligned_matches = 0
    aligned_positions = 0
    divergence_positions = []
    for example_id in fp32_rows:
        fp32_ids = fp32_rows[example_id]["output_token_ids"]
        fp8_ids = fp8_rows[example_id]["output_token_ids"]
        aligned = min(len(fp32_ids), len(fp8_ids))
        matches = sum(
            left == right
            for left, right in zip(fp32_ids[:aligned], fp8_ids[:aligned], strict=True)
        )
        divergence = _first_divergence(fp32_ids, fp8_ids)
        is_identical = divergence is None
        identical += is_identical
        aligned_matches += matches
        aligned_positions += aligned
        if divergence is not None:
            divergence_positions.append(divergence)
        paired.append(
            {
                "id": example_id,
                "fully_token_identical": is_identical,
                "aligned_positions": aligned,
                "aligned_token_matches": matches,
                "aligned_token_agreement_rate": matches / aligned if aligned else None,
                "first_divergence_position_zero_based": divergence,
                "fp32_output_token_count": len(fp32_ids),
                "fp8_output_token_count": len(fp8_ids),
                "fp32_invalid_reason": fp32_rows[example_id]["invalid_reason"],
                "fp8_invalid_reason": fp8_rows[example_id]["invalid_reason"],
            }
        )

    num_examples = len(paired)
    divergent = len(divergence_positions)
    return {
        "settings": fp32_task["settings"],
        "examples": paired,
        "aggregate": {
            "num_examples": num_examples,
            "fully_token_identical_count": identical,
            "fully_token_identical_percentage": 100 * identical / num_examples,
            "total_aligned_token_matches": aligned_matches,
            "total_aligned_token_positions": aligned_positions,
            "total_aligned_token_agreement_rate": (
                aligned_matches / aligned_positions if aligned_positions else None
            ),
            "divergent_example_count": divergent,
            "first_divergence_position_zero_based": {
                "mean": (
                    statistics.mean(divergence_positions)
                    if divergence_positions
                    else None
                ),
                "median": (
                    statistics.median(divergence_positions)
                    if divergence_positions
                    else None
                ),
                "p10": (
                    _percentile(divergence_positions, 0.10)
                    if len(divergence_positions) >= 10
                    else None
                ),
            },
            "fp32_obviously_invalid_failure_count": fp32_task["aggregate"][
                "obviously_invalid_failure_count"
            ],
            "fp8_obviously_invalid_failure_count": fp8_task["aggregate"][
                "obviously_invalid_failure_count"
            ],
            "fp32_non_finite_failure_count": fp32_task["aggregate"][
                "non_finite_failure_count"
            ],
            "fp8_non_finite_failure_count": fp8_task["aggregate"][
                "non_finite_failure_count"
            ],
        },
    }


def compare_gsm8k(
    fp32_task: dict[str, Any], fp8_task: dict[str, Any]
) -> dict[str, Any]:
    if fp32_task["settings"] != fp8_task["settings"]:
        raise ValueError("GSM8K task settings or question sets differ")
    fp32_rows = {row["id"]: row for row in fp32_task["examples"]}
    fp8_rows = {row["id"]: row for row in fp8_task["examples"]}
    if fp32_rows.keys() != fp8_rows.keys():
        raise ValueError("GSM8K example IDs differ")

    paired = []
    changed_correctness = 0
    for example_id in fp32_rows:
        fp32_row = fp32_rows[example_id]
        fp8_row = fp8_rows[example_id]
        differs = bool(fp32_row["correct"]) != bool(fp8_row["correct"])
        changed_correctness += differs
        paired.append(
            {
                "id": example_id,
                "question": fp32_row["question"],
                "gold_answer": fp32_row["gold_answer"],
                "fp32_extracted_answer": fp32_row["extracted_answer"],
                "fp8_extracted_answer": fp8_row["extracted_answer"],
                "fp32_correct": fp32_row["correct"],
                "fp8_correct": fp8_row["correct"],
                "correctness_differs": differs,
            }
        )

    fp32_accuracy = fp32_task["aggregate"]["exact_match_accuracy"]
    fp8_accuracy = fp8_task["aggregate"]["exact_match_accuracy"]
    return {
        "settings": fp32_task["settings"],
        "examples": paired,
        "aggregate": {
            "num_questions": len(paired),
            "fp32_exact_match_accuracy": fp32_accuracy,
            "fp8_exact_match_accuracy": fp8_accuracy,
            "accuracy_difference_percentage_points_fp8_minus_fp32": 100
            * (fp8_accuracy - fp32_accuracy),
            "absolute_accuracy_difference_percentage_points": 100
            * abs(fp8_accuracy - fp32_accuracy),
            "correctness_differs_count": changed_correctness,
        },
    }


def _validate_controlled_runs(
    fp32_run: dict[str, Any], fp8_run: dict[str, Any]
) -> None:
    if fp32_run["state_dtype"] != "float32":
        raise ValueError("--fp32-run must have state_dtype=float32")
    if fp8_run["state_dtype"] != "fp8_e4m3fn":
        raise ValueError("--fp8-run must have state_dtype=fp8_e4m3fn")
    for key in (
        "path_or_id",
        "model_dtype",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "enforce_eager",
        "seed",
    ):
        if fp32_run["model"][key] != fp8_run["model"][key]:
            raise ValueError(f"Controlled setting differs: model.{key}")
    for key in (
        "platform",
        "vllm",
        "torch",
        "cuda_runtime",
        "gpu",
        "gpu_capability",
        "relevant_environment",
        "evaluator_sha256",
    ):
        if fp32_run["environment"][key] != fp8_run["environment"][key]:
            raise ValueError(f"Controlled setting differs: environment.{key}")
    for key in ("revision", "tracked_diff_sha256"):
        if fp32_run["environment"]["git"][key] != fp8_run["environment"]["git"][key]:
            raise ValueError(f"Controlled setting differs: environment.git.{key}")
    for key in (
        "num_gdn_layers",
        "conv_state_dtype",
        "conv_state_shape_per_block",
        "ssm_state_shape_per_block",
        "scale_dtype",
        "scale_shape_per_block",
    ):
        if fp32_run["model"]["gdn_cache"][key] != fp8_run["model"]["gdn_cache"][key]:
            raise ValueError(f"Controlled setting differs: model.gdn_cache.{key}")


def _gdn_cache_summary(llm: Any, torch_module: Any, state_dtype: str) -> dict[str, Any]:
    context = llm.llm_engine.vllm_config.compilation_config.static_forward_context
    caches = []
    for layer in context.values():
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, (tuple, list)) or len(cache) != 3:
            continue
        conv_state, ssm_state, scales = cache
        if not all(
            isinstance(tensor, torch_module.Tensor)
            for tensor in (conv_state, ssm_state, scales)
        ):
            continue
        if ssm_state.ndim == 4 and scales.ndim == 4:
            caches.append((conv_state, ssm_state, scales))
    if len(caches) != 48:
        raise AssertionError(f"Expected 48 GDN caches, found {len(caches)}")

    expected_dtype = {
        "float32": torch_module.float32,
        "fp8_e4m3fn": torch_module.float8_e4m3fn,
    }[state_dtype]
    for _, ssm_state, scales in caches:
        if ssm_state.dtype != expected_dtype:
            raise AssertionError(
                f"Expected {expected_dtype} state, found {ssm_state.dtype}"
            )
        if scales.dtype != torch_module.float32:
            raise AssertionError(f"Expected FP32 scales, found {scales.dtype}")

    conv_state, ssm_state, scales = caches[0]
    return {
        "num_gdn_layers": len(caches),
        "num_blocks": ssm_state.shape[0],
        "conv_state_dtype": str(conv_state.dtype),
        "conv_state_shape_per_block": list(conv_state.shape[1:]),
        "ssm_state_dtype": str(ssm_state.dtype),
        "ssm_state_shape_per_block": list(ssm_state.shape[1:]),
        "scale_dtype": str(scales.dtype),
        "scale_shape_per_block": list(scales.shape[1:]),
    }


def run_evaluation(args: argparse.Namespace) -> None:
    import torch

    import vllm
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen3.8 state-quality evaluation requires CUDA")
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("The validated state-quality target is Hopper SM90")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The validated state-quality target is NVIDIA H100")

    gsm8k_data = None
    if "gsm8k" in args.tasks:
        gsm8k_data = load_gsm8k(args.gsm8k_data_dir)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        mamba_ssm_cache_dtype=args.state_dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen3_8_gdn_state_quality_run",
        "environment": _environment(torch, vllm.__version__),
        "model": {
            "path_or_id": args.model,
            "model_dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "mamba_ssm_cache_dtype": args.state_dtype,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": False,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
            "gdn_cache": _gdn_cache_summary(llm, torch, args.state_dtype),
        },
        "state_dtype": args.state_dtype,
        "tasks": {},
    }
    try:
        if "parity" in args.tasks:
            report["tasks"]["generation_parity"] = _run_parity(
                llm, SamplingParams, args.seed
            )
        if "gsm8k" in args.tasks:
            assert gsm8k_data is not None
            report["tasks"]["gsm8k"] = _run_gsm8k(
                llm,
                SamplingParams,
                args.seed,
                args.limit,
                *gsm8k_data,
            )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    _write_json(args.output, report)
    print(f"Wrote {args.output}", flush=True)


def compare_runs(args: argparse.Namespace) -> None:
    fp32_run = json.loads(args.fp32_run.read_text())
    fp8_run = json.loads(args.fp8_run.read_text())
    _validate_controlled_runs(fp32_run, fp8_run)
    if fp32_run["tasks"].keys() != fp8_run["tasks"].keys():
        raise ValueError("Input reports contain different task sets")
    common_tasks = fp32_run["tasks"].keys()

    tasks = {}
    if "generation_parity" in common_tasks:
        tasks["generation_parity"] = compare_parity(
            fp32_run["tasks"]["generation_parity"],
            fp8_run["tasks"]["generation_parity"],
        )
    if "gsm8k" in common_tasks:
        tasks["gsm8k"] = compare_gsm8k(
            fp32_run["tasks"]["gsm8k"], fp8_run["tasks"]["gsm8k"]
        )

    report = {
        "schema_version": 1,
        "kind": "qwen3_8_gdn_state_quality_comparison",
        "created_utc": datetime.now(UTC).isoformat(),
        "runs": {
            "fp32": {
                "path": str(args.fp32_run),
                "environment": fp32_run["environment"],
                "model": fp32_run["model"],
                "state_dtype": fp32_run["state_dtype"],
            },
            "fp8": {
                "path": str(args.fp8_run),
                "environment": fp8_run["environment"],
                "model": fp8_run["model"],
                "state_dtype": fp8_run["state_dtype"],
            },
        },
        "controlled_variable": "mamba_ssm_cache_dtype",
        "tasks": tasks,
        "interpretation": (
            "Token agreement measures deterministic generation sensitivity, not "
            "model quality. GSM8K exact match is the downstream accuracy measure."
        ),
    }
    _write_json(args.output, report)
    print(
        json.dumps({name: task["aggregate"] for name, task in tasks.items()}, indent=2)
    )
    print(f"Wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run one recurrent-state dtype in one model process"
    )
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--state-dtype", choices=STATE_DTYPES, required=True)
    run_parser.add_argument(
        "--tasks", nargs="+", choices=("parity", "gsm8k"), default=["parity", "gsm8k"]
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=GSM8K_TEST_SIZE,
        help="Number of GSM8K test questions; use 50 for smoke or 1319 for full",
    )
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--max-model-len", type=int, default=4096)
    run_parser.add_argument("--max-num-seqs", type=int, default=32)
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    run_parser.add_argument("--enforce-eager", action="store_true")
    run_parser.add_argument(
        "--gsm8k-data-dir",
        type=Path,
        default=Path.home() / ".cache" / "vllm" / "gsm8k",
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(func=run_evaluation)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare completed FP32 and FP8 JSON reports"
    )
    compare_parser.add_argument("--fp32-run", type=Path, required=True)
    compare_parser.add_argument("--fp8-run", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(func=compare_runs)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
