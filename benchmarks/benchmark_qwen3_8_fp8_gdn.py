# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched Qwen3.8 FP8 GDN end-to-end throughput benchmark."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from vllm import LLM, SamplingParams


def _make_prompt_ids(llm: LLM, prompt_len: int) -> list[int]:
    seed_text = (
        "Large language model inference uses GPU memory bandwidth, attention, "
        "recurrent state, KV cache, continuous batching, PagedAttention, "
        "tensor cores, CUDA kernels, Triton kernels, quantization, and "
        "optimized serving systems. "
    )
    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(seed_text * 300, add_special_tokens=False)
    if len(token_ids) < prompt_len:
        raise ValueError(f"Generated only {len(token_ids)} prompt tokens")
    return list(token_ids[:prompt_len])


def _fp8_cache_summary(llm: LLM) -> dict[str, object]:
    context = llm.llm_engine.vllm_config.compilation_config.static_forward_context
    caches = []
    for layer in context.values():
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, (tuple, list)) or len(cache) != 3:
            continue
        state, scales = cache[1], cache[2]
        if isinstance(state, torch.Tensor) and state.dtype == torch.float8_e4m3fn:
            caches.append((state, scales))
    if len(caches) != 48:
        raise AssertionError(f"Expected 48 FP8 GDN layers, found {len(caches)}")
    state, scales = caches[0]
    return {
        "num_layers": len(caches),
        "state_dtype": str(state.dtype),
        "scale_dtype": str(scales.dtype),
        "state_shape": list(state.shape),
        "scale_shape": list(scales.shape),
    }


def _run_batch(
    llm: LLM,
    prompt_ids: list[int],
    batch_size: int,
    output_len: int,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    prompts = [{"prompt_token_ids": list(prompt_ids)} for _ in range(batch_size)]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
        seed=0,
    )

    for _ in range(warmup):
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        if any(len(output.outputs[0].token_ids) != output_len for output in outputs):
            raise AssertionError("Warmup generation returned the wrong token count")

    durations = []
    outputs = None
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        if any(len(output.outputs[0].token_ids) != output_len for output in outputs):
            raise AssertionError("Measured generation returned the wrong token count")

    assert outputs is not None
    generated_tokens = batch_size * output_len
    throughputs = [generated_tokens / duration for duration in durations]
    digest_payload = [output.outputs[0].token_ids for output in outputs]
    token_digest = hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "batch_size": batch_size,
        "durations_s": durations,
        "output_tokens_per_second": throughputs,
        "mean_output_tokens_per_second": statistics.mean(throughputs),
        "median_output_tokens_per_second": statistics.median(throughputs),
        "token_digest": token_digest,
    }


def main(args: argparse.Namespace) -> None:
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        mamba_ssm_cache_dtype="fp8_e4m3fn",
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.batch_sizes),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    prompt_ids = _make_prompt_ids(llm, args.prompt_len)
    cache_summary = _fp8_cache_summary(llm)
    graph_mode = str(llm.llm_engine.vllm_config.compilation_config.cudagraph_mode)

    results = []
    for batch_size in args.batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        result = _run_batch(
            llm,
            prompt_ids,
            batch_size,
            args.output_len,
            args.warmup,
            args.repeat,
        )
        results.append(result)
        print(
            f"B={batch_size:3d} "
            f"mean={result['mean_output_tokens_per_second']:.2f} tok/s "
            f"median={result['median_output_tokens_per_second']:.2f} tok/s",
            flush=True,
        )

    report = {
        "stage": args.stage_label,
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": {
            "prompt_len": args.prompt_len,
            "output_len": args.output_len,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "enforce_eager": args.enforce_eager,
            "resolved_cudagraph_mode": graph_mode,
        },
        "cache": cache_summary,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    main(parser.parse_args())
