# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune the fused FP8 GDN packed-decode kernel for Qwen3.8-27B on H100."""

import argparse
import itertools
import json
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    _packed_decode_launch_config,
    fused_recurrent_gated_delta_rule_packed_decode,
)

H = 16
HV = 48
K = 128
V = 128
QKV_DIM = 2 * H * K + HV * V
BATCH_SIZES = (1, 32, 64, 128)
BV_CHOICES = (16, 32)
NUM_WARPS_CHOICES = (1, 2, 4)
NUM_STAGES_CHOICES = (1, 2, 3)


def _quantize_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = state.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
    return (state / scales).to(torch.float8_e4m3fn), scales


def _make_inputs(batch_size: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(1234 + batch_size)
    device = torch.device("cuda")
    num_states = batch_size + 1
    base_state = (
        torch.randn(
            num_states,
            HV,
            V,
            K,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    state, scales = _quantize_state(base_state)
    return {
        "mixed_qkv": (
            torch.randn(
                batch_size,
                QKV_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.05
        ).contiguous(),
        "a": (
            torch.randn(batch_size, HV, dtype=torch.bfloat16, device=device) * 0.05
        ).contiguous(),
        "b": (
            torch.randn(batch_size, HV, dtype=torch.bfloat16, device=device) * 0.05
        ).contiguous(),
        "A_log": (
            torch.randn(HV, dtype=torch.float32, device=device) * 0.05
        ).contiguous(),
        "dt_bias": (
            torch.randn(HV, dtype=torch.float32, device=device) * 0.05
        ).contiguous(),
        "state": state.contiguous(),
        "scales": scales.contiguous(),
        "state_indices": torch.arange(
            1, batch_size + 1, dtype=torch.int32, device=device
        ),
    }


def _run(
    inputs: dict[str, torch.Tensor],
    state: torch.Tensor,
    scales: torch.Tensor | None,
    out: torch.Tensor,
    _launch_config: tuple[int, int, int] | None,
) -> None:
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=inputs["mixed_qkv"],
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=K**-0.5,
        initial_state=state,
        out=out,
        ssm_state_indices=inputs["state_indices"],
        ssm_state_scales=scales,
        use_qk_l2norm_in_kernel=True,
        _launch_config=_launch_config,
    )


def _reference(
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    state = inputs["state"].float() * inputs["scales"]
    out = torch.empty(
        inputs["mixed_qkv"].shape[0],
        1,
        HV,
        V,
        dtype=torch.bfloat16,
        device="cuda",
    )
    _run(inputs, state, None, out, (32, 1, 3))
    return out, state


def _check_config(
    inputs: dict[str, torch.Tensor],
    reference_out: torch.Tensor,
    reference_state: torch.Tensor,
    _launch_config: tuple[int, int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = inputs["state"].clone()
    scales = inputs["scales"].clone()
    out = torch.empty_like(reference_out)
    _run(inputs, state, scales, out, _launch_config)
    torch.testing.assert_close(out, reference_out, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        state.float() * scales,
        reference_state,
        rtol=2e-2,
        atol=2e-2,
    )
    return state, scales, out


def _time_eager(call: Callable[[], None], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeat


def _time_cudagraph(call: Callable[[], None], warmup: int, repeat: int) -> float:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        graph.replay()
    end.record()
    end.synchronize()
    latency_us = start.elapsed_time(end) * 1000.0 / repeat
    graph.reset()
    return latency_us


def _benchmark_config(
    inputs: dict[str, torch.Tensor],
    state: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    _launch_config: tuple[int, int, int] | None,
    mode: str,
    warmup: int,
    repeat: int,
) -> float:
    def call() -> None:
        _run(inputs, state, scales, out, _launch_config)

    if mode == "eager":
        return _time_eager(call, warmup, repeat)
    return _time_cudagraph(call, warmup, repeat)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU.")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This benchmark targets Hopper SM90.")

    configs: tuple[tuple[int, int, int] | None, ...]
    if args.selected_only:
        configs = (None,)
    else:
        configs = tuple(
            itertools.product(BV_CHOICES, NUM_WARPS_CHOICES, NUM_STAGES_CHOICES)
        )
    results: list[dict[str, int | float | str]] = []
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"configs={len(configs)} modes={args.mode} batches={args.batch_sizes}")

    for batch_size in args.batch_sizes:
        inputs = _make_inputs(batch_size)
        reference_out, reference_state = _reference(inputs)
        torch.cuda.synchronize()
        for _launch_config in configs:
            resolved_config = _launch_config or _packed_decode_launch_config(
                batch_size, True
            )
            state, scales, out = _check_config(
                inputs, reference_out, reference_state, _launch_config
            )
            for mode in args.mode:
                latency_us = _benchmark_config(
                    inputs,
                    state,
                    scales,
                    out,
                    _launch_config,
                    mode,
                    args.warmup,
                    args.repeat,
                )
                row = {
                    "batch_size": batch_size,
                    "bv": resolved_config[0],
                    "num_warps": resolved_config[1],
                    "num_stages": resolved_config[2],
                    "mode": mode,
                    "latency_us": latency_us,
                }
                results.append(row)
                print(
                    f"B={batch_size:3d} mode={mode:9s} "
                    f"BV={resolved_config[0]:2d} w={resolved_config[1]} "
                    f"s={resolved_config[2]} {latency_us:9.3f} us"
                )
        del inputs, reference_out, reference_state
        torch.cuda.empty_cache()

    print("\nBest configurations")
    for batch_size in args.batch_sizes:
        for mode in args.mode:
            matching = [
                row
                for row in results
                if row["batch_size"] == batch_size and row["mode"] == mode
            ]
            best = min(matching, key=lambda row: float(row["latency_us"]))
            print(
                f"B={batch_size:3d} mode={mode:9s} "
                f"BV={best['bv']} w={best['num_warps']} s={best['num_stages']} "
                f"{best['latency_us']:.3f} us"
            )

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCH_SIZES))
    parser.add_argument(
        "--mode", nargs="+", choices=("eager", "cudagraph"), default=["eager"]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=str)
    parser.add_argument("--selected-only", action="store_true")
    main(parser.parse_args())
