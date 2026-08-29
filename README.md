# Qwen3.8 FP8 GDN recurrent-state cache for vLLM

This branch is an experimental vLLM v0.27.1 implementation of an FP8 E4M3
persistent recurrent-state cache for Qwen3.8-27B Gated DeltaNet (GDN) layers on
NVIDIA Hopper H100.

The project reduces recurrent-state memory and decode traffic while preserving
FP32 recurrent computation. It does not modify the attention KV cache.

> **Validated target:** Qwen3.8-27B, one NVIDIA H100 80GB HBM3, BF16 model and
> activations, tensor parallel size 1. The checkpoint declares
> `Qwen3_5ForConditionalGeneration` with a `qwen3_5_text` configuration.

## What this branch contains

- FP8 E4M3 physical GDN recurrent state with FP32 row scales.
- FP32 recurrence for prefill and decode.
- Fused Triton FP8 dequantization, recurrence, requantization, and persistent
  state writeback for packed non-speculative decode.
- Mixed prefill and decode support with decode-first metadata ordering.
- Continuous batching and chunked prefill compatibility.
- Normal vLLM CUDA Graph capture and replay support.
- Matched Stage-1, Stage-2, and Stage-3 H100 benchmarks.
- PyTorch Profiler and Nsight Compute attribution.
- Deterministic generation and GSM8K quality-comparison tooling.

The complete design and measurement notes are in
[`docs/design/qwen3_8_fp8_gdn_recurrent_state.md`](docs/design/qwen3_8_fp8_gdn_recurrent_state.md).

## Model and cache geometry

Qwen3.8-27B has 64 language layers: 48 GDN layers and 16 full-attention layers.
At TP1, each GDN layer uses 48 value heads with `K=128` and `V=128`.

| Tensor | Per-request shape | Storage dtype |
| --- | --- | --- |
| Convolution state | `[10240, 3]` per GDN layer | BF16 in the validated run |
| Recurrent state | `[48, 128, 128]` per GDN layer | FP8 E4M3 |
| Row scales | `[48, 128, 1]` per GDN layer | FP32 |

Across all 48 GDN layers, the original FP32 recurrent state occupies 144 MiB
per request. This branch stores 36 MiB of FP8 state plus 1.125 MiB of FP32
scales. BF16 convolution state adds 2.8125 MiB, for 39.9375 MiB of total GDN
state per request in the validated configuration.

## Implementation stages

### Stage 1: FP8 physical state

The GDN cache tuple is extended to
`(conv_state, ssm_state, ssm_state_scales)`. State is dynamically quantized per
contiguous 128-element row:

```text
scale = 1                         if amax == 0
scale = amax / 448                otherwise
state_fp8 = E4M3(state_fp32 / scale)
state_fp32 = float(state_fp8) * scale
```

Prefill runs the existing recurrence in FP32 and writes its final state back as
FP8 plus FP32 scales. The Stage-1 decode baseline gathers active state into
FP32 scratch, runs the float recurrent kernel, and quantizes the result back.

### Stage 2: fused packed decode

The packed recurrent Triton kernel directly consumes persistent FP8 state and
FP32 scales. It dequantizes into FP32 registers, performs the existing
normalized gated-delta recurrence, produces output from updated FP32 state,
then requantizes and writes state and scales in place. This removes the
Stage-1 Python gather, FP32 scratch, Q/DQ, and indexed writeback operations.

### Stage 3: launch tuning and CUDA Graph

The final fixed dispatch is:

| Persistent state | Batch | `BV` | Warps | Stages |
| --- | ---: | ---: | ---: | ---: |
| FP8 E4M3 | `B <= 32` | 16 | 1 | 3 |
| FP8 E4M3 | `B > 32` | 32 | 2 | 1 |
| FP32/FP16/BF16 | any | 32 | 1 | 3 |

No Triton autotuner runs in the serving path. FP32, FP16, and BF16 retain their
original cache behavior and use the non-FP8 launch configuration.

The controlled H100 sweep evaluated all 18 combinations of `BV={16,32}`,
`num_warps={1,2,4}`, and `num_stages={1,2,3}` at the real
`H=16, HV=48, K=128, V=128` geometry. The table compares the original Stage-2
launch configuration with the selected Stage-3 dispatch:

| Batch | Stage 2 eager | Stage 3 eager | Speedup | Stage 2 graph | Stage 3 graph | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30.050 us | 28.563 us | 1.052x | 4.610 us | 3.413 us | 1.351x |
| 32 | 28.403 us | 26.650 us | 1.066x | 23.941 us | 21.292 us | 1.124x |
| 64 | 66.141 us | 50.390 us | 1.313x | 65.763 us | 49.605 us | 1.326x |
| 128 | 125.917 us | 94.306 us | 1.335x | 125.605 us | 93.402 us | 1.345x |

These are isolated packed-kernel measurements. They establish the Stage-3
kernel tuning result, not a material Stage-3 eager improvement for the full
27B serving workload.

## Canonical H100 end-to-end result

The final matched run used Qwen3.8-27B, prompt length 512, output length 128,
two warmups, and five measured repetitions. Values are mean end-to-end output
token throughput.

| Batch | Stage 1 eager | Stage 2 eager | Stage 3 eager | Stage 3 CUDA Graph |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.20 tok/s | 13.81 tok/s | 13.63 tok/s | 48.38 tok/s |
| 32 | 257.27 tok/s | 356.31 tok/s | 359.25 tok/s | 977.73 tok/s |
| 64 | 417.51 tok/s | 616.24 tok/s | 625.20 tok/s | 1374.65 tok/s |
| 128 | 549.66 tok/s | 999.28 tok/s | 986.72 tok/s | 1785.19 tok/s |

At BS128, Stage 1 to Stage 2 is approximately `1.818x`. Stage-3 isolated kernel
tuning is measurable, but a material Stage-2-to-Stage-3 eager improvement for
the 27B end-to-end run is not established. CUDA Graph results include normal
vLLM compilation and launch-overhead reductions and are not recurrent-kernel-
only gains.

Raw canonical results are stored in
[`docs/assets/qwen3_8_fp8_gdn_h100_results.json`](docs/assets/qwen3_8_fp8_gdn_h100_results.json).

## Matched online serving: FP32 state versus final FP8 state

A separate H100 run compared this branch's existing FP32 recurrent-state path
with the final FP8 recurrent-state path. The two servers ran sequentially on
the same GPU with identical BF16 model execution, TP1, CUDA Graph, prefix
caching off, prompt length 512, output length 128, and 128 requests per
concurrency point.
The only controlled serving variable was `mamba_ssm_cache_dtype`.

| Concurrency | FP32 output tok/s | FP8 output tok/s | Speedup | FP32 p50 TPOT | FP8 p50 TPOT | FP32 p99 TPOT | FP8 p99 TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 46.35 | 46.43 | 1.002x | 20.95 ms | 20.79 ms | 20.96 ms | 20.80 ms |
| 32 | 533.36 | 567.60 | 1.064x | 26.03 ms | 23.77 ms | 32.91 ms | 28.75 ms |
| 64 | 1254.79 | 1306.01 | 1.041x | 35.48 ms | 32.61 ms | 45.78 ms | 40.63 ms |
| 128 | 1273.20 | 1749.18 | 1.374x | 45.92 ms | 42.24 ms | 52.40 ms | 60.60 ms |

At concurrency 128, FP8 increased output throughput by 37.4% and reduced
median TPOT by 8.0%, but p99 TPOT regressed by 15.6%. Median ITL improved from
31.90 to 29.42 ms while p99 ITL regressed from 541.12 to 682.19 ms. At the same
point, p99 TTFT improved from 9480.54 to 5616.21 ms and p99 end-to-end latency
improved from 12844.22 to 9343.68 ms. These mixed tail results reflect one
saturated `request_rate=inf` run and scheduler interactions; they do not support
a blanket tail-latency improvement claim.

With the same `gpu_memory_utilization=0.9`, the allocator reported:

| Capacity metric | FP32 state | FP8 state | Change |
| --- | ---: | ---: | ---: |
| GPU cache blocks (dtype-specific geometry) | 346 | 1241 | 3.587x |
| Cache token capacity | 157468 | 231051 | 1.467x |
| Maximum concurrency at 4096 tokens | 38.44x | 56.41x | 1.467x |

Block counts use different dtype-dependent physical block geometries, so the
3.587x block-count ratio is not itself a request-capacity multiplier. The token
capacity and 4096-token concurrency rows are the comparable end-to-end capacity
metrics. These values come from the vLLM cache configuration, not the 3.88x
theoretical recurrent-state byte ratio. The compact machine-readable result is
stored in
[`docs/assets/qwen3_8_fp8_gdn_online_serving_h100.json`](docs/assets/qwen3_8_fp8_gdn_online_serving_h100.json).

## Kernel profiling

Nsight Compute measured the BS128 Stage-1 and Stage-2 packed recurrent kernels:

| Metric | Stage 1 | Stage 2 | Change |
| --- | ---: | ---: | ---: |
| Kernel latency | 267.872 us | 120.608 us | 2.221x faster |
| DRAM traffic | 751.314 MiB | 181.130 MiB | 4.148x lower |
| L2 traffic | 1181.867 MiB | 311.897 MiB | 3.789x lower |
| DRAM throughput | 87.73% | 46.99% | bottleneck shifted |
| SM throughput | 16.74% | 41.65% | higher compute utilization |
| Active warps | 11.67% | 11.77% | approximately unchanged |
| Long-scoreboard stall | 48.30% | 41.40% | lower, still material |

These are measured profiler values, not theoretical traffic estimates.
PyTorch Profiler confirms that the Stage-1 scratch/QDQ operations disappear
from the Stage-2 packed decode path. The NCU values come from one capture per
implementation with unlocked GPU clocks, so they are profiler attribution
rather than repeated locked-clock latency statistics.

The compact machine-readable kernel, Profiler, and NCU evidence is stored in
[`docs/assets/qwen3_8_fp8_gdn_h100_kernel_profile.json`](docs/assets/qwen3_8_fp8_gdn_h100_kernel_profile.json).

## Quality smoke result

A controlled 500-question GSM8K 5-shot run compared FP32 and FP8 recurrent-state
storage in separate sequential processes on the same H100. Both used greedy
decoding, seed 0, BF16 model execution, TP1, and identical eager serving
settings.

| Recurrent-state storage | Correct | Exact match | Invalid answers |
| --- | ---: | ---: | ---: |
| FP32 | 383/500 | 76.6% | 0 |
| FP8 E4M3 | 387/500 | 77.4% | 0 |

Correctness differed on 18 paired examples: 7 were FP32-only correct and 11
were FP8-only correct. The `+0.8` percentage-point difference is a smoke result,
not evidence that FP8 improves accuracy. This subset and task do not establish
general model-quality preservation.

See [`benchmarks/accuracy/README.md`](benchmarks/accuracy/README.md) for the
methodology and full 1,319-question commands.

## Reproduction

Create the development environment and install this branch:

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Run the focused tests:

```bash
.venv/bin/python -m pytest \
  tests/kernels/test_fused_recurrent_packed_decode.py -q

.venv/bin/python -m pytest \
  tests/kernels/mamba/test_gdn_forward_core_split.py \
  -q -k forward_core_mixed_fp8

.venv/bin/python -m pytest \
  tests/benchmarks/test_qwen3_8_gdn_state_quality.py -q
```

Run the isolated packed-kernel benchmark:

```bash
.venv/bin/python benchmarks/kernels/benchmark_fused_recurrent_gdn_fp8.py \
  --batch-sizes 1 32 64 128 \
  --mode eager cudagraph \
  --warmup 20 \
  --repeat 100 \
  --output qwen38_stage3_kernel_tune.json
```

Run the 27B benchmark with a user-supplied checkpoint path or model ID:

```bash
MODEL="<local-checkpoint-path-or-model-id>"
VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
.venv/bin/python benchmarks/benchmark_qwen3_8_fp8_gdn.py \
  --model "$MODEL" \
  --stage-label stage3-cudagraph \
  --output stage3-cudagraph.json \
  --prompt-len 512 \
  --output-len 128 \
  --batch-sizes 1 32 64 128 \
  --warmup 2 \
  --repeat 5
```

Run the matched online-serving benchmark directly:

```bash
MODEL="<local-checkpoint-path-or-model-id>"
VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
.venv/bin/python benchmarks/qwen3_8_gdn_online_serving.py \
  --model "$MODEL" \
  --output-dir qwen38-online-serving \
  --prompt-len 512 \
  --output-len 128 \
  --num-prompts 128 \
  --concurrencies 1 32 64 128 \
  --num-warmups 2
```

The Modal H100 wrapper used for the published run is
[`benchmarks/modal/qwen3_8_gdn_online_serving.py`](benchmarks/modal/qwen3_8_gdn_online_serving.py).
Its model path is configurable through `--model-path`.

## Scope and limitations

- The fused FP8 packed-decode path is limited to non-speculative decode.
- Speculative decoding is not supported.
- Validation and performance claims are limited to TP1 on H100.
- No multi-GPU performance or correctness claim is made.
- Prefix caching was disabled in the reported serving runs.
- Mixed prefill/decode, continuous batching, chunked prefill, and normal vLLM
  CUDA Graph replay are supported in the validated configuration.
- Active positive state indices in a packed batch must be unique. Duplicate
  slots would race in-place state and scale writes.
- The attention KV cache is unchanged.
