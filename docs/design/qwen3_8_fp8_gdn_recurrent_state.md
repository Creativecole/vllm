# Qwen3.8 FP8 GDN recurrent-state cache on Hopper

This document describes an experimental vLLM v0.27.1 implementation for
Qwen3.8-27B on one NVIDIA H100. The checkpoint used for validation declares
`Qwen3_5ForConditionalGeneration` and a `qwen3_5_text` configuration. This is a
focused recurrent-state project: the attention KV cache is unchanged.

## Problem

The model has 64 language layers: 48 Gated DeltaNet (GDN) layers and 16 full
attention layers. At tensor parallel size 1, every GDN layer maintains:

- convolution state: `[..., 10240, 3]` in the model activation dtype;
- recurrent state: `[..., 48, 128, 128]`;
- row scale in the FP8 implementation: `[..., 48, 128, 1]` in FP32.

An FP32 recurrent state occupies 144 MiB per request across the 48 GDN layers.
The FP8 state occupies 36 MiB and its FP32 row scales occupy 1.125 MiB. The
BF16 convolution states add 2.8125 MiB, giving 39.9375 MiB of GDN state per
request in the validated configuration.

Memory capacity was only half of the issue. Decode reads and updates the entire
recurrent matrix for every active request and every GDN layer, so materializing
an FP32 state for an FP8 cache can create substantial HBM traffic.

## Stage 1: FP8 physical state

Stage 1 changes the physical GDN cache tuple from `(conv_state, ssm_state)` to
`(conv_state, ssm_state, ssm_state_scales)`:

- `conv_state` keeps its existing dtype and layout;
- `ssm_state` is stored as `torch.float8_e4m3fn`;
- `ssm_state_scales` is FP32 with one scale per contiguous 128-element row;
- recurrence is still computed in FP32;
- `scale = 1` for an all-zero row, otherwise `amax / 448`;
- state writes use the exact E4M3 bit representation.

The prefill path initializes fresh requests with zero FP32 state, dequantizes
only requests that already own valid state, runs the existing FP32 chunk
recurrence, then quantizes the final state back into the persistent cache.

The Stage-1 decode baseline gathers valid cache slots into FP32 scratch, keeps
scratch slot 0 reserved, remaps active requests to slots `1..N`, runs the
existing float packed recurrent kernel, and quantizes and writes active slots
back. NULL slot 0 and PAD `-1` never read or modify persistent state.

## Stage 2: fused FP8 packed decode

Stage 2 removes the Python gather, FP32 scratch, Q/DQ, and indexed writeback
from non-speculative packed decode. The Triton kernel now:

1. loads FP8 persistent state and its FP32 row scale;
2. dequantizes into FP32 registers;
3. performs the existing normalized gated-delta recurrence in FP32;
4. produces the output from the updated FP32 state;
5. reduces `amax` across the 128-element row;
6. requantizes to E4M3 and writes state plus FP32 scale to the original slot.

The float32, float16, and bfloat16 paths retain their previous state dtype and
launch configuration.

## Mixed serving

The normal non-speculative mixed batch is decode-first. Its decode prefix uses
the same fused Stage-2 kernel directly on persistent FP8 state; its prefill tail
uses the Stage-1 FP32 chunk recurrence and FP8 writeback. Outputs are stitched
back in decode-first order.

This path has been tested with continuous batching and chunked prefill. The
convolution path and metadata ordering are unchanged.

## Performance

### Earlier matched eager run (non-canonical)

The following older matched results use Qwen3.8-27B, one H100, a 512-token
prompt, 128 generated tokens, and eager execution. They compare only the FP8
non-speculative decode implementation; allocator, prefill, convolution, and
serving behavior are shared. This is retained as an earlier matched eager run
and is not the canonical final end-to-end result.

| Batch | Stage 1 scratch/QDQ | Stage 2 fused | Stage 2 speedup |
| ---: | ---: | ---: | ---: |
| 1 | 12.47 tok/s | 16.90 tok/s | 1.355x |
| 32 | 294.69 tok/s | 436.04 tok/s | 1.480x |
| 64 | 448.69 tok/s | 725.01 tok/s | 1.616x |
| 128 | 558.04 tok/s | 1109.19 tok/s | 1.988x |

These are end-to-end output-token throughput results, not isolated kernel
speedups. In particular, the earlier 1.988x result is not the final canonical
Stage-1-to-Stage-2 E2E number.

## Profiling

PyTorch Profiler attributes the Stage-1 overhead to the active-state gather,
dequantization, FP32 scratch operations, row-wise quantization, and indexed
writeback around packed decode. Those operations disappear from the Stage-2
decode trace; prefill quantization remains by design.

The matched BS128 trace (`prompt_len=512`, `output_len=128`, one profiled eager
run after one warmup) recorded these selected operator totals across the full
profiled generation:

| Operator | Stage 1 self-device time | Stage 2 self-device time | Stage 1 calls | Stage 2 calls |
| --- | ---: | ---: | ---: | ---: |
| `aten::gather` | 477.532 ms | 0 ms | 12,611 | 0 |
| `aten::div` | 2189.874 ms | 18.811 ms | 13,056 | 480 |
| `aten::amax` | 1439.813 ms | 14.481 ms | 6,528 | 240 |
| `aten::index_copy_` | 1742.357 ms | 18.126 ms | 13,200 | 624 |
| packed recurrent kernel | 1661.296 ms | 746.946 ms | 6,288 | 6,288 |

The residual Stage-2 quantization-related calls include prefill processing,
which is intentionally unchanged. These totals are attribution from one full
profiled run, not per-kernel latency measurements.

Nsight Compute measured the BS128 packed recurrent kernel at 267.872 us for
Stage 1 and 120.608 us for Stage 2, a 2.221x kernel speedup. Measured DRAM
traffic fell from about 751.3 MiB to 181.1 MiB (about 4.15x), and measured L2
traffic fell about 3.79x. These are profiler measurements, not theoretical byte
counts. Exact counters, profiler totals, and the Stage-3 kernel sweep are in
[`qwen3_8_fp8_gdn_h100_kernel_profile.json`](../assets/qwen3_8_fp8_gdn_h100_kernel_profile.json).

| NCU metric | Stage 1 | Stage 2 | Change |
| --- | ---: | ---: | ---: |
| Kernel latency | 267.872 us | 120.608 us | 2.221x faster |
| DRAM traffic | 751.314 MiB | 181.130 MiB | 4.148x lower |
| L2 traffic | 1181.867 MiB | 311.897 MiB | 3.789x lower |
| DRAM throughput | 87.73% | 46.99% | lower saturation |
| SM throughput | 16.74% | 41.65% | higher utilization |
| Active warps | 11.67% | 11.77% | approximately unchanged |
| Long-scoreboard stall | 48.30% | 41.40% | lower, still material |

GPU clocks were not locked for these single NCU captures. The counters support
bottleneck attribution; the latency rows should not be interpreted as a
repeated locked-clock microbenchmark distribution.

Stage 1 reached 87.73% DRAM throughput. Stage 2 reached 46.99% DRAM throughput,
41.65% SM throughput, about 11.77% active warps, and about 41.4% long-scoreboard
stall. The bottleneck therefore moved away from the Stage-1 bulk scratch
traffic, while the fused kernel still has latency-hiding limits.

## Stage 3: launch tuning and CUDA Graph

Stage 3 evaluated all 18 combinations of `BV={16,32}`,
`num_warps={1,2,4}`, and `num_stages={1,2,3}` at the real
`H=16, HV=48, K=128, V=128` geometry and batches 1, 32, 64, and 128. Every
candidate was checked against an FP32-recurrence reference before timing.

The selected fixed dispatch is:

| Persistent state | Batch | BV | Warps | Stages |
| --- | ---: | ---: | ---: | ---: |
| FP8 E4M3 | `B <= 32` | 16 | 1 | 3 |
| FP8 E4M3 | `B > 32` | 32 | 2 | 1 |
| FP32/FP16/BF16 | any | 32 | 1 | 3 |

The matched kernel sweep produced:

| Batch | Stage-2 config eager | Stage-3 eager | Speedup | Stage-2 graph | Stage-3 graph | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30.050 us | 28.563 us | 1.052x | 4.610 us | 3.413 us | 1.351x |
| 32 | 28.403 us | 26.650 us | 1.066x | 23.941 us | 21.292 us | 1.124x |
| 64 | 66.141 us | 50.390 us | 1.313x | 65.763 us | 49.605 us | 1.326x |
| 128 | 125.917 us | 94.306 us | 1.335x | 125.605 us | 93.402 us | 1.345x |

No Triton autotuner is used at serving time. The FP8 cache and scale tensors
remain stable graph inputs, and state indices remain device data read at replay
time. A focused CUDA Graph test captures the packed kernel, restores persistent
state after capture, replays it, changes the state-index buffer, and verifies a
second replay against eager execution. NULL/PAD rows remain ignored.

Active positive state indices must be unique within a packed batch. Duplicate
positive slots would race the in-place recurrent-state and scale writes. This
is a scheduler invariant; the decode hot path does not add an expensive runtime
uniqueness check.

### Canonical final four-way H100 rerun

The canonical final 27B run used two warmups and five measured repetitions per
point.
The table reports the mean end-to-end output-token throughput; raw durations
and token digests are in
[`qwen3_8_fp8_gdn_h100_results.json`](../assets/qwen3_8_fp8_gdn_h100_results.json).

| Batch | Stage 1 eager | Stage 2 eager | Stage 3 eager | Stage 3 CUDA Graph | Graph / Stage 3 eager |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.20 tok/s | 13.81 tok/s | 13.63 tok/s | 48.38 tok/s | 3.55x |
| 32 | 257.27 tok/s | 356.31 tok/s | 359.25 tok/s | 977.73 tok/s | 2.72x |
| 64 | 417.51 tok/s | 616.24 tok/s | 625.20 tok/s | 1374.65 tok/s | 2.20x |
| 128 | 549.66 tok/s | 999.28 tok/s | 986.72 tok/s | 1785.19 tok/s | 1.81x |

In this canonical rerun, BS128 Stage 1 to Stage 2 is 549.66 to 999.28 tok/s,
approximately 1.818x.

The Stage-2-to-Stage-3 eager differences are near run-to-run noise: +0.83% at
BS32 and +1.45% at BS64, while the BS1 and BS128 means moved by about -1.3%.
At BS128 the Stage-3 median was 994.84 tok/s versus 993.30 tok/s for Stage 2.
The data therefore supports a packed-kernel improvement, but not a claim of a
material eager end-to-end Stage-3 speedup.

The CUDA Graph run used vLLM's resolved `FULL_AND_PIECEWISE` mode, not a custom
graph wrapper. It captured 35 mixed prefill/decode piecewise graphs and 19 full
decode graphs. Capture completed without a cache, scale, kernel, or state-index
error and used a 0.57 GiB graph pool. The graph-versus-eager numbers include
vLLM compilation and launch-overhead reductions; they must not be attributed to
the recurrent kernel alone.

## Model-quality validation protocol

The final quality comparison is implemented in
[`qwen3_8_gdn_state_quality.py`](../../benchmarks/accuracy/qwen3_8_gdn_state_quality.py),
with commands and extraction details in its
[`README`](../../benchmarks/accuracy/README.md).
It runs FP32 and FP8 recurrent-state storage in separate, otherwise matched
processes and compares deterministic token sensitivity plus GSM8K 5-shot exact
match.

A 500-question H100 smoke run used the first 500 GSM8K test examples, greedy
decoding, seed 0, BF16 model execution, TP1, eager mode, and identical serving
settings except for recurrent-state storage dtype. FP32 state scored 383/500
(76.6%) and FP8 E4M3 state scored 387/500 (77.4%), a +0.8 percentage-point
difference. Correctness differed on 18 paired examples: 7 were FP32-only
correct and 11 were FP8-only correct. Neither run produced an invalid answer.
This limited smoke result does not establish an FP8 accuracy improvement or
general model-quality preservation; the full 1,319-question evaluation and
broader tasks remain necessary for stronger conclusions.

## Reproduction

Run the isolated H100 configuration sweep:

```bash
.venv/bin/python benchmarks/kernels/benchmark_fused_recurrent_gdn_fp8.py \
  --batch-sizes 1 32 64 128 \
  --mode eager cudagraph \
  --warmup 20 \
  --repeat 100 \
  --output qwen38_stage3_kernel_tune.json
```

Run the focused kernel and mixed-serving tests:

```bash
.venv/bin/python -m pytest \
  tests/kernels/test_fused_recurrent_packed_decode.py -q

.venv/bin/python -m pytest \
  tests/kernels/mamba/test_gdn_forward_core_split.py \
  -q -k forward_core_mixed_fp8
```

Run one end-to-end benchmark case. Add `--enforce-eager` for eager execution;
omit it for the normal vLLM CUDA Graph path. Set `MODEL` to a user-supplied
local checkpoint path or model identifier.

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

For a matched Stage-1/Stage-2/Stage-3 comparison, run the same command from
each controlled worktree. Change only the decode implementation between source
revisions and keep every benchmark argument identical.

## Limitations

- Speculative decoding with the FP8 GDN recurrent-state cache is not supported.
- Validation is for tensor parallel size 1 on H100; this project makes no
  multi-GPU performance or correctness claim.
- Prefix caching was disabled in the reported serving runs.
- Token digests and generation sanity checks are not model-quality evaluation.
  A downstream accuracy evaluation is required before claiming quality
  preservation.
