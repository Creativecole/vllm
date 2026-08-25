# Qwen3.8 GDN recurrent-state quality comparison

This evaluator compares two otherwise identical Qwen3.8-27B vLLM runs whose
only controlled change is `mamba_ssm_cache_dtype`: `float32` versus
`fp8_e4m3fn`. Each `run` command loads one model once. The two state dtypes are
separate processes because a vLLM engine cannot change its allocated cache
dtype after initialization. `compare` reads the resulting JSON files without
loading a model.

The deterministic generation task contains 96 fixed prompts across 16 domains
and six context-length variants. Each response uses greedy decoding for exactly
256 tokens with EOS ignored. The comparison reports fully identical responses,
aligned token agreement, zero-based first-divergence positions, and obvious
token-output failures. First-divergence p10 is reported only when at least ten
prompts diverge. Token agreement measures sensitivity to recurrent-state storage
and is not a model-quality metric.

GSM8K uses the first five training examples as demonstrations and evaluates the
first `--limit` test examples in source order. It uses greedy generation with a
256-token cap and stops before a subsequent `Question:` block. Answer extraction
uses text following the final `####` marker when present; otherwise it takes the
final signed integer or decimal. Thousands separators are removed and values
are normalized with Python `Decimal` before exact match. The source file URLs
and SHA-256 hashes are recorded in every run report.

Run the 50-question H100 smoke comparison in eager mode:

```bash
MODEL="<local-checkpoint-path-or-model-id>"
OUT="qwen38-gdn-quality-smoke"
GSM8K_DATA="<shared-gsm8k-cache-directory>"

VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
.venv/bin/python benchmarks/accuracy/qwen3_8_gdn_state_quality.py run \
  --model "$MODEL" --state-dtype float32 --tasks parity gsm8k --limit 50 \
  --seed 0 --enforce-eager --max-model-len 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 --gsm8k-data-dir "$GSM8K_DATA" \
  --output "$OUT-fp32.json"

VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
.venv/bin/python benchmarks/accuracy/qwen3_8_gdn_state_quality.py run \
  --model "$MODEL" --state-dtype fp8_e4m3fn --tasks parity gsm8k --limit 50 \
  --seed 0 --enforce-eager --max-model-len 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 --gsm8k-data-dir "$GSM8K_DATA" \
  --output "$OUT-fp8.json"

.venv/bin/python benchmarks/accuracy/qwen3_8_gdn_state_quality.py compare \
  --fp32-run "$OUT-fp32.json" --fp8-run "$OUT-fp8.json" \
  --output "$OUT-comparison.json"
```

After reviewing the smoke output, run all 1,319 GSM8K questions. The parity
task need not be repeated:

```bash
MODEL="<local-checkpoint-path-or-model-id>"
OUT="qwen38-gdn-quality-gsm8k-full"
GSM8K_DATA="<shared-gsm8k-cache-directory>"

VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
.venv/bin/python benchmarks/accuracy/qwen3_8_gdn_state_quality.py run \
  --model "$MODEL" --state-dtype float32 --tasks gsm8k --limit 1319 \
  --seed 0 --enforce-eager --max-model-len 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 --gsm8k-data-dir "$GSM8K_DATA" \
  --output "$OUT-fp32.json"

VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
.venv/bin/python benchmarks/accuracy/qwen3_8_gdn_state_quality.py run \
  --model "$MODEL" --state-dtype fp8_e4m3fn --tasks gsm8k --limit 1319 \
  --seed 0 --enforce-eager --max-model-len 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 --gsm8k-data-dir "$GSM8K_DATA" \
  --output "$OUT-fp8.json"

.venv/bin/python benchmarks/accuracy/qwen3_8_gdn_state_quality.py compare \
  --fp32-run "$OUT-fp32.json" --fp8-run "$OUT-fp8.json" \
  --output "$OUT-comparison.json"
```

The smoke run has at most 74,752 generated tokens across both dtypes: 49,152
from parity plus 25,600 from GSM8K. The full GSM8K pair has at most 675,328
generated tokens. Actual GSM8K totals should be lower because EOS and the stop
sequence remain active. Wall time depends on answer length and model startup;
budget roughly 5-15 minutes for smoke and 30-90 minutes for the full pair on a
single H100, then replace those estimates with measured elapsed times.

## Recorded 500-question H100 smoke result

A sequential-process run on one H100 evaluated the first 500 GSM8K test
examples with the settings above, except that only the GSM8K task was run. FP32
state scored 383/500 (76.6%) and FP8 E4M3 state scored 387/500 (77.4%). The
paired correctness label changed for 18 examples: 7 were FP32-only correct and
11 were FP8-only correct. Both runs returned zero invalid answers. The
generation phases produced 79,352 and 78,744 tokens and took 173.0 and 175.0
seconds for FP32 and FP8, respectively; these timings include inference but not
model initialization.

The +0.8 percentage-point difference is a smoke result, not evidence that FP8
improves accuracy. This subset and task alone cannot establish general
model-quality preservation.

This evaluation covers one deterministic prompt set and one arithmetic
reasoning task. It does not measure perplexity, broad language understanding,
or every serving trajectory, and it cannot establish general model-quality
preservation by itself. Obvious/non-finite failure counts cover returned token
objects and generation length, not direct inspection of internal activations or
logits.
