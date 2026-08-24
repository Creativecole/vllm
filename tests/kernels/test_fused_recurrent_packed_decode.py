# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import quantize_scaled
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    _packed_decode_launch_config,
)


@pytest.mark.parametrize(
    ("batch_size", "is_fp8_state", "expected"),
    [
        (1, True, (16, 1, 3)),
        (32, True, (16, 1, 3)),
        (64, True, (32, 2, 1)),
        (128, True, (32, 2, 1)),
        (1, False, (32, 1, 3)),
        (128, False, (32, 1, 3)),
    ],
)
def test_packed_decode_launch_config(
    batch_size: int,
    is_fp8_state: bool,
    expected: tuple[int, int, int],
) -> None:
    assert _packed_decode_launch_config(batch_size, is_fp8_state) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype, strided_mixed_qkv: bool
):
    torch.manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = 32
    H = 4
    HV = 8  # grouped value attention: HV must be divisible by H
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device("cuda")

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B + 1, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).to(dtype).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Need NVIDIA CUDA device",
)
def test_fused_recurrent_packed_decode_fp8_state_matches_fp32_reference():
    torch.manual_seed(0)

    B = 6
    H = 2
    HV = 4
    K = 128
    V = 128
    num_states = 8
    qkv_dim = 2 * H * K + HV * V
    device = torch.device("cuda")
    activation_dtype = torch.bfloat16

    mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=activation_dtype)
    value_row_zero_offsets = 2 * H * K + torch.arange(HV, device=device) * V
    mixed_qkv[:, value_row_zero_offsets] = 0
    a = torch.randn((B, HV), device=device, dtype=activation_dtype)
    b = torch.randn((B, HV), device=device, dtype=activation_dtype)
    A_log = torch.randn((HV,), device=device, dtype=torch.float32)
    dt_bias = torch.randn((HV,), device=device, dtype=torch.float32)
    state_indices = torch.tensor([1, 3, 0, -1, 5, 6], device=device, dtype=torch.int32)

    initial_state = (
        torch.randn((num_states, HV, V, K), device=device, dtype=torch.float32) * 0.05
    )
    initial_state[:, :, 0, :] = 0
    state_fp8, state_scales = quantize_scaled(initial_state, torch.float8_e4m3fn)
    state_fp8_before = state_fp8.view(torch.uint8).clone()
    state_scales_before = state_scales.clone()
    state_ref = state_fp8.float() * state_scales

    out_ref = torch.empty((B, 1, HV, V), device=device, dtype=activation_dtype)
    out_fp8 = torch.empty_like(out_ref)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_ref,
        out=out_ref,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_fp8,
        out=out_fp8,
        ssm_state_indices=state_indices,
        ssm_state_scales=state_scales,
        use_qk_l2norm_in_kernel=True,
    )

    valid = state_indices > 0
    active_state_indices = state_indices[valid].long()
    expected_state, expected_scales = quantize_scaled(
        state_ref.index_select(0, active_state_indices), torch.float8_e4m3fn
    )

    torch.testing.assert_close(out_fp8[valid], out_ref[valid], rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        out_fp8[~valid], torch.zeros_like(out_fp8[~valid]), rtol=0, atol=0
    )
    assert torch.equal(
        state_fp8.index_select(0, active_state_indices).view(torch.uint8),
        expected_state.view(torch.uint8),
    )
    torch.testing.assert_close(
        state_scales.index_select(0, active_state_indices),
        expected_scales,
        rtol=1e-6,
        atol=0,
    )
    torch.testing.assert_close(
        state_scales[active_state_indices, :, 0, 0],
        torch.ones_like(state_scales[active_state_indices, :, 0, 0]),
        rtol=0,
        atol=0,
    )

    untouched_state_indices = torch.tensor(
        [0, 2, 4, 7], device=device, dtype=torch.int64
    )
    assert torch.equal(
        state_fp8.view(torch.uint8).index_select(0, untouched_state_indices),
        state_fp8_before.index_select(0, untouched_state_indices),
    )
    torch.testing.assert_close(
        state_scales.index_select(0, untouched_state_indices),
        state_scales_before.index_select(0, untouched_state_indices),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Need NVIDIA CUDA device",
)
def test_fused_recurrent_packed_decode_fp8_cudagraph_replay():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FP8 GDN CUDA Graph test targets Hopper SM90")

    torch.manual_seed(1)
    device = torch.device("cuda")
    B, H, HV, K, V = 4, 2, 4, 128, 128
    num_states = 7
    qkv_dim = 2 * H * K + HV * V

    mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=torch.bfloat16)
    a = torch.randn((B, HV), device=device, dtype=torch.bfloat16)
    b = torch.randn((B, HV), device=device, dtype=torch.bfloat16)
    A_log = torch.randn((HV,), device=device, dtype=torch.float32)
    dt_bias = torch.randn((HV,), device=device, dtype=torch.float32)
    state_indices = torch.tensor([1, 3, 0, -1], device=device, dtype=torch.int32)
    initial_state = (
        torch.randn((num_states, HV, V, K), device=device, dtype=torch.float32) * 0.05
    )
    state_fp8, state_scales = quantize_scaled(initial_state, torch.float8_e4m3fn)

    def run(
        state: torch.Tensor,
        scales: torch.Tensor,
        out: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=state,
            out=out,
            ssm_state_indices=indices,
            ssm_state_scales=scales,
            use_qk_l2norm_in_kernel=True,
        )

    # Compile before capture, as the vLLM graph runner does during warmup.
    warm_state = state_fp8.clone()
    warm_scales = state_scales.clone()
    warm_out = torch.empty((B, 1, HV, V), device=device, dtype=torch.bfloat16)
    run(warm_state, warm_scales, warm_out, state_indices)
    torch.cuda.synchronize()

    graph_state = state_fp8.clone()
    graph_scales = state_scales.clone()
    graph_out = torch.empty_like(warm_out)
    graph_indices = state_indices.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(graph_state, graph_scales, graph_out, graph_indices)

    def restore_graph_state() -> None:
        graph_state.view(torch.uint8).copy_(state_fp8.view(torch.uint8))
        graph_scales.copy_(state_scales)

    def assert_replay_matches_eager(indices: torch.Tensor) -> None:
        reference_state = state_fp8.clone()
        reference_scales = state_scales.clone()
        reference_out = torch.empty_like(graph_out)
        run(reference_state, reference_scales, reference_out, indices)
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(
            graph_state.view(torch.uint8), reference_state.view(torch.uint8)
        )
        torch.testing.assert_close(graph_scales, reference_scales, rtol=0, atol=0)
        torch.testing.assert_close(graph_out, reference_out, rtol=0, atol=0)

    # Capture executes once, so reset persistent state before the first replay.
    restore_graph_state()
    assert_replay_matches_eager(graph_indices)

    # State indices are replay inputs, not capture-time constants. NULL/PAD rows
    # remain ignored when active slots change between replays.
    restore_graph_state()
    graph_indices.copy_(torch.tensor([2, 4, 0, -1], device=device, dtype=torch.int32))
    assert_replay_matches_eager(graph_indices)
