# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    quantize_scaled,
)


def test_fp8_gdn_state_cache_dtype_and_shape():
    dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.bfloat16, "auto", "fp8_e4m3fn"
    )
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_world_size=1,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
    )
    copy_funcs = MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    assert dtypes == (torch.bfloat16, torch.float8_e4m3fn, torch.float32)
    assert shapes == ((3, 10240), (48, 128, 128), (48, 128, 1))
    assert copy_funcs == (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        get_temporal_copy_spec,
    )


def test_quantize_scaled_rowwise_qdq():
    state = torch.linspace(-8, 8, 2 * 3 * 128, dtype=torch.float32).reshape(2, 3, 128)
    state[0, 0] = 0

    quantized, scales = quantize_scaled(state, torch.float8_e4m3fn)
    dequantized = quantized.float() * scales

    expected_amax = state.abs().amax(dim=-1, keepdim=True)
    expected_scales = torch.where(
        expected_amax == 0,
        torch.ones_like(expected_amax),
        expected_amax / 448.0,
    )
    assert quantized.shape == state.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 3, 1)
    assert scales.dtype == torch.float32
    torch.testing.assert_close(scales, expected_scales)
    torch.testing.assert_close(dequantized, state, rtol=0.07, atol=0.02)


def test_float32_gdn_state_cache_legacy_dtype():
    dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        torch.bfloat16, "auto", "float32"
    )

    assert dtypes == (torch.bfloat16, torch.float32, torch.float32)
