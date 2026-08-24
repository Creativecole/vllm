# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _qwen_gdn_warmup_config,
)


@pytest.mark.parametrize(
    ("physical_state_dtype", "expected_compute_dtype"),
    [
        (torch.float8_e4m3fn, torch.float32),
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_qwen_gdn_warmup_compute_state_dtype(
    physical_state_dtype: torch.dtype,
    expected_compute_dtype: torch.dtype,
):
    ssm_state = torch.empty((3, 2, 4, 3), dtype=physical_state_dtype)
    layer = SimpleNamespace(
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=3,
        head_v_dim=4,
        conv_kernel_size=4,
        tp_size=1,
        kv_cache=(torch.empty((3, 14), dtype=torch.bfloat16), ssm_state),
        A_log=torch.empty(2),
        dt_bias=torch.empty(2),
    )
    config = _qwen_gdn_warmup_config({"model.layers.0.linear_attn": layer})

    assert ssm_state.dtype == physical_state_dtype
    assert config is not None
    assert config.compute_state_dtype == expected_compute_dtype
    assert config.state_stride_token == 2 * 4 * 3
