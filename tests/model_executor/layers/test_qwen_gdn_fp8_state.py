# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _prepare_fp8_packed_decode_state,
    _prepare_fp8_prefill_initial_state,
)


def test_fp8_packed_decode_reserves_null_scratch_slot():
    ssm_state = torch.zeros((5, 1, 1, 2), dtype=torch.float8_e4m3fn)
    ssm_state[0] = torch.nan
    ssm_state[3] = torch.tensor([[[2.0, -4.0]]])
    ssm_state[4] = torch.tensor([[[100.0, 100.0]]])
    scales = torch.full((5, 1, 1, 1), 0.5, dtype=torch.float32)
    state_indices = torch.tensor([0, 3, -1], dtype=torch.int32)

    scratch, kernel_indices, active_indices = _prepare_fp8_packed_decode_state(
        ssm_state, scales, state_indices
    )

    assert scratch.shape == (2, 1, 1, 2)
    torch.testing.assert_close(scratch[0], torch.zeros_like(scratch[0]))
    torch.testing.assert_close(scratch[1], torch.tensor([[[1.0, -2.0]]]))
    torch.testing.assert_close(
        kernel_indices, torch.tensor([-1, 1, -1], dtype=torch.int32)
    )
    torch.testing.assert_close(active_indices, torch.tensor([3], dtype=torch.int64))


class _UnreadableCache:
    shape = (4, 2, 3, 4)
    device = torch.device("cpu")

    def index_select(self, *_args, **_kwargs):
        raise AssertionError("fresh prefill must not read cache contents")


def test_fresh_fp8_prefill_does_not_read_cache():
    initial_state = _prepare_fp8_prefill_initial_state(
        _UnreadableCache(),  # type: ignore[arg-type]
        _UnreadableCache(),  # type: ignore[arg-type]
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([False, False]),
    )

    assert initial_state.dtype == torch.float32
    assert initial_state.shape == (2, 2, 3, 4)
    torch.testing.assert_close(initial_state, torch.zeros_like(initial_state))
