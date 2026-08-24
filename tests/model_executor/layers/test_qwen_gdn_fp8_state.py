# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _index_copy_fp8_state,
    _prepare_fp8_prefill_initial_state,
)


def test_index_copy_fp8_state_preserves_bits_and_untouched_rows():
    dst = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3).to(torch.float8_e4m3fn)
    src = torch.tensor(
        [
            [[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [-10.0, -11.0, -12.0]],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    indices = torch.tensor([1, 3], dtype=torch.int32)
    original_bits = dst.view(torch.uint8).clone()

    _index_copy_fp8_state(dst, indices, src)

    dst_bits = dst.view(torch.uint8)
    assert torch.equal(dst_bits.index_select(0, indices.long()), src.view(torch.uint8))
    assert torch.equal(dst_bits[0], original_bits[0])
    assert torch.equal(dst_bits[2], original_bits[2])


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
