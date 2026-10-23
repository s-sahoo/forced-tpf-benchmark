"""Inference-only grouped RMSNorm kernels for XLLM/BBQ."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _group_rms_norm_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_group = tl.program_id(0)
    group = row_group % num_groups
    row = row_group // num_groups
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < group_size
    base = row * (num_groups * group_size) + group * group_size

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / group_size
    rstd = tl.rsqrt(mean_square + eps)
    weight = tl.load(
        weight_ptr + group * group_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(y_ptr + base + offsets, x * rstd * weight, mask=mask)


@triton.jit
def _add_group_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    eps: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
    INPUT_IS_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse residual accumulation and grouped RMSNorm for inference.

    ``x`` is overwritten with the updated residual. Rounding the FP32 sum back
    to the input dtype before normalization preserves the existing XLLM path:
    ``residual = (x.float() + residual.float()).to(input_dtype)``.
    """
    row_group = tl.program_id(0)
    group = row_group % num_groups
    row = row_group // num_groups
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < group_size
    base = row * (num_groups * group_size) + group * group_size

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + base + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    updated_residual = x + residual
    if INPUT_IS_BF16:
        updated_residual = updated_residual.to(tl.bfloat16).to(tl.float32)
    elif INPUT_IS_FP16:
        updated_residual = updated_residual.to(tl.float16).to(tl.float32)

    mean_square = tl.sum(updated_residual * updated_residual, axis=0) / group_size
    rstd = tl.rsqrt(mean_square + eps)
    weight = tl.load(
        weight_ptr + group * group_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(x_ptr + base + offsets, updated_residual, mask=mask)
    tl.store(
        y_ptr + base + offsets,
        updated_residual * rstd * weight,
        mask=mask,
    )


def _launch_config(x: torch.Tensor, num_groups: int) -> tuple[int, int, int]:
    num_channels = x.shape[-1]
    group_size = num_channels // num_groups
    block_size = triton.next_power_of_2(group_size)
    num_rows = x.numel() // num_channels
    return group_size, block_size, num_rows


def _num_warps(block_size: int, num_rows: int) -> int:
    # On H200, one warp is faster for the BBQ checkpoint's 1024-wide groups
    # until larger row counts provide enough work to favor four warps.
    if block_size == 1024 and num_rows <= 512:
        return 1
    return 4 if block_size <= 1024 else 8


def group_rms_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply grouped RMSNorm and a per-channel affine weight."""
    group_size, block_size, num_rows = _launch_config(x, num_groups)
    y = torch.empty_like(x)
    _group_rms_norm_kernel[(num_rows * num_groups,)](
        x,
        weight,
        y,
        num_groups=num_groups,
        group_size=group_size,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size, num_rows),
    )
    return y


def add_group_rms_norm_(
    x: torch.Tensor,
    residual: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Update ``x`` with ``x + residual`` and return its grouped RMSNorm."""
    group_size, block_size, num_rows = _launch_config(x, num_groups)
    y = torch.empty_like(x)
    _add_group_rms_norm_kernel[(num_rows * num_groups,)](
        x,
        residual,
        weight,
        y,
        num_groups=num_groups,
        group_size=group_size,
        eps=eps,
        INPUT_IS_BF16=x.dtype == torch.bfloat16,
        INPUT_IS_FP16=x.dtype == torch.float16,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size, num_rows),
    )
    return y
