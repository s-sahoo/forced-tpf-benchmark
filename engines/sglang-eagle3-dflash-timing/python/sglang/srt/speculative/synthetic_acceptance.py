from __future__ import annotations

import math
from typing import Optional

import torch


def sample_simulated_accept_lengths(
    simulate_acc_len: float,
    simulate_acc_method: str,
    max_len: int,
    batch_size: int,
    device: torch.device,
    uniform_samples: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample benchmark-only emitted lengths independently for each request."""
    if batch_size <= 0:
        return torch.empty((0,), dtype=torch.int32, device=device)

    simulate_acc_len = max(1.0, min(float(max_len), float(simulate_acc_len)))
    if simulate_acc_method == "multinomial":
        return (
            torch.normal(
                mean=simulate_acc_len,
                std=1.0,
                size=(batch_size,),
                device=device,
            )
            .round_()
            .clamp_(min=1, max=max_len)
            .to(torch.int32)
        )
    if simulate_acc_method != "match-expected":
        raise ValueError(f"Invalid simulate_acc_method: {simulate_acc_method}")

    lower = math.floor(simulate_acc_len)
    fraction = simulate_acc_len - lower
    lengths = torch.full(
        (batch_size,), lower, dtype=torch.int32, device=device
    )
    if fraction == 0.0 or lower == max_len:
        return lengths

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (batch_size,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (batch_size,):
            raise ValueError(
                "uniform_samples shape mismatch: "
                f"expected {(batch_size,)}, got {tuple(uniform_samples.shape)}"
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)
    lengths.add_(uniform_samples.lt(fraction))
    return lengths
