# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for discrete diffusion (dLLM) models."""

from fractions import Fraction

from pydantic import Field

from vllm.config.utils import config


@config
class DiffusionConfig:
    """Configuration for discrete diffusion language models (dLLMs).

    dLLMs generate tokens via iterative denoising over a fixed-length canvas
    rather than left-to-right autoregressive decoding. They reuse the
    speculative-decoding data path (draft token ids, scheduled spec decode
    tokens) with overloaded semantics for block-based generation.
    """

    canvas_length: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """Length of the denoising canvas (block).  Also determines the number of
    speculative tokens scheduled per step."""

    max_denoising_steps: int | None = None
    """Maximum number of denoising iterations per canvas block.
    If not set, read from the model's generation_config.json."""

    forced_tpf: float | None = Field(default=None, gt=0)
    """Benchmark-only target tokens per denoising forward. DiffusionGemma
    realizes fractional targets with cumulative, deterministic integer step
    budgets shared by requests that start a canvas together."""

def get_forced_denoising_step_budget(
    canvas_length: int,
    forced_tpf: float,
    completed_token_slots: int,
    completed_forward_slots: int,
    cohort_size: int,
) -> int:
    """Return a batch-weighted balanced budget for one canvas cohort."""
    if canvas_length <= 0:
        raise ValueError("canvas_length must be positive")
    if forced_tpf <= 0:
        raise ValueError("forced_tpf must be positive")
    if completed_token_slots < 0 or completed_forward_slots < 0:
        raise ValueError("completed slot counts must be non-negative")
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")

    steps_per_canvas = Fraction(canvas_length, 1) / Fraction(str(forced_tpf))
    lower = steps_per_canvas.numerator // steps_per_canvas.denominator
    upper = -(-steps_per_canvas.numerator // steps_per_canvas.denominator)
    target_total = Fraction(
        completed_token_slots + cohort_size * canvas_length,
        1,
    ) / Fraction(str(forced_tpf))

    return min(
        {lower, upper},
        key=lambda budget: (
            abs(
                Fraction(completed_forward_slots + cohort_size * budget, 1)
                - target_total
            ),
            budget,
        ),
    )
