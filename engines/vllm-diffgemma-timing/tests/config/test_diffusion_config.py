# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.diffusion import get_forced_denoising_step_budget

pytestmark = pytest.mark.cpu_test


def test_forced_tpf_balances_fractional_step_budget():
    token_slots = 0
    forward_slots = 0
    budgets = []

    for _ in range(21):
        budget = get_forced_denoising_step_budget(
            canvas_length=256,
            forced_tpf=16.8,
            completed_token_slots=token_slots,
            completed_forward_slots=forward_slots,
            cohort_size=1,
        )
        budgets.append(budget)
        token_slots += 256
        forward_slots += budget

    assert [i for i, budget in enumerate(budgets, start=1) if budget == 16] == [
        3,
        7,
        11,
        15,
        19,
    ]
    assert token_slots == 5376
    assert forward_slots == 320
    assert token_slots / forward_slots == pytest.approx(16.8)


def test_forced_tpf_weights_changing_cohort_sizes():
    token_slots = 0
    forward_slots = 0

    for cohort_size in [8, 2, 7, 1, 4, 3]:
        budget = get_forced_denoising_step_budget(
            canvas_length=256,
            forced_tpf=16.8,
            completed_token_slots=token_slots,
            completed_forward_slots=forward_slots,
            cohort_size=cohort_size,
        )
        assert budget in (15, 16)
        token_slots += cohort_size * 256
        forward_slots += cohort_size * budget

    assert abs(forward_slots - token_slots / 16.8) <= 4
