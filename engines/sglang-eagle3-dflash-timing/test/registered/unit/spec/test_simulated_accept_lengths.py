import unittest

import torch

from sglang.srt.speculative.synthetic_acceptance import (
    sample_simulated_accept_lengths,
)


class TestSimulatedAcceptLengths(unittest.TestCase):
    def test_fractional_rounding_is_per_request(self):
        uniforms = torch.tensor([0.10, 0.24, 0.25, 0.90])
        actual = sample_simulated_accept_lengths(
            simulate_acc_len=2.25,
            simulate_acc_method="match-expected",
            max_len=4,
            batch_size=4,
            device=uniforms.device,
            uniform_samples=uniforms,
        )
        torch.testing.assert_close(
            actual, torch.tensor([3, 3, 2, 2], dtype=torch.int32)
        )

    def test_integer_and_bounds(self):
        actual = sample_simulated_accept_lengths(
            simulate_acc_len=9.0,
            simulate_acc_method="match-expected",
            max_len=4,
            batch_size=3,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            actual, torch.full((3,), 4, dtype=torch.int32)
        )

if __name__ == "__main__":
    unittest.main()
