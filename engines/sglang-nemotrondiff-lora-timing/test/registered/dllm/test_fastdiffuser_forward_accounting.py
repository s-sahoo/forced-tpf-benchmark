import unittest
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.fastdiffuser import FastDiffuser
from sglang.srt.dllm.config import DllmConfig


class _FakeRunner:
    def __init__(self, block_size: int, vocab_size: int):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.calls = 0
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(eos_token_id=None)
        )

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.calls += 1
        logits = torch.zeros(
            forward_batch.input_ids.numel(), self.vocab_size, dtype=torch.float32
        )
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits),
            can_run_graph=False,
        )


def _config(*, block_size: int, max_steps: int, threshold):
    return DllmConfig(
        algorithm="FastDiffuser",
        algorithm_config={"threshold": threshold},
        block_size=block_size,
        mask_id=5,
        max_running_requests=1,
        max_steps=max_steps,
        causal_context=True,
    )


def _batch(input_ids):
    return SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor(input_ids, dtype=torch.int64),
        dllm_causal_kv_update=False,
    )


class TestFastDiffuserForwardAccounting(unittest.TestCase):
    def test_counts_denoising_and_final_kv_update(self):
        decoder = FastDiffuser(_config(block_size=4, max_steps=2, threshold=None))
        runner = _FakeRunner(block_size=4, vocab_size=6)

        decoder.run(runner, _batch([5, 5, 5, 5]))

        self.assertEqual(runner.calls, 3)
        self.assertEqual(decoder._last_decode_forward_passes, [3])

    def test_counts_extra_kv_update_after_force_commit(self):
        decoder = FastDiffuser(_config(block_size=4, max_steps=1, threshold=0.9))
        runner = _FakeRunner(block_size=4, vocab_size=6)

        decoder.run(runner, _batch([5, 5, 5, 5]))

        self.assertEqual(runner.calls, 3)
        self.assertEqual(decoder._last_decode_forward_passes, [3])

    def test_prefill_clears_decode_counts(self):
        decoder = FastDiffuser(_config(block_size=4, max_steps=1, threshold=None))
        decoder._last_decode_forward_passes = [7]
        runner = _FakeRunner(block_size=4, vocab_size=6)

        decoder.run(runner, _batch([1, 2, 3, 4]))

        self.assertEqual(decoder._last_decode_forward_passes, [])


if __name__ == "__main__":
    unittest.main()
