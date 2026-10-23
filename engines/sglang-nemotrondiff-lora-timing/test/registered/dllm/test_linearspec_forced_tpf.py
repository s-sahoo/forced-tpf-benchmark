from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

import unittest
from fractions import Fraction
from random import Random
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.linear_spec import LinearSpec, _forced_emit_count
from sglang.srt.dllm.config import DllmConfig


class TestLinearSpecForcedTPF(unittest.TestCase):
    def test_integer_tpf(self):
        rng = Random(0)
        emitted = 0
        counts = []
        for completed_forwards in range(0, 8, 2):
            count = _forced_emit_count(
                Fraction("5"), 12, completed_forwards, emitted, rng
            )
            counts.append(count)
            emitted += count
        self.assertEqual(counts, [10, 10, 10, 10])

    def test_fractional_tpf_stochastically_rounds_cumulative_target(self):
        class FixedRng:
            def __init__(self, value):
                self.value = value

            def random(self):
                return self.value

        target = Fraction("5.25")
        self.assertEqual(_forced_emit_count(target, 12, 0, 0, FixedRng(0.49)), 11)
        self.assertEqual(_forced_emit_count(target, 12, 0, 0, FixedRng(0.50)), 10)

    def test_cumulative_target_compensates_for_short_cycle(self):
        rng = Random(0)
        target = Fraction("5")
        emitted = 0
        max_emits = [4, 12, 12, 12]
        counts = []
        for cycle, max_emit in enumerate(max_emits):
            count = _forced_emit_count(target, max_emit, 2 * cycle, emitted, rng)
            counts.append(count)
            emitted += count

        self.assertEqual(counts, [4, 12, 12, 12])
        self.assertEqual(Fraction(emitted, 2 * len(counts)), target)

    def test_run_keeps_full_forwards_and_forces_verified_input_prefix(self):
        block_size = 12
        mask_id = 19
        vocab_size = 20
        config = DllmConfig(
            algorithm="LinearSpec",
            algorithm_config={"forced_tpf": 5.25, "forced_tpf_seed": 1},
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=1,
            max_steps=block_size,
            causal_context=True,
        )
        decoder = LinearSpec(config)
        decoder._seed_tokens["req"] = 7

        def make_logits(tokens):
            logits = torch.full((block_size, vocab_size), -100.0)
            logits[torch.arange(block_size), torch.tensor(tokens)] = 100.0
            return logits

        # Verification deliberately rejects at the first speculative token.
        # Forced mode must nevertheless retain the already-computed verify
        # input prefix, rather than inventing correction tokens with stale KV.
        draft_tokens = list(range(1, block_size + 1))
        verify_tokens = [15] * block_size

        class FakeRunner:
            def __init__(self):
                self.calls = 0
                self.model_config = SimpleNamespace(
                    # The algorithm must transport this token unchanged. EOS
                    # policy belongs to the scheduler, which knows ignore_eos.
                    hf_config=SimpleNamespace(eos_token_id=2)
                )

            def forward(self, forward_batch, pp_proxy_tensors=None):
                source = draft_tokens if self.calls % 2 == 0 else verify_tokens
                self.calls += 1
                return SimpleNamespace(
                    logits_output=SimpleNamespace(full_logits=make_logits(source)),
                    can_run_graph=False,
                )

        runner = FakeRunner()

        def run_cycle():
            batch = SimpleNamespace(
                batch_size=1,
                input_ids=torch.full((block_size,), mask_id, dtype=torch.int64),
                rids=["req"],
                dllm_causal_kv_update=False,
            )
            _, outputs, _ = decoder.run(runner, batch)
            return outputs[0]

        first = run_cycle()
        second = run_cycle()

        self.assertEqual(runner.calls, 4)
        self.assertEqual(first.numel(), 11)
        self.assertEqual(second.numel(), 10)
        self.assertEqual(first[0].item(), 7)
        self.assertTrue(torch.equal(first[1:], torch.arange(2, 12)))
        self.assertEqual(Fraction(first.numel() + second.numel(), 4), Fraction(21, 4))

        decoder.release_request_state("req")
        self.assertNotIn("req", decoder._seed_tokens)
        self.assertNotIn("req", decoder._forced_num_forwards)
        self.assertNotIn("req", decoder._forced_num_tokens)


if __name__ == "__main__":
    unittest.main()
