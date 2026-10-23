from types import SimpleNamespace
import unittest

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder


class TestDllmFullBlockAdmission(unittest.TestCase):
    BLOCK_SIZE = 32

    @staticmethod
    def _adder(remaining_tokens: int) -> PrefillAdder:
        adder = object.__new__(PrefillAdder)
        adder.dllm_config = SimpleNamespace(block_size=32)
        adder.dllm_block_size = 32
        adder.rem_dllm_tokens = 32 * 1024
        adder.rem_total_token_offset = 0
        adder.cur_rem_token_offset = 0
        adder.rem_input_tokens = 32 * 1024
        adder.rem_chunk_tokens = None
        adder.can_run_list = []
        adder.log_hit_tokens = 0
        adder.log_input_tokens = 0
        adder.page_size = 1
        adder.is_hybrid_swa = False
        adder.is_hybrid_ssm_cache = False
        adder.new_token_ratio = 1.0
        adder.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
        adder.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=lambda: remaining_tokens
        )
        return adder

    @staticmethod
    def _req(extend_input_len: int = 32, *, ignore_eos: bool = True):
        return SimpleNamespace(
            extend_input_len=extend_input_len,
            prefix_indices=range(10),
            fill_ids=list(range(10 + extend_input_len)),
            output_ids=[],
            sampling_params=SimpleNamespace(
                max_new_tokens=5000,
                ignore_eos=ignore_eos,
            ),
        )

    def test_prompt_admission_reserves_full_ignore_eos_output(self):
        adder = self._adder(remaining_tokens=4500)
        req = self._req()

        self.assertEqual(
            adder.add_dllm_prompt_cache_req(req),
            AddReqResult.NO_TOKEN,
        )
        self.assertEqual(adder.can_run_list, [])

    def test_prompt_admission_keeps_clipped_eos_estimate(self):
        adder = self._adder(remaining_tokens=4500)
        req = self._req(ignore_eos=False)

        self.assertEqual(
            adder.add_dllm_prompt_cache_req(req),
            AddReqResult.CONTINUE,
        )
        self.assertEqual(adder.can_run_list, [req])

    def test_running_ignore_eos_reservation_is_not_clipped(self):
        adder = self._adder(remaining_tokens=10_000)
        req = self._req()

        self.assertEqual(adder._get_running_request_total_token_offset(req), 5000)

    def test_running_eos_estimate_uses_standard_ratio(self):
        adder = self._adder(remaining_tokens=10_000)
        adder.new_token_ratio = 0.25
        req = self._req(ignore_eos=False)

        self.assertEqual(adder._get_running_request_total_token_offset(req), 1024)

    def test_defers_request_when_only_partial_block_fits(self):
        adder = self._adder(remaining_tokens=self.BLOCK_SIZE - 1)
        req = self._req()

        result = adder.add_dllm_staging_req(req)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(req.extend_input_len, self.BLOCK_SIZE)
        self.assertEqual(len(req.fill_ids), 10 + self.BLOCK_SIZE)

    def test_admits_exactly_one_full_block(self):
        adder = self._adder(remaining_tokens=self.BLOCK_SIZE)
        req = self._req()

        result = adder.add_dllm_staging_req(req)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(req.extend_input_len, self.BLOCK_SIZE)

    def test_rejects_malformed_staging_shape(self):
        adder = self._adder(remaining_tokens=self.BLOCK_SIZE)

        with self.assertRaisesRegex(ValueError, "exactly 32"):
            adder.add_dllm_staging_req(self._req(extend_input_len=2))


if __name__ == "__main__":
    unittest.main()
