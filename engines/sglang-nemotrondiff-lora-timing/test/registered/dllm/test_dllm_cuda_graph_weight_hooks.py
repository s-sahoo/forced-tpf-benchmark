from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

import unittest

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner


class TestDllmCudaGraphWeightHooks(unittest.TestCase):
    def test_dispatches_draft_and_verify_hooks_in_order(self):
        runner = object.__new__(CudaGraphRunner)
        calls = []
        runner._dllm_pre_draft_hook = lambda: calls.append("draft")
        runner._dllm_pre_verify_hook = lambda: calls.append("verify")

        runner._run_dllm_weight_hook("draft")
        runner._run_dllm_weight_hook("verify")

        self.assertEqual(calls, ["draft", "verify"])

    def test_missing_hook_is_a_noop(self):
        runner = object.__new__(CudaGraphRunner)
        runner._run_dllm_weight_hook("draft")


if __name__ == "__main__":
    unittest.main()
