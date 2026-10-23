import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.dllm.algorithm.linear_spec import LinearSpec


class _Event:
    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.25


class TestLinearSpecDecodeTiming(unittest.TestCase):
    def test_records_and_resolves_one_forward_without_hot_path_sync(self):
        decoder = LinearSpec.__new__(LinearSpec)
        decoder.block_size = 32
        decoder._decode_timing_enabled = True
        decoder._decode_timing_step = 1
        decoder._decode_forward_events = []
        runner = SimpleNamespace(
            forward=lambda batch, pp_proxy_tensors=None: SimpleNamespace(
                can_run_graph=True
            )
        )
        batch = SimpleNamespace(batch_size=2)

        with patch.object(torch.cuda, "Event", return_value=_Event()), patch.object(
            torch.cuda, "synchronize"
        ) as synchronize:
            decoder._timed_forward(runner, batch, "draft")
            synchronize.assert_not_called()
            timing = decoder.get_decode_timing()
            synchronize.assert_called_once()

        self.assertEqual(
            timing["forward_records"],
            [
                {
                    "step_index": 1,
                    "role": "draft",
                    "batch_size": 2,
                    "forward_width": 32,
                    "cuda_graph_hit": True,
                    "lm_head_in_graph": True,
                    "gpu_ms": 1.25,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
