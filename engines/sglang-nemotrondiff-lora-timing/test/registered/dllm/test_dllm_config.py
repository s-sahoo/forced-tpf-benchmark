from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from sglang.srt.dllm.config import DllmConfig


class TestDllmConfig(TestCase):
    def _make_server_args(
        self,
        max_running_requests: int,
        dp_size: int,
        enable_dp_attention: bool = False,
    ):
        return SimpleNamespace(
            dllm_algorithm="LinearSpec",
            dllm_algorithm_config=None,
            model_path="unused",
            revision=None,
            max_running_requests=max_running_requests,
            dp_size=dp_size,
            enable_dp_attention=enable_dp_attention,
        )

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_standard_dp_preserves_limit_per_replica(self, model_config):
        model_config.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["NemotronLabsDiffusionModel"]
            )
        )

        config = DllmConfig.from_server_args(
            self._make_server_args(max_running_requests=1024, dp_size=8)
        )

        self.assertEqual(config.max_running_requests, 1024)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_dp_attention_divides_server_limit(self, model_config):
        model_config.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["NemotronLabsDiffusionModel"]
            )
        )

        config = DllmConfig.from_server_args(
            self._make_server_args(
                max_running_requests=1024,
                dp_size=8,
                enable_dp_attention=True,
            )
        )

        self.assertEqual(config.max_running_requests, 128)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_single_dp_preserves_server_limit(self, model_config):
        model_config.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["NemotronLabsDiffusionModel"]
            )
        )

        config = DllmConfig.from_server_args(
            self._make_server_args(max_running_requests=1024, dp_size=1)
        )

        self.assertEqual(config.max_running_requests, 1024)


if __name__ == "__main__":
    main()
