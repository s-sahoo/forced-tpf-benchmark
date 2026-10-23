import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionResponseChoice,
)
from sglang.srt.entrypoints.openai.serving_chat import (
    _choice_meta_info as chat_choice_meta_info,
)
from sglang.srt.entrypoints.openai.serving_completions import (
    _choice_meta_info as completion_choice_meta_info,
)


class TestDllmOutputMetadata(unittest.TestCase):
    def test_choice_metadata_is_filtered_and_serialized(self):
        meta_info = {
            "dllm_generated_tokens": 51,
            "dllm_decode_forward_passes": 10,
            "dllm_decode_tpf": 5.1,
            "internal_only": "ignored",
        }
        expected = {
            "dllm_generated_tokens": 51,
            "dllm_decode_forward_passes": 10,
            "dllm_decode_tpf": 5.1,
        }

        self.assertEqual(chat_choice_meta_info(meta_info), expected)
        self.assertEqual(completion_choice_meta_info(meta_info), expected)

        chat_choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="answer"),
            finish_reason="length",
            meta_info=expected,
        )
        completion_choice = CompletionResponseChoice(
            index=0,
            text="answer",
            finish_reason="length",
            meta_info=expected,
        )

        self.assertEqual(chat_choice.model_dump()["meta_info"], expected)
        self.assertEqual(completion_choice.model_dump()["meta_info"], expected)


if __name__ == "__main__":
    unittest.main()
