# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.outputs import CompletionOutput, RequestOutput

pytestmark = pytest.mark.cpu_test


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def test_completion_output_dllm_meta_info():
    output = CompletionOutput(
        index=0,
        text="output",
        token_ids=[1, 2, 3],
        cumulative_logprob=None,
        logprobs=None,
        dllm_generated_tokens=256,
        dllm_decode_forward_passes=15,
        dllm_decode_tpf=256 / 15,
    )

    assert output.dllm_meta_info() == {
        "dllm_generated_tokens": 256,
        "dllm_decode_forward_passes": 15,
        "dllm_decode_tpf": 256 / 15,
    }
