from __future__ import annotations

import argparse
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nano_vllm_uno.engine.async_engine import AsyncLLMEngine, GenerationResult
from nano_vllm_uno.sampling_params import SamplingParams


class ChatCompletionRequest(BaseModel):
    """Minimal request contract for one server-configured generation mode."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None

    @model_validator(mode="after")
    def require_one_prompt_form(self):
        if (self.messages is None) == (self.prompt is None):
            raise ValueError("Exactly one of messages or prompt must be provided")
        return self


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nano-vllm-uno"


def _build_prompt(
    request: ChatCompletionRequest,
    tokenizer: Any,
) -> str:
    if request.messages is not None:
        try:
            return tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to apply the model chat template: {exc}",
            ) from exc
    assert request.prompt is not None
    return request.prompt


def _openai_response(
    result: GenerationResult,
    *,
    model_name: str,
    prompt_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    completion_tokens = len(result.token_ids)
    finish_reason = "length" if completion_tokens >= max_tokens else "stop"
    return {
        "id": f"chatcmpl-{result.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_app(
    async_engine: AsyncLLMEngine,
    *,
    model_name: str,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async_engine.start()
        try:
            yield
        finally:
            await asyncio.to_thread(async_engine.shutdown)

    app = FastAPI(
        title="nano-vLLM-UNO API Server",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        if async_engine.is_running:
            return {"status": "ok"}
        error = async_engine.fatal_error
        detail = str(error) if error is not None else "engine is not running"
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": detail},
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [ModelCard(id=model_name)],
        }

    @app.post("/v1/chat/completions")
    async def chat_completion(request: ChatCompletionRequest):
        if request.model != model_name:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model {request.model!r}",
            )

        prompt = _build_prompt(request, async_engine.engine.tokenizer)
        prompt_token_ids = async_engine.engine.tokenizer.encode(prompt)
        try:
            future = async_engine.submit(prompt_token_ids)
            result = await asyncio.wrap_future(future)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _openai_response(
            result,
            model_name=model_name,
            prompt_tokens=len(prompt_token_ids),
            max_tokens=async_engine.sampling_params.max_tokens,
        )

    return app


def _comma_separated_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve nano-vLLM-UNO over HTTP")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--attention-backend",
        choices=["fa2", "fa3", "fa4"],
        default="fa3",
    )
    parser.add_argument("--gated-lora-path")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tree-verify-size", type=int)
    parser.add_argument("--tree-candidate-top-k", type=int, default=16)
    parser.add_argument(
        "--cuda-graph-batch-sizes",
        type=_comma_separated_ints,
    )
    parser.add_argument("--decode-timing", action="store_true")

    sampling = parser.add_argument_group("server-wide sampling")
    sampling.add_argument("--max-tokens", type=int, default=64)
    sampling.add_argument("--temperature", type=float, default=1.0)
    sampling.add_argument("--top-p", type=float)
    sampling.add_argument("--top-k", type=int)
    sampling.add_argument("--ignore-eos", action="store_true")
    sampling.add_argument(
        "--stop-token-ids",
        type=_comma_separated_ints,
    )
    sampling.add_argument("--diffusion-block-size", type=int, default=1)
    sampling.add_argument("--mask-token-id", type=int)
    sampling.add_argument(
        "--noise-mode",
        choices=["random_uniform", "deterministic_uniform", "mask"],
        default="random_uniform",
    )
    sampling.add_argument("--noise-salt", type=int)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    from nano_vllm_uno.engine.llm_engine import LLMEngine

    args = _parse_args()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ignore_eos=args.ignore_eos,
        stop_token_ids=args.stop_token_ids,
        diffusion_block_size=args.diffusion_block_size,
        mask_token_id=args.mask_token_id,
        noise_mode=args.noise_mode,
        noise_salt=args.noise_salt,
    )
    engine = LLMEngine(
        args.model,
        tokenizer_path=args.tokenizer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        attention_backend=args.attention_backend,
        gated_lora_path=args.gated_lora_path,
        enforce_eager=args.enforce_eager,
        max_diffusion_block_size=args.diffusion_block_size,
        cuda_graph_block_sizes=sorted({1, args.diffusion_block_size}),
        cuda_graph_batch_sizes=args.cuda_graph_batch_sizes,
        tree_verify_size=args.tree_verify_size,
        tree_candidate_top_k=args.tree_candidate_top_k,
        decode_timing=args.decode_timing,
    )
    async_engine = AsyncLLMEngine(engine, sampling_params)
    app = create_app(
        async_engine,
        model_name=args.served_model_name or args.model,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
