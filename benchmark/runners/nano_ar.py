#!/usr/bin/env python3
"""Measure one nano-vLLM-UNO block-1 AR GPU point."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from timing import (
    AR_SCHEMA_VERSION,
    NANO_NORMALIZATION_DIVISOR,
    TIMING_SCOPE,
    standardize_result,
    summarize_nano_ar_trace,
    write_json,
)
from nano_vllm_uno import LLM, SamplingParams

from nano_fa3 import configure_nano_fa3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--fa3-provider",
        choices=["native", "sgl_kernel"],
        default=os.environ.get("NANO_FA3_PROVIDER", "native"),
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument("--input-len", type=int, default=1000)
    parser.add_argument("--output-len", type=int, default=5000)
    parser.add_argument("--warmup-output-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    parser.add_argument("--prompt-vocab-size", type=int, required=True)
    parser.add_argument(
        "--excluded-prompt-token-id", action="append", default=[], type=int
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--detailed-tracing",
        action="store_true",
        help="Opt in to per-forward CUDA-event diagnostics.",
    )
    return parser.parse_args()


def random_prompts(
    vocab_size: int,
    excluded_token_ids: set[int],
    batch_size: int,
    input_len: int,
    seed: int,
) -> list[list[int]]:
    excluded_token_ids = {
        token_id for token_id in excluded_token_ids if 0 <= token_id < vocab_size
    }
    if len(excluded_token_ids) == vocab_size:
        raise ValueError("prompt token filter excludes the complete vocabulary")
    rng = random.Random(seed)
    def draw() -> int:
        while (token_id := rng.randrange(vocab_size)) in excluded_token_ids:
            pass
        return token_id
    return [
        [draw() for _ in range(input_len)]
        for _ in range(batch_size)
    ]


def timed_generate(
    llm: LLM,
    prompts: list[list[int]],
    sampling_params: SamplingParams,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    llm.reset_decode_timing()
    llm.model_runner.cuda_graph_hits = 0
    llm.model_runner.cuda_graph_misses = 0
    llm.scheduler.reset_stats()
    torch.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
        detokenize=False,
    )
    torch.cuda.synchronize()
    elapsed_s = (time.perf_counter_ns() - started_ns) / 1e9
    trace = llm.get_decode_timing()
    return outputs, elapsed_s, trace


def main() -> None:
    args = parse_args()
    fa3_provider = configure_nano_fa3(args.fa3_provider)
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    trace_path = args.trace_path or args.result_path.with_name(
        args.result_path.stem + ".trace.json"
    )
    prefill_budget = max(
        args.input_len * args.batch_size,
        args.input_len + args.output_len + 1,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.input_len + args.output_len + 1,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=prefill_budget,
        fail_on_preemption=True,
        max_diffusion_block_size=1,
        cuda_graph_block_sizes=[1],
        cuda_graph_batch_sizes=[args.batch_size],
        decode_timing=args.detailed_tracing,
    )
    try:
        vocab_size = int(llm.model_runner.config.hf_config.vocab_size)
        if args.prompt_vocab_size != vocab_size:
            raise ValueError(
                f"prompt policy vocab size {args.prompt_vocab_size} does not "
                f"match model vocab size {vocab_size}"
            )
        excluded_prompt_token_ids = set(args.excluded_prompt_token_id)
        sampling_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "ignore_eos": True,
            "diffusion_block_size": 1,
            "mask_token_id": args.mask_token_id,
            "noise_mode": "random_uniform",
        }
        llm.generate(
            random_prompts(
                args.prompt_vocab_size,
                excluded_prompt_token_ids,
                args.batch_size,
                min(args.input_len, 256),
                args.seed + 1,
            ),
            SamplingParams(
                max_tokens=args.warmup_output_len,
                **sampling_kwargs,
            ),
            use_tqdm=False,
            detokenize=False,
        )

        # Keep the timed request identical between repetitions.
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        outputs, request_elapsed_s, trace = timed_generate(
            llm,
            random_prompts(
                args.prompt_vocab_size,
                excluded_prompt_token_ids,
                args.batch_size,
                args.input_len,
                args.seed + 2,
            ),
            SamplingParams(max_tokens=args.output_len, **sampling_kwargs),
        )
        trace.update(
            {
                "schema_version": AR_SCHEMA_VERSION,
                "engine": "nano_uno_ar",
                "batch_size": args.batch_size,
                "block_size": 1,
                "forwards_per_raw_cycle": 2,
                "tokens_per_raw_cycle": 2,
                "normalization_divisor": NANO_NORMALIZATION_DIVISOR,
            }
        )
        token_counts = [len(output["token_ids"]) for output in outputs]
        generated_tokens = sum(
            min(count, args.output_len) for count in token_counts
        )
        stats = {
            key: sum(int(output["stats"][key]) for output in outputs)
            for key in outputs[0]["stats"]
        }
        graph_hits = int(llm.model_runner.cuda_graph_hits)
        graph_misses = int(llm.model_runner.cuda_graph_misses)
        prefill_batch_sizes = list(llm.scheduler.prefill_batch_sizes)
        decode_batch_sizes = list(llm.scheduler.decode_batch_sizes)
        num_preemptions = int(llm.scheduler.num_preemptions)
        full_batch_resident = (
            num_preemptions == 0
            and bool(decode_batch_sizes)
            and decode_batch_sizes[0] == args.batch_size
            and prefill_batch_sizes == [args.batch_size]
        )
        timing = (
            {
                "enabled": False,
                "scope": "end_to_end_only",
                "all_cuda_graph_hits": graph_misses == 0,
            }
            if not args.detailed_tracing
            else summarize_nano_ar_trace(trace, batch_size=args.batch_size)
        )
        result = {
            "schema_version": AR_SCHEMA_VERSION,
            "engine": "nano_uno_ar",
            "algorithm": "normal_block_1_two_pass_cycle_divided_by_2",
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "requested_tpf": None,
            "measured_tpf": 1.0,
            "executed_tokens": sum(token_counts),
            "generated_tokens": generated_tokens,
            "expected_tokens": args.batch_size * args.output_len,
            "per_request_token_counts": token_counts,
            "end_to_end_elapsed_s": request_elapsed_s,
            "aggregate_tps": generated_tokens / request_elapsed_s,
            "per_user_tps": generated_tokens / request_elapsed_s / args.batch_size,
            "timing_scope": TIMING_SCOPE,
            "decode_stats": stats,
            "raw_forwards_per_cycle": 2,
            "raw_tokens_per_cycle": 2,
            "reported_normalization_divisor": 2,
            "reported_unit": "one_forward_and_one_generated_token",
            "cuda_graph_hits": graph_hits,
            "cuda_graph_misses": graph_misses,
            "cuda_graph_memory": llm.model_runner.cuda_graph_memory,
            "full_batch_resident": full_batch_resident,
            "num_preemptions": num_preemptions,
            "per_request_retractions": None,
            "prefill_batch_sizes": prefill_batch_sizes,
            "decode_batch_sizes": decode_batch_sizes,
            "fa3_provider": fa3_provider,
            "timing": timing,
            "trace_path": str(trace_path),
            "model": args.model,
            "optimizations": {
                "decode_cuda_graph": True,
                "attention_backend": "fa3",
                "fa3_provider": fa3_provider,
                "sampling_path": "native_sparse_top_k_top_p",
                "detailed_decode_tracing": args.detailed_tracing,
            },
            "seed": args.seed,
            "status": (
                "complete"
                if (
                    generated_tokens == args.batch_size * args.output_len
                    and full_batch_resident
                    and graph_misses == 0
                )
                else "invalid"
            ),
        }
        write_json(trace_path, trace)
        result = standardize_result(result)
        write_json(args.result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

        if not timing["all_cuda_graph_hits"] or graph_misses:
            raise RuntimeError(
                f"timed nano decode had CUDA graph misses: {graph_misses}"
            )
        if not full_batch_resident:
            raise RuntimeError(
                "timed requests did not decode as one resident cohort: "
                f"prefill={prefill_batch_sizes}, decode={decode_batch_sizes}, "
                f"preemptions={num_preemptions}"
            )
        expected_forwards = (
            generated_tokens
            if not args.detailed_tracing
            else 2 * len(trace["step_wall_ms"]) * args.batch_size
        )
        if stats["forwards"] != expected_forwards:
            raise RuntimeError(
                "nano block-1 trace did not execute exactly two forwards "
                "per scheduler cycle: "
                f"expected={expected_forwards}, actual={stats['forwards']}"
            )
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
