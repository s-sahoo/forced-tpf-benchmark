#!/usr/bin/env python3
"""Measure one stationary two-pass UNO cycle on Nano-vLLM-UNO."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from typing import Any

from timing import (
    standardize_result,
    summarize_uno_cycle_trace,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=1000)
    parser.add_argument("--output-len", type=int, default=5000)
    parser.add_argument("--warmup-output-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt-vocab-size", type=int, required=True)
    parser.add_argument(
        "--excluded-prompt-token-id",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--fa3-provider",
        choices=("native", "sgl_kernel"),
        default="sgl_kernel",
    )
    return parser.parse_args()


def random_prompts(args: argparse.Namespace, length: int, seed: int) -> list[list[int]]:
    excluded = set(args.excluded_prompt_token_id)
    rng = random.Random(seed)

    def draw() -> int:
        while (token := rng.randrange(args.prompt_vocab_size)) in excluded:
            pass
        return token

    return [[draw() for _ in range(length)] for _ in range(args.batch_size)]


def run_nano(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nano_fa3 import configure_nano_fa3
    from nano_vllm_uno import LLM, SamplingParams

    configure_nano_fa3(args.fa3_provider)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    budget = max(
        args.input_len + args.output_len + args.block_size,
        args.batch_size * args.input_len,
    )
    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=args.input_len + args.output_len + args.block_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=budget,
        fail_on_preemption=True,
        gated_lora_path=args.lora_path,
        max_diffusion_block_size=args.block_size,
        cuda_graph_block_sizes=[1, args.block_size],
        cuda_graph_batch_sizes=[args.batch_size],
        attention_backend="fa3",
        decode_timing=True,
    )
    try:
        sampling = dict(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            ignore_eos=True,
            diffusion_block_size=args.block_size,
            mask_token_id=151669,
            noise_mode="random_uniform",
        )
        llm.generate(
            random_prompts(args, min(args.input_len, 256), args.seed + 1),
            SamplingParams(max_tokens=args.warmup_output_len, **sampling),
            use_tqdm=False,
            detokenize=False,
        )
        llm.reset_decode_timing()
        llm.model_runner.cuda_graph_hits = 0
        llm.model_runner.cuda_graph_misses = 0
        llm.scheduler.reset_stats()
        outputs = llm.generate(
            random_prompts(args, args.input_len, args.seed + 2),
            SamplingParams(max_tokens=args.output_len, **sampling),
            use_tqdm=False,
            detokenize=False,
        )
        trace = llm.get_decode_timing()
        trace.update(
            {
                "engine": "nano",
                "block_size": args.block_size,
                "batch_size": args.batch_size,
                "lm_head_in_cuda_graph": False,
            }
        )
        timing = summarize_uno_cycle_trace(
            trace, engine="nano", batch_size=args.batch_size
        )
        token_counts = [len(output["token_ids"]) for output in outputs]
        decode_batches = list(llm.scheduler.decode_batch_sizes)
        full_cohort = (
            llm.scheduler.num_preemptions == 0
            and bool(decode_batches)
            and decode_batches[0] == args.batch_size
        )
        return {
            "engine": "nano",
            "token_counts": token_counts,
            "generated_tokens": sum(
                min(count, args.output_len) for count in token_counts
            ),
            "full_batch_resident": full_cohort,
            "num_preemptions": int(llm.scheduler.num_preemptions),
            "cuda_graph_hits": int(llm.model_runner.cuda_graph_hits),
            "cuda_graph_misses": int(llm.model_runner.cuda_graph_misses),
            "timing": timing,
            "trace": trace,
            "fa3_provider": args.fa3_provider,
        }
    finally:
        llm.exit()


def main() -> None:
    args = parse_args()
    if args.block_size not in (2, 4, 8, 16):
        raise ValueError("block size must be 2, 4, 8, or 16")
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.trace_path.parent.mkdir(parents=True, exist_ok=True)
    measured = run_nano(args)
    trace = measured.pop("trace")
    expected_tokens = args.batch_size * args.output_len
    result = {
        "schema_version": 1,
        "status": "complete",
        "algorithm": "uno_cycle_timing",
        "model": args.model,
        "lora_path": args.lora_path,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "requested_tpf": None,
        "measured_tpf": None,
        "timing_scope": "stationary_full_cohort_uno_cycles",
        "expected_tokens": expected_tokens,
        "trace_path": str(args.trace_path),
        "hostname": platform.node(),
        **measured,
    }
    if (
        result["generated_tokens"] != expected_tokens
        or not result["full_batch_resident"]
        or result["num_preemptions"]
        or result["cuda_graph_misses"]
    ):
        result["status"] = "invalid"
    write_json(args.trace_path, trace)
    result = standardize_result(result)
    write_json(args.result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("invalid Figure 2 timing result")


if __name__ == "__main__":
    main()
