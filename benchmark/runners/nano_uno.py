#!/usr/bin/env python3
"""Run one exact-cohort nano-vLLM-UNO forced-TPF GPU point."""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import torch

from timing import (
    FORCED_TPF_SCHEMA_VERSION,
    TIMING_SCOPE,
    ForcedTpfTask,
    cap_two_pass_tpf,
    standardize_result,
    write_json,
)
from nano_vllm_uno import LLM, SamplingParams
from nano_vllm_uno.config import Config

from nano_fa3 import configure_nano_fa3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="uno")
    parser.add_argument("--engine", default="nano_uno")
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention-backend", choices=["fa2", "fa3", "fa4"], default="fa3")
    parser.add_argument(
        "--fa3-provider",
        choices=["native", "sgl_kernel"],
        default=os.environ.get("NANO_FA3_PROVIDER", "native"),
    )
    parser.add_argument("--gated-lora-path")
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--tree-verify-size", type=int)
    parser.add_argument("--tree-candidate-top-k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--forced-tpf", type=float, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
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
        "--excluded-prompt-token-id",
        action="append",
        default=[],
        type=int,
        help="Token ID to exclude from deterministic synthetic prompts.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument(
        "--expected-prefill-batch-sizes", type=int, nargs="+", required=True
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


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_generate(
    llm: LLM,
    prompts: list[list[int]],
    sampling_params: SamplingParams,
) -> tuple[list[dict[str, object]], float]:
    """Time the public generation call, including admission and prefill."""
    sync_cuda()
    started_ns = time.perf_counter_ns()
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
        detokenize=False,
    )
    sync_cuda()
    return outputs, (time.perf_counter_ns() - started_ns) / 1e9


def main() -> None:
    args = parse_args()
    fa3_provider = configure_nano_fa3(args.fa3_provider)
    if args.block_size not in (1, 2, 4, 8, 16):
        raise ValueError("block_size must be one of 1, 2, 4, 8, 16")
    if args.tree_verify_size is not None and args.tree_verify_size < 1:
        raise ValueError("tree_verify_size must be positive")
    effective_forced_tpf = cap_two_pass_tpf(
        args.forced_tpf,
        args.block_size,
    )
    if effective_forced_tpf != args.forced_tpf:
        logging.warning(
            "Capping requested forced_tpf %.6f to the two-pass b=%d limit %.6f",
            args.forced_tpf,
            args.block_size,
            effective_forced_tpf,
        )
    if sum(args.expected_prefill_batch_sizes) != args.batch_size:
        raise ValueError("expected prefill batch sizes must sum to batch size")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    reserve_size = max(args.block_size, args.tree_verify_size or 0)
    engine_options = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.input_len + args.output_len + reserve_size,
        "fail_on_preemption": True,
        "cuda_graph_block_sizes": sorted({1, args.block_size}),
        "cuda_graph_batch_sizes": [args.batch_size],
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    if "attention_backend" in Config.__dataclass_fields__:
        engine_options["attention_backend"] = args.attention_backend
    if "decode_timing" in Config.__dataclass_fields__:
        engine_options["decode_timing"] = False
    if "max_diffusion_block_size" in Config.__dataclass_fields__:
        engine_options["max_diffusion_block_size"] = args.block_size
    else:
        engine_options["diffusion_block_size"] = args.block_size
    if args.gated_lora_path is not None:
        engine_options["gated_lora_path"] = args.gated_lora_path
    if args.tree_verify_size is not None:
        if "tree_verify_size" not in Config.__dataclass_fields__:
            raise RuntimeError("The selected nano-vLLM-UNO does not support trees")
        engine_options["tree_verify_size"] = args.tree_verify_size
        engine_options["tree_candidate_top_k"] = args.tree_candidate_top_k

    llm = LLM(
        args.model,
        **engine_options,
    )
    try:
        vocab_size = int(llm.model_runner.config.hf_config.vocab_size)
        if args.prompt_vocab_size != vocab_size:
            raise ValueError(
                f"prompt policy vocab size {args.prompt_vocab_size} does not "
                f"match model vocab size {vocab_size}"
            )
        excluded_prompt_token_ids = set(args.excluded_prompt_token_id)
        warmup_prompts = random_prompts(
            args.prompt_vocab_size,
            excluded_prompt_token_ids,
            args.batch_size,
            args.input_len,
            args.seed + 1,
        )
        timed_prompts = random_prompts(
            args.prompt_vocab_size,
            excluded_prompt_token_ids,
            args.batch_size,
            args.input_len,
            args.seed + 2,
        )
        common_sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "ignore_eos": True,
            "diffusion_block_size": args.block_size,
            "forced_accept_tpf": effective_forced_tpf,
            "mask_token_id": args.mask_token_id,
            "noise_mode": "random_uniform",
        }
        llm.generate(
            warmup_prompts,
            SamplingParams(
                max_tokens=args.warmup_output_len, **common_sampling
            ),
            use_tqdm=False,
            detokenize=False,
        )

        # Warmup/autotuning may consume RNG. Reset here so cold and warm caches
        # start the timed request from the same sampling and diffusion state.
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        two_pass_decoder = llm.model_runner.two_pass_decoder
        reset_forced_tpf_state = getattr(
            two_pass_decoder,
            "reset_forced_tpf_state",
            None,
        )
        if callable(reset_forced_tpf_state):
            reset_forced_tpf_state(args.seed)

        scheduler = llm.scheduler
        reset_stats = getattr(scheduler, "reset_stats", None)
        if callable(reset_stats):
            reset_stats()
        model_runner = llm.model_runner
        model_runner.cuda_graph_hits = 0
        model_runner.cuda_graph_misses = 0
        outputs, elapsed = timed_generate(
            llm,
            timed_prompts,
            SamplingParams(max_tokens=args.output_len, **common_sampling),
        )

        token_counts = [len(output["token_ids"]) for output in outputs]
        executed_tokens = sum(token_counts)
        expected_tokens = args.batch_size * args.output_len
        # The engine may finish a block a few tokens beyond max_tokens. Count
        # exactly the requested useful tokens while retaining executed
        # tokens and acceptance statistics separately.
        generated_tokens = sum(
            min(count, args.output_len) for count in token_counts
        )
        decode_stats = {
            key: sum(int(output["stats"][key]) for output in outputs)
            for key in outputs[0]["stats"]
        }
        forwards = decode_stats.get("forwards", 0)
        accepts = decode_stats.get("accepts", generated_tokens)
        acceptance_tpf = accepts / forwards if forwards else None
        measured_tpf = generated_tokens / forwards if forwards else None
        target_length = 2.0 * effective_forced_tpf
        fractional_length = target_length - math.floor(target_length)
        cycles = forwards / 2.0
        rounding_se = (
            math.sqrt(fractional_length * (1.0 - fractional_length) / cycles)
            / 2.0
            if cycles > 0
            else 0.0
        )
        tpf_tolerance = max(0.01, 6.0 * rounding_se)
        forced_tpf_matches = (
            measured_tpf is not None
            and abs(measured_tpf - effective_forced_tpf) <= tpf_tolerance
        )
        prefill_batch_sizes = list(scheduler.prefill_batch_sizes)
        decode_batch_sizes = list(scheduler.decode_batch_sizes)
        num_preemptions = int(scheduler.num_preemptions)
        first_decode_batch_size = (
            decode_batch_sizes[0] if decode_batch_sizes else None
        )
        full_cohort = (
            num_preemptions == 0
            and first_decode_batch_size == args.batch_size
            and prefill_batch_sizes == args.expected_prefill_batch_sizes
        )
        cuda_graph_hits = int(model_runner.cuda_graph_hits)
        cuda_graph_misses = int(model_runner.cuda_graph_misses)
        task = ForcedTpfTask(
            method=args.method,
            engine=args.engine,
            block_size=args.block_size,
            batch_size=args.batch_size,
            forced_tpf=args.forced_tpf,
        )
        result = {
            "schema_version": FORCED_TPF_SCHEMA_VERSION,
            "task": task.as_dict(),
            "method": args.method,
            "engine": args.engine,
            "algorithm": (
                "draft_tree"
                if args.tree_verify_size is not None
                else (
                    "autoregressive_block_1"
                    if args.block_size == 1
                    else "two_pass"
                )
            ),
            "block_size": args.block_size,
            "forward_width": args.tree_verify_size or args.block_size,
            "tree_verify_size": args.tree_verify_size,
            "tree_candidate_top_k": (
                args.tree_candidate_top_k
                if args.tree_verify_size is not None
                else None
            ),
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "prompt_token_id_min": 0,
            "prompt_token_id_max_exclusive": args.prompt_vocab_size,
            "excluded_prompt_token_ids": sorted(excluded_prompt_token_ids),
            "warmup_input_len": args.input_len,
            "requested_tpf": args.forced_tpf,
            "effective_forced_tpf": effective_forced_tpf,
            "measured_tpf": measured_tpf,
            "forced_tpf_error": (
                measured_tpf - effective_forced_tpf
                if measured_tpf is not None
                else None
            ),
            "forced_tpf_tolerance": tpf_tolerance,
            "forced_tpf_matches": forced_tpf_matches,
            "acceptance_tpf": acceptance_tpf,
            "generated_tokens": generated_tokens,
            "executed_tokens": executed_tokens,
            "expected_tokens": expected_tokens,
            "target_forwards": forwards,
            "accepts": accepts,
            "lookaheads": decode_stats.get("lookaheads", 0),
            "end_to_end_elapsed_s": elapsed,
            "timing_scope": TIMING_SCOPE,
            "detailed_decode_tracing": False,
            "aggregate_tps": generated_tokens / elapsed,
            "per_user_tps": (
                generated_tokens / elapsed / args.batch_size
            ),
            "per_request_token_counts": token_counts,
            "per_request_retractions": None,
            "full_batch_resident": full_cohort,
            "all_requests_entered_decode_together": (
                first_decode_batch_size == args.batch_size
            ),
            "first_decode_batch_size": first_decode_batch_size,
            "cohort_validation": "first_decode_batch_and_no_preemption",
            "num_preemptions": num_preemptions,
            "prefill_batch_sizes": prefill_batch_sizes,
            "expected_prefill_batch_sizes": (
                args.expected_prefill_batch_sizes
            ),
            "cuda_graph_batch_sizes": [args.batch_size],
            "cuda_graph_hits": cuda_graph_hits,
            "cuda_graph_misses": cuda_graph_misses,
            "cuda_graph_hit_rate": (
                cuda_graph_hits / (cuda_graph_hits + cuda_graph_misses)
                if cuda_graph_hits + cuda_graph_misses
                else None
            ),
            "cuda_graph_memory": model_runner.cuda_graph_memory,
            "attention_backend": args.attention_backend,
            "fa3_provider": fa3_provider,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "model": args.model,
            "lora_path": args.gated_lora_path,
            "seed": args.seed,
            "status": (
                "complete"
                if (
                    executed_tokens >= expected_tokens
                    and full_cohort
                    and forwards > 0
                    and forced_tpf_matches
                    and cuda_graph_misses == 0
                )
                else "invalid"
            ),
        }
        result = standardize_result(result)
        write_json(args.result_path, result)

        if executed_tokens < expected_tokens:
            raise RuntimeError(
                f"Expected at least {expected_tokens} tokens, "
                f"got {executed_tokens}"
            )
        if not full_cohort:
            raise RuntimeError(
                "Timed requests did not decode as one resident cohort: "
                f"prefills={prefill_batch_sizes}, "
                f"first_decode={first_decode_batch_size}, "
                f"preemptions={num_preemptions}"
            )
        if forwards <= 0:
            raise RuntimeError("Per-sequence forward counters were absent")
        if not forced_tpf_matches:
            raise RuntimeError(
                "Measured forced TPF missed its target: "
                f"target={effective_forced_tpf:.6f}, "
                f"measured={measured_tpf:.6f}, "
                f"tolerance={tpf_tolerance:.6f}"
            )
        if cuda_graph_misses:
            raise RuntimeError(
                f"Timed decode had {cuda_graph_misses} CUDA graph misses"
            )
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
