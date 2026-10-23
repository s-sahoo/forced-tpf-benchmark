#!/usr/bin/env python3
"""Run one exact-capacity speculative GPU point on SGLang."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import sglang as sgl

from timing import (
    FORCED_TPF_SCHEMA_VERSION,
    TIMING_SCOPE,
    ForcedTpfTask,
    standardize_result,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=("eagle3", "dflash"), required=True
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--block-size", type=int, required=True)
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
        "--excluded-prompt-token-id", action="append", default=[], type=int
    )
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--draft-attention-backend", default="fa3")
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="Run the synchronous SGLang scheduler for an overlap A/B test.",
    )
    parser.add_argument("--speculative-num-steps", type=int)
    parser.add_argument("--speculative-eagle-topk", type=int)
    parser.add_argument("--speculative-num-draft-tokens", type=int)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--max-prefill-tokens", type=int, default=None)
    parser.add_argument(
        "--allow-batch-splitting",
        action="store_true",
        help=(
            "Measure the complete request cohort even when KV capacity forces "
            "the scheduler to split or preempt it."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.block_size not in (2, 4, 8, 16):
        raise ValueError("block_size must be one of 2, 4, 8, 16")
    if not 1.0 <= args.forced_tpf <= args.block_size:
        raise ValueError(
            f"forced_tpf must be in [1, block_size], got {args.forced_tpf}"
        )
    if not args.draft_model:
        raise ValueError(f"{args.algorithm} requires --draft-model")
    if args.algorithm == "eagle3":
        tree_args = (
            args.speculative_num_steps,
            args.speculative_eagle_topk,
            args.speculative_num_draft_tokens,
        )
        if any(value is not None for value in tree_args) and not all(
            value is not None for value in tree_args
        ):
            raise ValueError(
                "EAGLE tree configuration requires all of "
                "--speculative-num-steps, --speculative-eagle-topk, and "
                "--speculative-num-draft-tokens"
            )


def random_prompts(
    batch_size: int,
    input_len: int,
    seed: int,
    vocab_size: int,
    excluded_token_ids: set[int],
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


def output_meta(output: object) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {}
    return output.get("meta_info") or {}


def output_token_count(output: object) -> int:
    meta = output_meta(output)
    if "completion_tokens" in meta:
        return int(meta["completion_tokens"])
    if isinstance(output, dict) and "output_ids" in output:
        return len(output["output_ids"])
    raise RuntimeError("SGLang output omitted completion token accounting")


def max_total_num_tokens(llm: object) -> int:
    """Read scheduler capacity across the vendored and current Engine APIs."""
    scheduler_info = getattr(llm, "scheduler_info", None)
    if scheduler_info is not None:
        return int(scheduler_info["max_total_num_tokens"])
    init_result = getattr(llm, "_scheduler_init_result", None)
    scheduler_infos = getattr(init_result, "scheduler_infos", None)
    if scheduler_infos:
        return int(scheduler_infos[0]["max_total_num_tokens"])
    return int(llm.get_server_info()["max_total_num_tokens"])


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    context_reserve = 64
    if args.algorithm == "eagle3":
        num_steps = args.speculative_num_steps or (args.block_size - 1)
        eagle_topk = args.speculative_eagle_topk or 1
        num_draft_tokens = (
            args.speculative_num_draft_tokens or args.block_size
        )
        # SGLang stores EAGLE tree nodes in output-token slots and therefore
        # adds max(top-k * steps, draft tokens) to each request's input length.
        context_reserve = max(
            context_reserve, eagle_topk * num_steps, num_draft_tokens
        )
    kwargs: dict[str, Any] = {
        "model_path": args.model,
        "tp_size": 1,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        # The benchmark supplies token IDs and consumes token IDs. Bypass
        # detokenization so the timed SGLang call matches nano's token-ID-only
        # point runner rather than charging text conversion to SGLang alone.
        "skip_tokenizer_init": True,
        "attention_backend": args.attention_backend,
        "mem_fraction_static": args.mem_fraction_static,
        # Normal speculative workers retain one scheduler slot across the
        # prefill/decode transition. Give the scheduler one slot of headroom
        # while keeping the actual benchmark cohort and CUDA graph at exactly c.
        "max_running_requests": args.batch_size + 1,
        "context_length": args.input_len + args.output_len + context_reserve,
        "chunked_prefill_size": args.chunked_prefill_size,
        "max_prefill_tokens": (
            args.max_prefill_tokens
            if args.max_prefill_tokens is not None
            else args.chunked_prefill_size
        ),
        "random_seed": args.seed,
        "disable_cuda_graph": False,
        "disable_cuda_graph_padding": False,
    }
    if args.algorithm == "eagle3":
        # Current SGLang names decode and prefill graph controls separately.
        kwargs["cuda_graph_max_bs_decode"] = args.batch_size
        kwargs["cuda_graph_bs_decode"] = [args.batch_size]
        kwargs.update(
            {
                "speculative_algorithm": "EAGLE3",
                "speculative_draft_model_path": args.draft_model,
                # B total target-query positions: B-1 draft steps plus bonus.
                "speculative_num_steps": num_steps,
                "speculative_eagle_topk": eagle_topk,
                "speculative_num_draft_tokens": num_draft_tokens,
                "speculative_draft_attention_backend": (
                    args.draft_attention_backend
                ),
            }
        )
        if eagle_topk > 1:
            kwargs["disable_overlap_schedule"] = True
    else:
        kwargs["cuda_graph_max_bs_decode"] = args.batch_size
        kwargs["cuda_graph_bs_decode"] = [args.batch_size]
        kwargs.update(
            {
                "speculative_algorithm": "DFLASH",
                "speculative_draft_model_path": args.draft_model,
                "speculative_dflash_block_size": args.block_size,
                "speculative_draft_attention_backend": (
                    args.draft_attention_backend
                ),
                # DFlash otherwise auto-enables an admission delayer that keeps
                # the last few cohort requests waiting until several running
                # slots become free, invalidating a fixed-cohort benchmark.
                "min_free_slots_delay": 1,
            }
        )
    if args.disable_overlap_schedule:
        kwargs["disable_overlap_schedule"] = True
    return kwargs


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)

    selected_engine_kwargs = engine_kwargs(args)
    llm = sgl.Engine(**selected_engine_kwargs)
    try:
        if args.algorithm == "eagle3":
            forward_width = (
                args.speculative_num_draft_tokens or args.block_size
            )
            max_emitted_tokens = (
                args.speculative_num_steps or (args.block_size - 1)
            ) + 1
        else:
            forward_width = args.block_size
            max_emitted_tokens = args.block_size
        max_total_tokens = max_total_num_tokens(llm)
        required_live_tokens = args.batch_size * (
            args.input_len + args.output_len + forward_width
        )
        if (
            max_total_tokens < required_live_tokens
            and not args.allow_batch_splitting
        ):
            task = ForcedTpfTask(
                method=args.method,
                engine=args.engine,
                block_size=args.block_size,
                batch_size=args.batch_size,
                forced_tpf=args.forced_tpf,
            )
            write_json(
                args.result_path,
                standardize_result({
                    "schema_version": FORCED_TPF_SCHEMA_VERSION,
                    "status": "infeasible",
                    "reason": "resident_capacity",
                    "task": task.as_dict(),
                    "method": args.method,
                    "engine": args.engine,
                    "algorithm": args.algorithm,
                    "block_size": args.block_size,
                    "batch_size": args.batch_size,
                    "requested_tpf": args.forced_tpf,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "generated_tokens": 0,
                    "expected_tokens": args.batch_size * args.output_len,
                    "full_batch_resident": False,
                    "required_live_tokens": required_live_tokens,
                    "max_total_num_tokens": max_total_tokens,
                    "scheduler_overlap_enabled": not bool(
                        selected_engine_kwargs.get("disable_overlap_schedule", False)
                    ),
                }),
            )
            raise RuntimeError(
                "Full cohort cannot remain resident: "
                f"required={required_live_tokens}, capacity={max_total_tokens}"
            )

        sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.warmup_output_len,
            "ignore_eos": True,
        }
        excluded_prompt_token_ids = set(args.excluded_prompt_token_id)
        llm.generate(
            input_ids=random_prompts(
                args.batch_size,
                args.input_len,
                args.seed + 1,
                args.prompt_vocab_size,
                excluded_prompt_token_ids,
            ),
            sampling_params=sampling,
        )

        prompts = random_prompts(
            args.batch_size,
            args.input_len,
            args.seed + 2,
            args.prompt_vocab_size,
            excluded_prompt_token_ids,
        )
        sampling["max_new_tokens"] = args.output_len
        start = time.perf_counter()
        outputs = llm.generate(
            input_ids=prompts,
            sampling_params=sampling,
        )
        elapsed = time.perf_counter() - start

        if not isinstance(outputs, list):
            outputs = [outputs]
        token_counts = [output_token_count(output) for output in outputs]
        metas = [output_meta(output) for output in outputs]
        retractions = [int(meta.get("total_retractions", 0)) for meta in metas]
        generated_tokens = sum(token_counts)
        expected_tokens = args.batch_size * args.output_len

        target_forwards = sum(
            int(meta.get("spec_verify_ct", 0)) for meta in metas
        )
        accepts = generated_tokens
        lookaheads = 0

        full_batch_resident = (
            len(outputs) == args.batch_size
            and not any(retractions)
            and max_total_tokens >= required_live_tokens
        )
        cohort_prefill_tokens = args.batch_size * args.input_len
        prefill_decode_separation_requested = (
            args.chunked_prefill_size > 0
            and (
                args.max_prefill_tokens
                if args.max_prefill_tokens is not None
                else args.chunked_prefill_size
            )
            >= cohort_prefill_tokens
        )
        expected_prefill_batches = (
            cohort_prefill_tokens + args.chunked_prefill_size - 1
        ) // args.chunked_prefill_size
        task = ForcedTpfTask(
            method=args.method,
            engine=args.engine,
            block_size=args.block_size,
            batch_size=args.batch_size,
            forced_tpf=args.forced_tpf,
        )
        acceptance_tpf = accepts / target_forwards if target_forwards else None
        measured_tpf = (
            generated_tokens / target_forwards if target_forwards else None
        )
        effective_forced_tpf = min(float(max_emitted_tokens), args.forced_tpf)
        forced_tpf_tolerance = 0.02
        forced_tpf_matches = (
            measured_tpf is not None
            and abs(measured_tpf - effective_forced_tpf) <= forced_tpf_tolerance
        )
        result = {
            "schema_version": FORCED_TPF_SCHEMA_VERSION,
            "task": task.as_dict(),
            "method": args.method,
            "engine": args.engine,
            "algorithm": args.algorithm,
            "block_size": args.block_size,
            "forward_width": forward_width,
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
            "acceptance_tpf": acceptance_tpf,
            "forced_tpf_tolerance": forced_tpf_tolerance,
            "forced_tpf_matches": forced_tpf_matches,
            "generated_tokens": generated_tokens,
            "expected_tokens": expected_tokens,
            "target_forwards": target_forwards,
            "accepts": accepts,
            "lookaheads": lookaheads,
            "end_to_end_elapsed_s": elapsed,
            "timing_scope": TIMING_SCOPE,
            "aggregate_tps": generated_tokens / elapsed,
            "per_user_tps": (
                generated_tokens / elapsed / args.batch_size
            ),
            "per_request_token_counts": token_counts,
            "per_request_retractions": retractions,
            "full_batch_resident": full_batch_resident,
            "num_preemptions": sum(retractions),
            "batch_splitting_allowed": args.allow_batch_splitting,
            "batch_execution_mode": (
                "resident" if full_batch_resident else "split_or_preempted"
            ),
            "prefill_decode_separation_requested": (
                prefill_decode_separation_requested
            ),
            "prefill_scheduling_policy": "prefill_priority_no_mixed_chunk",
            "expected_prefill_batches": expected_prefill_batches,
            "prefill_chunking_used": expected_prefill_batches > 1,
            # SGLang exposes residency/retraction counters, but not the first
            # decode-batch width in its public offline output.
            "all_requests_entered_decode_together": None,
            "first_decode_batch_size": None,
            "cohort_validation": "resident_capacity_and_no_retractions",
            "required_live_tokens": required_live_tokens,
            "max_total_num_tokens": max_total_tokens,
            "chunked_prefill_size": args.chunked_prefill_size,
            "max_prefill_tokens": (
                args.max_prefill_tokens
                if args.max_prefill_tokens is not None
                else args.chunked_prefill_size
            ),
            "cuda_graph_batch_sizes": [args.batch_size],
            "skip_tokenizer_init": True,
            "attention_backend": args.attention_backend,
            "draft_attention_backend": args.draft_attention_backend,
            "mem_fraction_static": args.mem_fraction_static,
            "max_running_requests": args.batch_size + 1,
            "model": args.model,
            "draft_model": args.draft_model,
            "speculative_num_steps": args.speculative_num_steps,
            "speculative_eagle_topk": args.speculative_eagle_topk,
            "speculative_num_draft_tokens": (
                args.speculative_num_draft_tokens
            ),
            "forced_acceptance_method": os.environ.get(
                "SGLANG_SIMULATE_ACC_METHOD"
            ),
            "forced_acceptance_token_mode": os.environ.get(
                "SGLANG_SIMULATE_ACC_TOKEN_MODE"
            ),
            "torch_compile_enabled": selected_engine_kwargs.get(
                "enable_torch_compile", False
            ),
            "scheduler_overlap_enabled": not bool(
                selected_engine_kwargs.get("disable_overlap_schedule", False)
            ),
            "seed": args.seed,
            "status": (
                "complete"
                if (
                    generated_tokens == expected_tokens
                    and full_batch_resident
                    and target_forwards > 0
                    and forced_tpf_matches
                )
                else "invalid"
            ),
        }
        result = standardize_result(result)
        write_json(args.result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

        if generated_tokens != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens} generated tokens, "
                f"got {generated_tokens}"
            )
        if not full_batch_resident and not args.allow_batch_splitting:
            raise RuntimeError(
                f"Cohort split or retracted: retractions={retractions}"
            )
        if target_forwards <= 0:
            raise RuntimeError("Per-request target-forward counters were absent")
        if not forced_tpf_matches:
            raise RuntimeError(
                "Measured output TPF missed the forced target: "
                f"requested={effective_forced_tpf}, measured={measured_tpf}"
            )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
