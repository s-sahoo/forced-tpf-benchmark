#!/usr/bin/env python3
"""Measure one uncompiled SGLang AR GPU point."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from timing import (
    AR_SCHEMA_VERSION,
    TIMING_SCOPE,
    read_json,
    standardize_result,
    summarize_sglang_ar_trace,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
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
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    parser.add_argument(
        "--detailed-tracing",
        action="store_true",
        help="Opt in to per-step timing events.",
    )
    return parser.parse_args()


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
    raise RuntimeError("SGLang output omitted completion-token accounting")


def max_total_num_tokens(llm: object) -> int:
    init_result = getattr(llm, "_scheduler_init_result", None)
    scheduler_infos = getattr(init_result, "scheduler_infos", None)
    if scheduler_infos:
        return int(scheduler_infos[0]["max_total_num_tokens"])
    return int(llm.get_server_info()["max_total_num_tokens"])


def wait_for_trace(path: Path, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return read_json(path)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(0.01)
    raise RuntimeError(
        f"scheduler did not publish a complete timing trace at {path}"
    ) from last_error


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    prefill_budget = max(8192, args.batch_size * args.input_len + 1)
    return {
        "model_path": args.model,
        "tp_size": 1,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        # This is SGLang's current fastest/default MHA backend on Hopper.
        "attention_backend": "fa3",
        "sampling_backend": "flashinfer",
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.batch_size + 1,
        "context_length": args.input_len + args.output_len + 64,
        "chunked_prefill_size": prefill_budget,
        "max_prefill_tokens": prefill_budget,
        "random_seed": args.seed,
        # Use an exact-shape full decode graph. The diagnostic may toggle
        # overlap scheduling; torch.compile is always disabled.
        "cuda_graph_backend_decode": "full",
        "cuda_graph_max_bs_decode": args.batch_size,
        "cuda_graph_bs_decode": [args.batch_size],
        "disable_cuda_graph_padding": False,
        "disable_overlap_schedule": args.disable_overlap_schedule,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    trace_path = args.trace_path or args.result_path.with_name(
        args.result_path.stem + ".trace.json"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        trace_path.unlink()
    except FileNotFoundError:
        pass
    if not args.detailed_tracing:
        os.environ.pop("SGLANG_DECODE_TIMING_PATH", None)
    else:
        os.environ["SGLANG_DECODE_TIMING_PATH"] = str(trace_path.resolve())

    # Import after setting the path so every scheduler subprocess inherits the
    # opt-in trace configuration.
    import sglang as sgl

    kwargs = engine_kwargs(args)
    llm = sgl.Engine(**kwargs)
    try:
        max_tokens = max_total_num_tokens(llm)
        required_tokens = args.batch_size * (
            args.input_len + args.output_len + 1
        )
        if max_tokens < required_tokens:
            raise RuntimeError(
                "full cohort cannot remain resident: "
                f"required={required_tokens}, capacity={max_tokens}"
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
                min(args.input_len, 256),
                args.seed + 1,
                args.prompt_vocab_size,
                excluded_prompt_token_ids,
            ),
            sampling_params=sampling,
        )
        if args.detailed_tracing:
            wait_for_trace(trace_path)
            trace_path.unlink()

        prompts = random_prompts(
            args.batch_size,
            args.input_len,
            args.seed + 2,
            args.prompt_vocab_size,
            excluded_prompt_token_ids,
        )
        sampling["max_new_tokens"] = args.output_len
        request_start_ns = time.perf_counter_ns()
        outputs = llm.generate(
            input_ids=prompts,
            sampling_params=sampling,
        )
        request_elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
        if not isinstance(outputs, list):
            outputs = [outputs]
        trace = (
            {
                "enabled": False,
                "overlap_schedule": not args.disable_overlap_schedule,
            }
            if not args.detailed_tracing
            else wait_for_trace(trace_path)
        )
        trace.update(
            {
                "schema_version": AR_SCHEMA_VERSION,
                "engine": "sglang_ar",
                "batch_size": args.batch_size,
                "forwards_per_cycle": 1,
                "tokens_per_cycle": 1,
            }
        )
        timing = (
            {
                "enabled": False,
                "scope": "end_to_end_only",
            }
            if not args.detailed_tracing
            else summarize_sglang_ar_trace(
                trace,
                batch_size=args.batch_size,
            )
        )
        token_counts = [output_token_count(output) for output in outputs]
        generated_tokens = sum(token_counts)
        expected_tokens = args.batch_size * args.output_len
        retractions = [
            int(output_meta(output).get("total_retractions", 0))
            for output in outputs
        ]
        full_batch_resident = (
            len(outputs) == args.batch_size
            and generated_tokens == expected_tokens
            and not any(retractions)
            and max_tokens >= required_tokens
        )
        result = {
            "schema_version": AR_SCHEMA_VERSION,
            "engine": "sglang_ar",
            "algorithm": "autoregressive",
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "requested_tpf": None,
            "measured_tpf": 1.0,
            "generated_tokens": generated_tokens,
            "expected_tokens": expected_tokens,
            "per_request_token_counts": token_counts,
            "per_request_retractions": retractions,
            "end_to_end_elapsed_s": request_elapsed_s,
            "aggregate_tps": generated_tokens / request_elapsed_s,
            "per_user_tps": generated_tokens / request_elapsed_s / args.batch_size,
            "timing_scope": TIMING_SCOPE,
            "raw_forwards_per_cycle": 1,
            "raw_tokens_per_cycle": 1,
            "reported_normalization_divisor": 1,
            "reported_unit": "one_forward_and_one_generated_token",
            "required_live_tokens": required_tokens,
            "max_total_num_tokens": max_tokens,
            "full_batch_resident": full_batch_resident,
            "num_preemptions": sum(retractions),
            "optimizations": {
                "attention_backend": "fa3",
                "sampling_backend": "flashinfer",
                "decode_cuda_graph_backend": "full",
                "decode_cuda_graph_batch_sizes": [args.batch_size],
                "overlap_schedule": bool(trace["overlap_schedule"]),
                "detailed_decode_tracing": args.detailed_tracing,
            },
            "timing": timing,
            "trace_path": str(trace_path),
            "model": args.model,
            "seed": args.seed,
            "status": "complete" if full_batch_resident else "invalid",
        }
        write_json(trace_path, trace)
        result = standardize_result(result)
        write_json(args.result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

        if len(outputs) != args.batch_size:
            raise RuntimeError(
                f"expected {args.batch_size} outputs, got {len(outputs)}"
            )
        if generated_tokens != expected_tokens:
            raise RuntimeError(
                f"expected {expected_tokens} generated tokens, "
                f"got {generated_tokens}"
            )
        if any(retractions):
            raise RuntimeError(f"requests were retracted: {retractions}")
        if bool(trace["overlap_schedule"]) == args.disable_overlap_schedule:
            raise RuntimeError(
                "SGLang overlap-schedule state did not match the request"
            )
        if args.detailed_tracing and int(trace["missing_graph_events"]):
            raise RuntimeError(
                "SGLang timing missed graph events: "
                f"{trace['missing_graph_events']}"
            )
        if args.detailed_tracing and not timing["all_cuda_graph_hits"]:
            raise RuntimeError("SGLang timed decode had CUDA graph misses")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
