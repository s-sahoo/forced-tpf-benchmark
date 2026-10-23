#!/usr/bin/env python3
"""Run AIME25-style JSONL prompts with nano-vllm-uno data parallelism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from transformers import AutoTokenizer

from nano_vllm_uno import LLM, SamplingParams


PROBLEM_FIELDS = ("problem", "question", "prompt", "completion_input")
ID_FIELDS = ("id", "problem_id", "index", "row")
ANSWER_FIELDS = ("ground_truth", "answer")


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def first_field(record: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return None


def format_prompt(tokenizer: Any, problem: str, instruction: str) -> str:
    instruction = instruction.strip()
    content = (
        f"{instruction}\n\n{problem.strip()}"
        if instruction
        else problem.strip()
    )
    messages = [{"role": "user", "content": content}]
    if tokenizer.chat_template is None:
        return content
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("aime25-generations.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--num-samples",
        "--repetitions",
        dest="num_samples",
        type=int,
        default=1,
        help="Generations per problem; --repetitions is a compatibility alias.",
    )
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--stop-token-ids", type=parse_int_list)
    parser.add_argument("--diffusion-block-size", type=int, default=1)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument(
        "--noise-mode",
        choices=("random_uniform", "deterministic_uniform", "mask"),
        default="random_uniform",
    )
    parser.add_argument("--noise-salt", type=int)
    parser.add_argument("--gated-lora-path")
    parser.add_argument(
        "--cuda-graph-block-sizes",
        type=parse_int_list,
        default=None,
        help="Comma-separated forward lengths; defaults to the active block size.",
    )
    parser.add_argument(
        "--cuda-graph-batch-sizes",
        type=parse_int_list,
        default=None,
        help="Optional comma-separated CUDA graph batch sizes.",
    )
    parser.add_argument(
        "--instruction",
        default="Please reason step by step and put your final answer in \\boxed{}.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.diffusion_block_size > 1 and args.mask_token_id is None:
        raise ValueError("--mask-token-id is required when diffusion block size exceeds 1")

    records = load_records(args.data, args.limit)
    tokenizer_path = args.tokenizer_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )

    prompts: list[str] = []
    problems: list[str] = []
    record_ids: list[Any] = []
    ground_truths: list[Any] = []
    for index, record in enumerate(records):
        problem = first_field(record, PROBLEM_FIELDS)
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(
                f"Record {index} needs a nonempty field from {PROBLEM_FIELDS}"
            )
        problems.append(problem)
        record_id = first_field(record, ID_FIELDS)
        if record_id is None:
            record_id = index
        record_ids.append(record_id)
        ground_truths.append(first_field(record, ANSWER_FIELDS))
        prompts.append(format_prompt(tokenizer, problem, args.instruction))

    # Question-major ordering: q0 sample0..N-1, q1 sample0..N-1, ...
    prompts = [
        prompt
        for prompt in prompts
        for _ in range(args.num_samples)
    ]

    graph_block_sizes = args.cuda_graph_block_sizes or [args.diffusion_block_size]
    llm = LLM(
        model=args.model,
        tokenizer_path=tokenizer_path,
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_diffusion_block_size=args.diffusion_block_size,
        cuda_graph_block_sizes=graph_block_sizes,
        cuda_graph_batch_sizes=args.cuda_graph_batch_sizes,
        gated_lora_path=args.gated_lora_path,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        stop_token_ids=args.stop_token_ids,
        mask_token_id=args.mask_token_id,
        noise_mode=args.noise_mode,
        noise_salt=args.noise_salt,
        diffusion_block_size=args.diffusion_block_size,
    )

    start = perf_counter()
    try:
        outputs = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=not args.no_progress,
        )
        elapsed = perf_counter() - start
        stats = getattr(llm, "last_generate_stats", {})
    finally:
        llm.exit()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_output_tokens = 0
    with args.output.open("w", encoding="utf-8") as handle:
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Expected {len(prompts)} outputs, got {len(outputs)}"
            )
        for flat_index, output in enumerate(outputs):
            record_index, sample_index = divmod(flat_index, args.num_samples)
            record_id = record_ids[record_index]
            problem = problems[record_index]
            ground_truth = ground_truths[record_index]
            token_ids = output["token_ids"]
            total_output_tokens += len(token_ids)
            row = {
                "id": f"{record_id}:sample{sample_index}",
                "source_row": record_id,
                "sample_index": sample_index,
                "problem": problem,
                "ground_truth": ground_truth,
                "generation": output["text"],
                "output_token_count": len(token_ids),
                "stats": output["stats"],
            }
            if args.save_token_ids:
                row["token_ids"] = token_ids
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "num_problems": len(records),
        "num_samples_per_problem": args.num_samples,
        "num_generations": len(outputs),
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": total_output_tokens / elapsed,
        "decoder_stats": stats,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
