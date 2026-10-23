#!/usr/bin/env python3
"""Measure one forced-TPF diffusion baseline GPU point."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import threading
import time
from pathlib import Path
from typing import Any

from timing import standardize_result, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("nemotron_sglang", "diffgemma_sglang", "diffgemma_vllm"),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dllm-config", type=Path)
    parser.add_argument("--forced-tpf", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt-vocab-size", type=int, required=True)
    parser.add_argument(
        "--excluded-prompt-token-id", action="append", default=[], type=int
    )
    parser.add_argument("--warmup-output-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--vllm-cudagraph-mode",
        choices=("FULL_AND_PIECEWISE", "FULL_DECODE_ONLY"),
        default="FULL_AND_PIECEWISE",
        help="vLLM CUDA-graph policy; ignored by SGLang engines.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=8192,
        help="Maximum SGLang activation batch during cohort prefill.",
    )
    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=None,
        help=(
            "Aggregate prefill admission budget. Defaults to the complete "
            "cohort plus one token of SGLang admission slack."
        ),
    )
    parser.add_argument(
        "--disable-model-eos",
        action="store_true",
        help=(
            "Diagnostic only: override the model EOS token to null so the "
            "dLLM algorithm cannot truncate before the scheduler applies "
            "ignore_eos."
        ),
    )
    return parser.parse_args()


def prompt_ids(
    length: int, seed: int, vocab_size: int, excluded_token_ids: set[int]
) -> list[int]:
    rng = random.Random(seed)
    excluded_token_ids = {
        token_id for token_id in excluded_token_ids if 0 <= token_id < vocab_size
    }
    if len(excluded_token_ids) == vocab_size:
        raise ValueError("prompt token filter excludes the complete vocabulary")
    result = []
    while len(result) < length:
        token_id = rng.randrange(vocab_size)
        if token_id not in excluded_token_ids:
            result.append(token_id)
    return result


def sglang_meta(output: object) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {}
    return output.get("meta_info") or {}


def sglang_token_count(output: object) -> int:
    meta = sglang_meta(output)
    if "completion_tokens" in meta:
        return int(meta["completion_tokens"])
    if isinstance(output, dict) and "output_ids" in output:
        return len(output["output_ids"])
    raise RuntimeError("SGLang output omitted completion token accounting")


def random_prompt_batch(args: argparse.Namespace, seed: int) -> list[list[int]]:
    return [
        prompt_ids(
            args.input_len,
            seed + request_index,
            args.prompt_vocab_size,
            set(args.excluded_prompt_token_id),
        )
        for request_index in range(args.batch_size)
    ]


def sglang_retractions(meta: dict[str, Any]) -> int:
    return int(meta.get("num_retractions", meta.get("total_retractions", 0)))


def vllm_metrics(llm: object) -> list[object]:
    """Read vLLM's in-memory Prometheus snapshot."""
    get_metrics = getattr(llm, "get_metrics", None)
    if callable(get_metrics):
        return get_metrics()
    llm_engine = getattr(llm, "llm_engine", None)
    get_metrics = getattr(llm_engine, "get_metrics", None)
    if not callable(get_metrics):
        raise RuntimeError("vLLM metrics are unavailable")
    return get_metrics()


def vllm_preemption_count(llm: object) -> int:
    """Read vLLM's cumulative preemption counter without modifying its engine."""
    total = 0
    found = False
    for metric in vllm_metrics(llm):
        if metric.name == "vllm:num_preemptions" and hasattr(metric, "value"):
            total += int(metric.value)
            found = True
    if not found:
        raise RuntimeError("vLLM did not expose its preemption counter")
    return total


class VllmCohortMonitor:
    """Poll scheduler gauges while synchronous offline generation is running."""

    def __init__(
        self,
        llm: object,
        batch_size: int,
        poll_interval_s: float = 0.1,
    ) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.poll_interval_s = poll_interval_s
        self.max_running = 0
        self.max_waiting = 0
        self.full_cohort_observed = False
        self.samples = 0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._poll,
            name="vllm-cohort-monitor",
            daemon=True,
        )

    def _sample(self) -> None:
        values: dict[str, int] = {}
        names = {
            "vllm:num_requests_running": "running",
            "vllm:num_requests_waiting": "waiting",
        }
        for metric in vllm_metrics(self.llm):
            key = names.get(getattr(metric, "name", ""))
            if key is not None and hasattr(metric, "value"):
                values[key] = values.get(key, 0) + int(metric.value)
        if set(values) != {"running", "waiting"}:
            raise RuntimeError("vLLM scheduler residency gauges are unavailable")
        running = values["running"]
        waiting = values["waiting"]
        self.max_running = max(self.max_running, running)
        self.max_waiting = max(self.max_waiting, waiting)
        self.full_cohort_observed |= (
            running >= self.batch_size and waiting == 0
        )
        self.samples += 1

    def _poll(self) -> None:
        try:
            while not self._stop.wait(self.poll_interval_s):
                self._sample()
        except BaseException as error:
            self._error = error
            self._stop.set()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("vLLM cohort monitor failed") from self._error


def sglang_max_total_num_tokens(llm: object) -> int:
    """Read scheduler capacity across the vendored SGLang Engine APIs."""
    scheduler_info = getattr(llm, "scheduler_info", None)
    if scheduler_info is not None:
        return int(scheduler_info["max_total_num_tokens"])
    init_result = getattr(llm, "_scheduler_init_result", None)
    scheduler_infos = getattr(init_result, "scheduler_infos", None)
    if scheduler_infos:
        return int(scheduler_infos[0]["max_total_num_tokens"])
    return int(llm.get_server_info()["max_total_num_tokens"])


def run_sglang(args: argparse.Namespace) -> dict[str, Any]:
    import sglang as sgl

    if args.dllm_config is None:
        raise ValueError("SGLang engines require --dllm-config")

    is_nemotron = args.engine == "nemotron_sglang"
    context_length = args.input_len + args.output_len + 512
    prefill_budget = args.max_prefill_tokens or (
        args.batch_size * args.input_len + 1
    )
    kwargs: dict[str, Any] = {
        "model_path": args.model,
        "tp_size": 1,
        "dp_size": 1,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "skip_tokenizer_init": True,
        "max_running_requests": args.batch_size,
        "prefill_max_requests": args.batch_size,
        "context_length": context_length,
        "mem_fraction_static": args.gpu_memory_utilization,
        "random_seed": args.seed,
        "dllm_algorithm": "LinearSpec" if is_nemotron else "Gemma4Renoise",
        "dllm_algorithm_config": str(args.dllm_config),
    }
    if args.disable_model_eos:
        kwargs["json_model_override_args"] = json.dumps({"eos_token_id": None})
    if is_nemotron:
        # LinearSpec's optimized decode graph contains the draft-only gated
        # LoRA delta. Capture the exact measured cohort width; DP and TP stay one.
        kwargs.update(
            {
                "attention_backend": "flashinfer",
                "chunked_prefill_size": min(
                    args.prefill_chunk_size, prefill_budget
                ),
                "max_prefill_tokens": prefill_budget,
                "cuda_graph_max_bs": args.batch_size,
                "cuda_graph_bs": [args.batch_size],
                # Forced-TPF state is per request. Padding a shrinking tail
                # batch to the original cohort width can duplicate request
                # slots and corrupt per-sequence forward accounting.
                "disable_cuda_graph_padding": True,
            }
        )
    else:
        # The DiffusionGemma fork enforces eager Triton attention and
        # unchunked prefill for its 512-wide bidirectional attention head.
        kwargs.update(
            {
                "attention_backend": "triton",
                "chunked_prefill_size": -1,
            }
        )

    llm = sgl.Engine(**kwargs)
    try:
        max_total_tokens = sglang_max_total_num_tokens(llm)
        forward_width = 32 if is_nemotron else 512
        required_live_tokens = args.batch_size * (
            args.input_len + args.output_len + forward_width
        )
        if required_live_tokens >= max_total_tokens:
            return {
                "engine": args.engine,
                "engine_family": "sglang",
                "model": args.model,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "gpu_count": 1,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "batch_size": args.batch_size,
                "requested_tpf": args.forced_tpf,
                "generated_tokens": 0,
                "expected_tokens": args.batch_size * args.output_len,
                "decode_forward_passes": 0,
                "measured_tpf": None,
                "full_batch_resident": False,
                "num_preemptions": 0,
                "status": "infeasible",
                "reason": "resident_capacity",
                "required_live_tokens": required_live_tokens,
                "max_total_num_tokens": max_total_tokens,
                "dllm_config": str(args.dllm_config),
            }
        sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.warmup_output_len,
            "ignore_eos": True,
        }
        llm.generate(
            input_ids=random_prompt_batch(args, args.seed + 1),
            sampling_params=sampling,
        )

        sampling["max_new_tokens"] = args.output_len
        started = time.perf_counter()
        output = llm.generate(
            input_ids=random_prompt_batch(args, args.seed + 2),
            sampling_params=sampling,
        )
        elapsed = time.perf_counter() - started
        outputs = output if isinstance(output, list) else [output]
        metas = [sglang_meta(item) for item in outputs]
        token_counts = [sglang_token_count(item) for item in outputs]
        generated = sum(token_counts)
        forwards = sum(
            int(meta.get("dllm_decode_forward_passes", 0)) for meta in metas
        )
        measured_tpf = generated / forwards if forwards else None
        retractions = [sglang_retractions(meta) for meta in metas]
        full_batch_resident = (
            len(outputs) == args.batch_size and not any(retractions)
        )
        cohort_prefill_tokens = args.batch_size * args.input_len
        chunk_size = min(args.prefill_chunk_size, prefill_budget)
        return {
            "engine": args.engine,
            "engine_family": "sglang",
            "model": args.model,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "gpu_count": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "requested_tpf": args.forced_tpf,
            "generated_tokens": generated,
            "expected_tokens": args.batch_size * args.output_len,
            "per_request_token_counts": token_counts,
            "decode_forward_passes": forwards,
            "measured_tpf": measured_tpf,
            "end_to_end_elapsed_s": elapsed,
            "aggregate_tps": generated / elapsed,
            "per_user_tps": generated / elapsed / args.batch_size,
            "timing_scope": "end_to_end_including_prefill",
            "warmup_output_len": args.warmup_output_len,
            "full_batch_resident": full_batch_resident,
            "per_request_retractions": retractions,
            "num_preemptions": sum(retractions),
            "cohort_validation": "output_count_and_no_retractions",
            "prefill_decode_separation_requested": (
                prefill_budget > cohort_prefill_tokens
            ),
            "prefill_scheduling_policy": "prefill_priority_no_mixed_chunk",
            "prefill_chunk_size": chunk_size,
            "max_prefill_tokens": prefill_budget,
            "required_live_tokens": required_live_tokens,
            "max_total_num_tokens": max_total_tokens,
            "expected_prefill_batches": (
                cohort_prefill_tokens + chunk_size - 1
            )
            // chunk_size,
            "dllm_config": str(args.dllm_config),
            "model_eos_disabled": args.disable_model_eos,
            "raw_meta_info": metas,
        }
    finally:
        llm.shutdown()


def run_vllm(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    context_length = args.input_len + args.output_len + 512
    requested_prefill_budget = args.max_prefill_tokens or max(
        context_length, args.batch_size * args.input_len
    )
    prefill_budget = requested_prefill_budget
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=context_length,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=prefill_budget,
        attention_backend="TRITON_ATTN",
        generation_config="vllm",
        # These benchmarks contain token IDs only. Avoid loading/profiling the
        # unused vision tower, whose encoder-cache budget otherwise scales with
        # max_num_batched_tokens and can poison large text-only graph captures.
        language_model_only=True,
        mm_processor_cache_gb=0,
        # Keep graph policy explicit in every result. FULL_DECODE_ONLY is the
        # parameter-only fallback when large piecewise prefill capture fails.
        compilation_config={"cudagraph_mode": args.vllm_cudagraph_mode},
        hf_overrides={
            "diffusion_sampler": "entropy_bound",
            "diffusion_entropy_bound": 0.1,
        },
        diffusion_config={
            "canvas_length": 256,
            "forced_tpf": args.forced_tpf,
        },
        seed=args.seed,
        disable_log_stats=False,
    )

    cache_config = llm.llm_engine.vllm_config.cache_config
    kv_cache_tokens = getattr(cache_config, "kv_cache_size_tokens", None)
    kv_cache_max_concurrency = getattr(
        cache_config, "kv_cache_max_concurrency", None
    )
    if kv_cache_max_concurrency is None:
        raise RuntimeError("vLLM did not expose resolved KV-cache capacity")
    if float(kv_cache_max_concurrency) < args.batch_size:
        return {
            "engine": args.engine,
            "engine_family": "vllm",
            "model": args.model,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "gpu_count": 1,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "batch_size": args.batch_size,
            "requested_tpf": args.forced_tpf,
            "generated_tokens": 0,
            "expected_tokens": args.batch_size * args.output_len,
            "decode_forward_passes": 0,
            "full_batch_resident": False,
            "reason": "resident_capacity",
            "required_concurrency": args.batch_size,
            "kv_cache_size_tokens": kv_cache_tokens,
            "kv_cache_max_concurrency": float(kv_cache_max_concurrency),
            "requested_max_num_batched_tokens": requested_prefill_budget,
            "max_num_batched_tokens": prefill_budget,
            "max_num_seqs": args.batch_size,
            "cudagraph_mode": args.vllm_cudagraph_mode,
        }

    warmup = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.warmup_output_len,
        ignore_eos=True,
        detokenize=False,
    )
    measured = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.output_len,
        ignore_eos=True,
        detokenize=False,
    )
    llm.generate(
        [
            TokensPrompt(prompt_token_ids=prompt)
            for prompt in random_prompt_batch(args, args.seed + 1)
        ],
        warmup,
        use_tqdm=False,
    )
    preemptions_before = vllm_preemption_count(llm)

    cohort_monitor = VllmCohortMonitor(llm, args.batch_size)
    cohort_monitor.start()
    started = time.perf_counter()
    try:
        outputs = llm.generate(
            [
                TokensPrompt(prompt_token_ids=prompt)
                for prompt in random_prompt_batch(args, args.seed + 2)
            ],
            measured,
            use_tqdm=False,
        )
    finally:
        elapsed = time.perf_counter() - started
        cohort_monitor.stop()
    preemptions = vllm_preemption_count(llm) - preemptions_before
    if len(outputs) != args.batch_size or any(
        len(request_output.outputs) != 1 for request_output in outputs
    ):
        raise RuntimeError(
            f"expected {args.batch_size} single-completion vLLM outputs"
        )
    completions = [request_output.outputs[0] for request_output in outputs]
    token_counts = [len(completion.token_ids) for completion in completions]
    generated = sum(token_counts)
    forwards = sum(
        int(completion.dllm_decode_forward_passes or 0)
        for completion in completions
    )
    measured_tpf = generated / forwards if forwards else None
    # At short output lengths, per-sequence forward counts are integral, so
    # the closest observable aggregate TPF can be more than 0.01 away from
    # the requested value.  One per-request forward quantum is the strictest
    # meaningful tolerance for validating forced TPF at this cohort size.
    tpf_tolerance = (
        max(0.01, args.forced_tpf * args.batch_size / forwards)
        if forwards
        else 0.01
    )
    return {
        "engine": args.engine,
        "engine_family": "vllm",
        "model": args.model,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "gpu_count": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "requested_tpf": args.forced_tpf,
        "generated_tokens": generated,
        "expected_tokens": args.batch_size * args.output_len,
        "per_request_token_counts": token_counts,
        "decode_forward_passes": forwards,
        "measured_tpf": measured_tpf,
        "forced_tpf_error": (
            measured_tpf - args.forced_tpf
            if measured_tpf is not None
            else None
        ),
        "forced_tpf_tolerance": tpf_tolerance,
        "end_to_end_elapsed_s": elapsed,
        "aggregate_tps": generated / elapsed,
        "per_user_tps": generated / elapsed / args.batch_size,
        "timing_scope": "end_to_end_including_prefill",
        "warmup_output_len": args.warmup_output_len,
        "full_batch_resident": (
            preemptions == 0
            and cohort_monitor.full_cohort_observed
            and cohort_monitor.max_waiting == 0
        ),
        "num_preemptions": preemptions,
        "cohort_validation": (
            "vllm_scheduler_gauges_zero_waiting_and_preemption_counter"
        ),
        "max_running_requests_observed": cohort_monitor.max_running,
        "max_waiting_requests_observed": cohort_monitor.max_waiting,
        "all_requests_running_together_observed": (
            cohort_monitor.full_cohort_observed
        ),
        "scheduler_metric_samples": cohort_monitor.samples,
        "prefill_decode_separation_requested": True,
        "requested_max_num_batched_tokens": requested_prefill_budget,
        "max_num_batched_tokens": prefill_budget,
        "max_num_seqs": args.batch_size,
        "language_model_only": True,
        "cudagraph_mode": args.vllm_cudagraph_mode,
        "diffusion_config": {
            "canvas_length": 256,
            "forced_tpf": args.forced_tpf,
        },
    }


def main() -> None:
    args = parse_args()
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("input and output lengths must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output_len % 256 != 0 and "diffgemma" in args.engine:
        raise ValueError("DiffusionGemma output length must be divisible by 256")

    result = run_vllm(args) if args.engine == "diffgemma_vllm" else run_sglang(args)
    expected_tokens = args.batch_size * args.output_len
    generated_all = result["generated_tokens"] == expected_tokens
    resident = bool(result.get("full_batch_resident"))
    has_forwards = bool(result["decode_forward_passes"])
    import torch

    result.update(
        {
            "algorithm": (
                "linearspec"
                if args.engine == "nemotron_sglang"
                else "diffusion_gemma"
            ),
            "model": args.model,
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "requested_tpf": args.forced_tpf,
            "expected_tokens": expected_tokens,
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "schema_version": 1,
            "status": (
                "complete"
                if generated_all and resident and has_forwards
                else "infeasible"
            ),
        }
    )
    result = standardize_result(result)
    write_json(args.result_path, result)
    if result["generated_tokens"] != expected_tokens:
        raise RuntimeError(
            f"generated {result['generated_tokens']} tokens, expected "
            f"{expected_tokens}"
        )
    if not result.get("full_batch_resident"):
        raise RuntimeError(
            "Timed cohort did not remain jointly resident: "
            f"num_preemptions={result.get('num_preemptions')}, "
            "max_running_requests_observed="
            f"{result.get('max_running_requests_observed')}, "
            "max_waiting_requests_observed="
            f"{result.get('max_waiting_requests_observed')}"
        )
    if not result["decode_forward_passes"]:
        raise RuntimeError("engine did not transport per-sequence forward counts")


if __name__ == "__main__":
    main()
