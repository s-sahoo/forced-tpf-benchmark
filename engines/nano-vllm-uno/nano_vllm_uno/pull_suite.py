from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nano_vllm_uno.engine.async_engine import AsyncLLMEngine
from nano_vllm_uno.eval.benchmarks import BENCHMARKS as RELEASE_BENCHMARKS
from nano_vllm_uno.sampling_params import SamplingParams


MAX_TOKENS = 32768
MAX_NUM_SEQS = 64
MAX_MODEL_LEN = 262144
MAX_NUM_BATCHED_TOKENS = 262144
GPU_MEMORY_UTILIZATION = 0.9
DIFFUSION_BLOCK_SIZE = 16
MAX_DIFFUSION_BLOCK_SIZE = 64
CUDA_GRAPH_BLOCK_SIZES = [16]
STOP_TOKEN_IDS = [151645, 151643]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    filename: str
    repetitions: int
    instruction: str = ""
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


BENCHMARK_REPETITIONS = {
    "gsm8k": 1,
    "math500": 1,
    "aime24": 1,
    "aime25": 1,
    "aime26": 1,
    "humaneval": 1,
    "mbpp": 1,
    "lcbv6": 1,
    "gpqa": 1,
    "gpqa_diamond": 1,
    "mmlu": 1,
    "mmlu_pro": 1,
    "arc_c": 1,
    "ifeval": 1,
}


def _release_spec(name: str) -> BenchmarkSpec:
    config = RELEASE_BENCHMARKS[name]
    return BenchmarkSpec(
        name=config.name,
        filename=config.data_path.name,
        repetitions=BENCHMARK_REPETITIONS[name],
        instruction=config.instruction,
        chat_template_kwargs=dict(config.chat_template_kwargs),
    )


BENCHMARKS = tuple(_release_spec(name) for name in BENCHMARK_REPETITIONS)


@dataclass
class PreparedSuite:
    jobs: list[dict[str, Any]]
    output_by_benchmark: dict[str, Path]
    total_by_benchmark: dict[str, int]
    completed_by_benchmark: dict[str, int]

    @property
    def missing_count(self) -> int:
        return len(self.jobs)


_MANAGER_JOB_QUEUE: queue.Queue | None = None
_MANAGER_RESULT_QUEUE: queue.Queue | None = None


def _job_queue() -> queue.Queue:
    return _MANAGER_JOB_QUEUE


def _result_queue() -> queue.Queue:
    return _MANAGER_RESULT_QUEUE


class PullQueueManager(BaseManager):
    pass


PullQueueManager.register("job_queue", callable=_job_queue)
PullQueueManager.register("result_queue", callable=_result_queue)


def format_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    instruction: str,
    chat_template_kwargs: dict[str, Any],
) -> tuple[list[int], str, list[dict[str, Any]]]:
    messages = [dict(message) for message in messages]
    system_message = next(
        (message for message in messages if message.get("role") == "system"),
        None,
    )
    if instruction:
        if system_message is None:
            messages.insert(0, {"role": "system", "content": instruction})
        elif instruction not in system_message["content"]:
            existing = system_message["content"].strip()
            system_message["content"] = (
                f"{instruction}\n\n{existing}" if existing else instruction
            )

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    prompt_token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **chat_template_kwargs,
    )
    return list(prompt_token_ids), str(rendered_prompt), messages


def _first_field(record: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in _load_records(path)}


def prepare_suite(
    *,
    specs: tuple[BenchmarkSpec, ...] | list[BenchmarkSpec],
    data_root: Path,
    results_root: Path,
    run_name: str,
    tokenizer: Any,
) -> PreparedSuite:
    jobs = []
    output_by_benchmark = {}
    total_by_benchmark = {}
    completed_by_benchmark = {}

    for spec in specs:
        records = _load_records(data_root / spec.filename)
        output = results_root / run_name / spec.name / "generations.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        completed_ids = _completed_ids(output)
        output_by_benchmark[spec.name] = output
        total_by_benchmark[spec.name] = len(records) * spec.repetitions
        completed_by_benchmark[spec.name] = len(completed_ids)

        for record_index, record in enumerate(records):
            prompt_token_ids, rendered_prompt, chat_input = format_prompt(
                tokenizer,
                record["chat_input"],
                spec.instruction,
                spec.chat_template_kwargs,
            )
            source_row = _first_field(record, ("id", "problem_id", "index", "row"))
            if source_row is None:
                source_row = record_index
            ground_truth = _first_field(record, ("ground_truth", "answer"))

            for sample_index in range(spec.repetitions):
                output_id = f"{source_row}:sample{sample_index}"
                if output_id in completed_ids:
                    continue
                row = {
                    "id": output_id,
                    "source_row": source_row,
                    "sample_index": sample_index,
                    "problem": rendered_prompt,
                    "rendered_prompt": rendered_prompt,
                    "chat_input": chat_input,
                    "ground_truth": ground_truth,
                }
                if record.get("completion_input") is not None:
                    row["completion_input"] = record["completion_input"]
                jobs.append(
                    {
                        "benchmark": spec.name,
                        "prompt_token_ids": prompt_token_ids,
                        "row": row,
                    }
                )

    return PreparedSuite(
        jobs=jobs,
        output_by_benchmark=output_by_benchmark,
        total_by_benchmark=total_by_benchmark,
        completed_by_benchmark=completed_by_benchmark,
    )


def write_results(result_queue: Any, suite: PreparedSuite) -> None:
    handles = {
        benchmark: path.open("a", encoding="utf-8")
        for benchmark, path in suite.output_by_benchmark.items()
    }
    bars = {
        benchmark: tqdm(
            total=suite.total_by_benchmark[benchmark],
            initial=suite.completed_by_benchmark[benchmark],
            desc=benchmark,
            unit="gen",
            position=position,
            leave=True,
        )
        for position, benchmark in enumerate(suite.output_by_benchmark)
    }
    try:
        for _ in range(suite.missing_count):
            result = result_queue.get()
            benchmark = result["benchmark"]
            handles[benchmark].write(
                json.dumps(result["row"], ensure_ascii=False) + "\n"
            )
            handles[benchmark].flush()
            bars[benchmark].update(1)
    finally:
        for handle in handles.values():
            handle.close()
        for bar in bars.values():
            bar.close()


def run_generation_worker(
    *,
    job_queue: Any,
    result_queue: Any,
    model: str,
    gated_lora_path: str | None,
) -> None:
    from nano_vllm_uno.engine.llm_engine import LLMEngine

    engine = LLMEngine(
        model,
        tokenizer_path=model,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        gated_lora_path=gated_lora_path,
        max_diffusion_block_size=MAX_DIFFUSION_BLOCK_SIZE,
        cuda_graph_block_sizes=CUDA_GRAPH_BLOCK_SIZES,
    )
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        stop_token_ids=STOP_TOKEN_IDS,
        noise_mode="random_uniform",
        diffusion_block_size=DIFFUSION_BLOCK_SIZE,
    )
    async_engine = AsyncLLMEngine(engine, sampling_params)
    async_engine.start()
    pending: dict[concurrent.futures.Future, dict[str, Any]] = {}
    no_more_jobs = False

    def submit(job: dict[str, Any]) -> None:
        future = async_engine.submit(job["prompt_token_ids"])
        pending[future] = job

    try:
        while pending or not no_more_jobs:
            while not no_more_jobs and len(pending) < MAX_NUM_SEQS:
                try:
                    job = job_queue.get(block=not pending)
                except queue.Empty:
                    break
                if job is None:
                    no_more_jobs = True
                    break
                submit(job)

            if not pending:
                continue
            finished, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in finished:
                job = pending.pop(future)
                result = future.result()
                stats = dict(result.stats)
                result_queue.put(
                    {
                        "benchmark": job["benchmark"],
                        "row": {
                            **job["row"],
                            "generation": result.text,
                            "stats": stats,
                        },
                    }
                )
    finally:
        async_engine.shutdown()


def _connect_manager(host: str, port: int, auth_key: bytes) -> PullQueueManager:
    manager = PullQueueManager(address=(host, port), authkey=auth_key)
    while True:
        try:
            manager.connect()
            return manager
        except ConnectionRefusedError:
            time.sleep(1)


def run_coordinator(
    *,
    worker_count: int,
    model: str,
    gated_lora_path: str | None,
    data_root: Path,
    results_root: Path,
    run_name: str,
    port: int,
    auth_key: bytes,
) -> None:
    job_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    global _MANAGER_JOB_QUEUE, _MANAGER_RESULT_QUEUE
    _MANAGER_JOB_QUEUE = job_queue
    _MANAGER_RESULT_QUEUE = result_queue

    manager = PullQueueManager(address=("", port), authkey=auth_key)
    server = manager.get_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=True,
        trust_remote_code=True,
    )
    suite = prepare_suite(
        specs=BENCHMARKS,
        data_root=data_root,
        results_root=results_root,
        run_name=run_name,
        tokenizer=tokenizer,
    )
    for benchmark in suite.output_by_benchmark:
        print(
            f"{benchmark}: {suite.completed_by_benchmark[benchmark]}/"
            f"{suite.total_by_benchmark[benchmark]} complete",
            flush=True,
        )
    print(f"Queued {suite.missing_count} generations", flush=True)

    for job in suite.jobs:
        job_queue.put(job)
    for _ in range(worker_count):
        job_queue.put(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(write_results, result_queue, suite)
        run_generation_worker(
            job_queue=job_queue,
            result_queue=result_queue,
            model=model,
            gated_lora_path=gated_lora_path,
        )
        writer.result()


def main() -> None:
    model = os.environ["NANO_MODEL"]
    gated_lora_path = os.environ.get("NANO_GATED_LORA_PATH")
    data_root = Path(os.environ["NANO_DATA_ROOT"])
    results_root = Path(os.environ["NANO_RESULTS_ROOT"])
    run_name = os.environ["NANO_RUN_NAME"]
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", str(rank)))
    worker_count = int(os.environ.get("SLURM_NTASKS", "1"))
    port = int(os.environ.get("NANO_PULL_COORDINATOR_PORT", "19191"))
    auth_key = os.environ.get("NANO_PULL_AUTH_KEY", "nano-vllm-uno").encode()
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    torch_port_base = int(
        os.environ.get(
            "NANO_TORCH_DISTRIBUTED_PORT_BASE",
            40000 + job_id % 10000,
        )
    )
    os.environ["TORCH_DISTRIBUTED_PORT"] = str(torch_port_base + local_rank)

    if rank == 0:
        run_coordinator(
            worker_count=worker_count,
            model=model,
            gated_lora_path=gated_lora_path,
            data_root=data_root,
            results_root=results_root,
            run_name=run_name,
            port=port,
            auth_key=auth_key,
        )
        return

    manager = _connect_manager(
        os.environ["NANO_PULL_COORDINATOR_HOST"],
        port,
        auth_key,
    )
    run_generation_worker(
        job_queue=manager.job_queue(),
        result_queue=manager.result_queue(),
        model=model,
        gated_lora_path=gated_lora_path,
    )


if __name__ == "__main__":
    main()
