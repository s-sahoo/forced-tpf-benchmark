import atexit
import logging
import os
from time import perf_counter, perf_counter_ns
from typing import Callable
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nano_vllm_uno.config import Config
from nano_vllm_uno.sampling_params import SamplingParams
from nano_vllm_uno.engine.sequence import Sequence
from nano_vllm_uno.engine.scheduler import Scheduler
from nano_vllm_uno.engine.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self, model, tokenizer_path=None, **kwargs):
        config = Config(model, **kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size

        tokenizer_path = tokenizer_path or model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
        )
        config.eos = self.tokenizer.eos_token_id
        config.pad = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        self.ps = []
        self.events = []
        self._exited = False  # Guard against double cleanup
        ctx = mp.get_context("spawn")

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)
        self.sampling_params: SamplingParams | None = None
        self._decode_step_wall_ns: list[int] = []
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True

        worker_id = os.environ.get("SLURM_PROCID")
        worker = f" worker={worker_id}" if worker_id is not None else ""
        print(
            f"Engine stats{worker}: "
            f"preemptions={self.scheduler.num_preemptions}",
            flush=True,
        )

        if self.model_runner is not None:
            self.model_runner.call("exit")
            self.model_runner = None

        for rank, process in enumerate(self.ps, start=1):
            process.join(timeout=10)
            if process.is_alive():
                logger.warning("Worker %d did not exit; terminating it", rank)
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    logger.warning("Worker %d did not terminate; killing it", rank)
                    process.kill()
        self.ps.clear()

    def get_stats(self) -> dict[str, int]:
        return {"preemptions": self.scheduler.num_preemptions}

    def _validate_sampling_params(self, sampling_params: SamplingParams) -> None:
        block_size = sampling_params.diffusion_block_size
        if block_size > self.config.max_diffusion_block_size:
            raise ValueError(
                f"SamplingParams.diffusion_block_size={block_size} exceeds "
                "LLM max_diffusion_block_size="
                f"{self.config.max_diffusion_block_size}; recreate the engine "
                "with a larger maximum so its buffers and CUDA graphs are prepared"
            )

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        self._validate_sampling_params(sampling_params)
        if self.scheduler.is_finished():
            self.sampling_params = sampling_params
        elif self.sampling_params is not sampling_params:
            raise ValueError(
                "All active requests must share one SamplingParams object"
            )
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        step_start_ns = perf_counter_ns() if self.config.decode_timing else 0
        sampling_params = self.sampling_params
        if sampling_params is None:
            raise RuntimeError("Cannot step without active sampling parameters")
        seqs, is_prefill = self.scheduler.schedule(sampling_params)
        num_prefill_tokens = (
            sum(len(seq) - seq.num_cached_tokens for seq in seqs)
            if is_prefill
            else 0
        )
        result = self.model_runner.call(
            "run",
            seqs,
            is_prefill,
            sampling_params,
        )

        # Make only KV-complete full pages visible to subsequent scheduler batches.
        for seq in seqs:
            self.scheduler.block_manager.publish_computed_prefix_blocks(seq)

        self.scheduler.postprocess(seqs, result, sampling_params)
        num_new_tokens = sum(len(token_ids) for token_ids in result)

        outputs = [
            (seq.seq_id, seq.completion_token_ids, dict(seq.stats))
            for seq in seqs
            if seq.is_finished
        ]
        if self.config.decode_timing and not is_prefill:
            self._decode_step_wall_ns.append(perf_counter_ns() - step_start_ns)
        num_tokens = num_prefill_tokens if is_prefill else -num_new_tokens
        return outputs, num_tokens

    def reset_decode_timing(self) -> None:
        self._decode_step_wall_ns.clear()
        self.model_runner.call("reset_decode_timing")

    def get_decode_timing(self) -> dict[str, object]:
        model_timing = self.model_runner.call("get_decode_timing")
        forward_records = model_timing["forward_records"]
        gpu_by_step: dict[int, float] = {}
        for record in forward_records:
            step_index = int(record["step_index"])
            gpu_by_step[step_index] = (
                gpu_by_step.get(step_index, 0.0) + float(record["gpu_ms"])
            )
        step_wall_ms = [value / 1e6 for value in self._decode_step_wall_ns]
        model_gpu_ms = [
            gpu_by_step.get(step_index, 0.0)
            for step_index in range(1, len(step_wall_ms) + 1)
        ]
        return {
            "enabled": bool(model_timing["enabled"]),
            "timing_scope": "decode_steps_only",
            "model_gpu_scope": "backbone_and_lm_head",
            "step_wall_ms": step_wall_ms,
            "model_gpu_ms": model_gpu_ms,
            "exposed_non_model_ms": [
                wall_ms - gpu_ms
                for wall_ms, gpu_ms in zip(step_wall_ms, model_gpu_ms)
            ],
            "forward_records": forward_records,
        }

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
        detokenize: bool = True,
        progress_callback: Callable[[list[dict[str, int]]], None] | None = None,
    ) -> list[dict[str, object]]:
        if isinstance(sampling_params, list):
            raise ValueError(
                "generate() now accepts exactly one SamplingParams object per call. "
                "All prompts in the call share that sampling configuration."
            )
        self._validate_sampling_params(sampling_params)
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        seq_ids = []
        for prompt in prompts:
            seq_ids.append(self.add_request(prompt, sampling_params))
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids, stats in output:
                outputs[seq_id] = (token_ids, stats)
            if output:
                completed = len(output)
                if use_tqdm:
                    pbar.update(completed)
                if progress_callback is not None:
                    progress_callback([stats for _, _, stats in output])
        outputs = [outputs[seq_id] for seq_id in seq_ids]

        outputs = [
            {"token_ids": token_ids, "stats": stats}
            for token_ids, stats in outputs
        ]
        if detokenize:
            for output in outputs:
                output["text"] = self.tokenizer.decode(output["token_ids"])
        if use_tqdm:
            pbar.close()
        return outputs
