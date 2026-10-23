from __future__ import annotations

from pathlib import Path

from benchmark.core import REPO
from .base import Launch, common_args, prefill_batches, prompt_args, required_env


ROOT = REPO / "engines/nano-vllm-uno"
RUNNERS = Path(__file__).resolve().parents[1] / "runners"
UNO_RUNNER = RUNNERS / "nano_uno.py"
AR_RUNNER = RUNNERS / "nano_ar.py"
TIMING = RUNNERS / "timing.py"
PROVIDER = RUNNERS / "nano_fa3.py"


def model_and_lora(model_id: str) -> tuple[str, str | None]:
    if model_id == "qwen3_8b":
        return required_env("QWEN_BASE_MODEL"), required_env("QWEN_UNO_LORA")
    if model_id == "bbq_8b":
        return required_env("BBQ_BASE_MODEL"), required_env("BBQ_GATED_LORA")
    raise ValueError(f"Nano has no model policy for {model_id}")


def memory_fraction(batch_size: int) -> float:
    if batch_size >= 128:
        return 0.95
    if batch_size >= 64:
        return 0.86
    return 0.85


def launch(point, task_dir: Path, result_path: Path) -> Launch:
    del task_dir
    mode = point.adapter.split("/", 1)[1]
    model, lora = model_and_lora(point.model_id)
    python = required_env("NANO_PYTHON")
    common = common_args(point, result_path, model)
    if mode == "uno":
        block_size = int(point.parameters["block_size"])
        verify_size = point.parameters.get("tree_verify_size")
        reserve = max(block_size, int(verify_size or 0))
        budget = max(
            point.input_len + point.output_len + reserve,
            min(64_000, point.batch_size * point.input_len),
        )
        command = [
            python, str(UNO_RUNNER),
            "--method", point.configuration_id,
            "--engine", f"nano_{point.model_id}_uno",
            "--attention-backend", "fa3",
            "--fa3-provider", "sgl_kernel",
            "--gated-lora-path", str(lora),
            "--block-size", str(block_size),
            "--forced-tpf", str(point.engine_forced_tpf),
            "--gpu-memory-utilization", str(memory_fraction(point.batch_size)),
            "--max-num-batched-tokens", str(budget),
            "--expected-prefill-batch-sizes",
            *map(str, prefill_batches(point.batch_size, point.input_len, budget)),
            *prompt_args(point.model_id, include_vocab=True),
            *common,
        ]
        if verify_size is not None:
            command += [
                "--tree-verify-size", str(verify_size),
                "--tree-candidate-top-k", str(point.parameters["tree_candidate_top_k"]),
            ]
        runners = [UNO_RUNNER, TIMING, PROVIDER]
    elif mode == "ar":
        command = [
            python, str(AR_RUNNER),
            "--fa3-provider", "sgl_kernel",
            "--gpu-memory-utilization", str(memory_fraction(point.batch_size)),
            *prompt_args(point.model_id, include_vocab=True),
            *common,
        ]
        runners = [AR_RUNNER, TIMING, PROVIDER]
    else:
        raise ValueError(f"unknown Nano adapter: {point.adapter}")
    return Launch(
        command=command,
        cwd=ROOT,
        pythonpath=ROOT,
        source_root=ROOT,
        runner_paths=runners,
        environment={
            "NANO_ATTENTION_BACKEND": "fa3",
            "NANO_FA3_PROVIDER": "sgl_kernel",
        },
    )
