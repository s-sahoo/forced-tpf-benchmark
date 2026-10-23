from __future__ import annotations

from pathlib import Path

from benchmark.core import REPO
from .base import Launch, prompt_args, required_env


ROOT = REPO / "engines/vllm-diffgemma-timing"
RUNNERS = Path(__file__).resolve().parents[1] / "runners"
RUNNER = RUNNERS / "diffusion.py"
TIMING = RUNNERS / "timing.py"


def launch(point, task_dir: Path, result_path: Path) -> Launch:
    del task_dir
    if point.adapter != "vllm/diffgemma":
        raise ValueError(f"unknown vLLM adapter: {point.adapter}")
    context_length = point.input_len + point.output_len + 512
    prefill_budget = max(context_length, point.batch_size * point.input_len)
    # Large admission probes may not fit the piecewise-prefill CUDA graph even
    # when their decode graph does. Keep decode graphed and let the runner read
    # vLLM's resolved KV capacity before attempting any generation.
    cudagraph_mode = (
        "FULL_AND_PIECEWISE" if point.batch_size <= 32 else "FULL_DECODE_ONLY"
    )
    python = Path(required_env("VLLM_DIFFGEMMA_PYTHON"))
    if python.name == "vllm-diffgemma-python":
        pythonpath = (
            python.parent.parent
            / "envs/vllm-diffgemma/lib/python3.12/site-packages"
        )
    else:
        pythonpath = python.parent.parent / "lib/python3.12/site-packages"
    command = [
        str(python), str(RUNNER),
        "--engine", "diffgemma_vllm",
        "--model", required_env("DIFFGEMMA_MODEL"),
        "--forced-tpf", str(point.engine_forced_tpf),
        "--batch-size", str(point.batch_size),
        "--input-len", str(point.input_len),
        "--output-len", str(point.output_len),
        "--result-path", str(result_path),
        "--seed", str(20260807 + point.repetition),
        "--temperature", str(point.parameters["temperature"]),
        "--top-p", str(point.top_p),
        "--top-k", str(point.top_k),
        "--warmup-output-len", "256",
        "--gpu-memory-utilization", "0.90",
        "--max-prefill-tokens", str(prefill_budget),
        "--vllm-cudagraph-mode", cudagraph_mode,
        *prompt_args(point.model_id, include_vocab=True),
    ]
    # The install step overlays the vendored Python delta onto a prebuilt vLLM
    # wheel. Import that wheel so its compiled extensions remain available.
    # In particular, do not run from ROOT: vLLM's model registry starts a
    # ``python -m`` subprocess, and ROOT on sys.path would make that subprocess
    # import the unbuilt vendored package instead of the wheel's native modules.
    return Launch(command, REPO, pythonpath, ROOT, [RUNNER, TIMING])
