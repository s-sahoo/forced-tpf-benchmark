from __future__ import annotations

import os
from pathlib import Path

from benchmark.core import REPO
from .base import Launch, common_args, prompt_args, required_env


SPEC_ROOT = REPO / "engines/sglang-eagle3-dflash-timing"
NEMOTRON_ROOT = REPO / "engines/sglang-nemotrondiff-lora-timing"
RUNNERS = Path(__file__).resolve().parents[1] / "runners"
SPEC_RUNNER = RUNNERS / "sglang_spec.py"
AR_RUNNER = RUNNERS / "sglang_ar.py"
DIFFUSION_RUNNER = RUNNERS / "diffusion.py"
TIMING = RUNNERS / "timing.py"


def prefill_budget(point) -> int:
    return max(8192, point.batch_size * point.input_len + 1)


def write_nemotron_config(point, path: Path) -> None:
    write_text(path,
        "\n".join(
            [
                "algorithm: LinearSpec",
                "causal_context: true",
                f"block_size: {point.parameters['block_size']}",
                f"lora_path: {required_env('NEMOTRON_LORA_PATH')}",
                "lora_mode: draft_only",
                f"forced_tpf: {point.engine_forced_tpf}",
                f"forced_tpf_seed: {20260807 + point.repetition}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_text(path: Path, value: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding=encoding)
    temporary.replace(path)


def launch(point, task_dir: Path, result_path: Path) -> Launch:
    mode = point.adapter.split("/", 1)[1]
    if mode == "ar":
        model = required_env("QWEN_BASE_MODEL")
        command = [
            required_env("SGLANG_SPEC_PYTHON"), str(AR_RUNNER),
            "--mem-fraction-static", "0.95" if point.batch_size >= 128 else "0.90",
            *prompt_args(point.model_id, include_vocab=True),
            *common_args(point, result_path, model),
        ]
        return Launch(
            command,
            SPEC_ROOT,
            SPEC_ROOT / "python",
            SPEC_ROOT,
            [AR_RUNNER, TIMING],
        )

    if mode in {"eagle3", "dflash"}:
        block = int(point.parameters["block_size"])
        budget = prefill_budget(point)
        draft_env = "EAGLE3_DRAFT_MODEL" if mode == "eagle3" else "DFLASH_DRAFT_MODEL"
        command = [
            required_env("SGLANG_SPEC_PYTHON"), str(SPEC_RUNNER),
            "--algorithm", mode,
            "--method", point.configuration_id,
            "--engine", f"sglang_{mode}",
            "--model", required_env("QWEN_BASE_MODEL"),
            "--draft-model", required_env(draft_env),
            "--block-size", str(block),
            "--batch-size", str(point.batch_size),
            "--forced-tpf", str(point.engine_forced_tpf),
            "--result-path", str(result_path),
            "--input-len", str(point.input_len),
            "--output-len", str(point.output_len),
            "--warmup-output-len", "32",
            "--seed", str(20260807 + point.repetition),
            "--temperature", str(point.parameters["temperature"]),
            "--top-p", str(point.top_p),
            "--top-k", str(point.top_k),
            "--attention-backend", "fa3",
            "--draft-attention-backend", "fa3",
            "--mem-fraction-static", "0.92" if point.batch_size >= 128 else "0.90",
            "--chunked-prefill-size", str(min(8192, budget)),
            "--max-prefill-tokens", str(budget),
            *prompt_args(point.model_id, include_vocab=True),
        ]
        if mode == "eagle3":
            command += [
                "--speculative-num-steps", str(point.parameters["speculative_num_steps"]),
                "--speculative-eagle-topk", str(point.parameters["speculative_eagle_topk"]),
                "--speculative-num-draft-tokens", str(point.parameters["speculative_num_draft_tokens"]),
            ]
        token_mode = (
            "fixed"
            if mode == "eagle3"
            and int(point.parameters.get("speculative_eagle_topk", 1)) > 1
            else "real-draft-token"
        )
        return Launch(
            command,
            SPEC_ROOT,
            SPEC_ROOT / "python",
            SPEC_ROOT,
            [SPEC_RUNNER, TIMING],
            environment={
                "SGLANG_SIMULATE_ACC_LEN": str(point.engine_forced_tpf),
                "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                "SGLANG_SIMULATE_ACC_TOKEN_MODE": token_mode,
            },
        )

    if mode == "nemotron":
        config = task_dir / "engine.yaml"
        write_nemotron_config(point, config)
        budget = prefill_budget(point)
        command = [
            required_env("SGLANG_PYTHON"), str(DIFFUSION_RUNNER),
            "--engine", "nemotron_sglang",
            "--model", required_env("NEMOTRON_MODEL"),
            "--dllm-config", str(config),
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
            "--prefill-chunk-size", "8192",
            "--max-prefill-tokens", str(budget),
            *prompt_args(point.model_id, include_vocab=True),
        ]
        return Launch(
            command,
            NEMOTRON_ROOT,
            NEMOTRON_ROOT / "python",
            NEMOTRON_ROOT,
            [DIFFUSION_RUNNER, TIMING],
        )

    raise ValueError(f"unknown SGLang adapter: {point.adapter}")
