from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.core import REPO, file_digest, read_json


@dataclass
class Launch:
    command: list[str]
    cwd: Path
    pythonpath: Path
    source_root: Path
    runner_paths: list[Path]
    environment: dict[str, str] = field(default_factory=dict)

    def implementation(self) -> dict[str, Any]:
        provenance_path = self.source_root / "SOURCE_PROVENANCE.json"
        return {
            "adapter": {
                str(path.relative_to(REPO)): file_digest(path)
                for path in self.runner_paths
            },
            "engine": (
                read_json(provenance_path) if provenance_path.exists() else None
            ),
        }


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is unset; configure assets.env or set "
            "FORCED_TPF_ASSET_ENV before sourcing benchmark.env"
        )
    return value


def common_args(point, result_path: Path, model: str) -> list[str]:
    return [
        "--model", model,
        "--batch-size", str(point.batch_size),
        "--input-len", str(point.input_len),
        "--output-len", str(point.output_len),
        "--warmup-output-len", "32",
        "--seed", str(20260807 + point.repetition),
        "--temperature", str(point.parameters["temperature"]),
        "--top-p", str(point.top_p),
        "--top-k", str(point.top_k),
        "--result-path", str(result_path),
    ]


PROMPTS = {
    "qwen3_8b": (151_936, 151_669),
    "bbq_8b": (250_624, 151_669),
    "nemotron_diffusion_14b": (131_072, 100),
    "diffusion_gemma_26b": (262_144, None),
}


def prompt_args(model_id: str, include_vocab: bool) -> list[str]:
    vocab_size, excluded = PROMPTS[model_id]
    result = ["--prompt-vocab-size", str(vocab_size)] if include_vocab else []
    if excluded is not None:
        result += ["--excluded-prompt-token-id", str(excluded)]
    return result


def prefill_batches(batch_size: int, input_len: int, budget: int) -> list[int]:
    width = max(1, budget // input_len)
    batches = []
    remaining = batch_size
    while remaining:
        current = min(width, remaining)
        batches.append(current)
        remaining -= current
    return batches
