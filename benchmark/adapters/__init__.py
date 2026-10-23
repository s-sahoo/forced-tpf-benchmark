"""Resolve a benchmark point into one engine-family launch policy."""

from __future__ import annotations

from pathlib import Path

from .base import Launch
from .nano import launch as launch_nano
from .sglang import launch as launch_sglang
from .vllm import launch as launch_vllm


def build_launch(point, task_dir, result_path) -> Launch:
    family = point.adapter.split("/", 1)[0]
    builders = {
        "nano": launch_nano,
        "sglang": launch_sglang,
        "vllm": launch_vllm,
    }
    try:
        launch = builders[family](point, task_dir, result_path)
    except KeyError as error:
        raise ValueError(f"unknown adapter: {point.adapter}") from error
    launch.runner_paths.extend(
        [
            Path(__file__),
            Path(__file__).with_name("base.py"),
            Path(__file__).with_name(f"{family}.py"),
        ]
    )
    return launch
