"""Run Figure 2 Nano-vLLM-UNO GPU measurements and aggregate their traces."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import statistics
import subprocess
import threading
from pathlib import Path

from benchmark.core import REPO, read_json, write_json


RUNNER = REPO / "benchmark/runners/uno_cycle.py"
NANO_ROOT = REPO / "engines/nano-vllm-uno"
ENGINE = "nano"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is unset; source benchmark.env")
    return value


def command(repetition: int, out_root: Path) -> tuple[list[str], Path, Path]:
    result = out_root / "results" / f"{ENGINE}-r{repetition}.json"
    trace = out_root / "traces" / f"{ENGINE}-r{repetition}.json"
    args = [
        required_env("NANO_PYTHON"),
        str(RUNNER),
        "--model", required_env("QWEN_BASE_MODEL"),
        "--lora-path", required_env("QWEN_UNO_LORA"),
        "--result-path", str(result),
        "--trace-path", str(trace),
        "--block-size", "4",
        "--batch-size", "1",
        "--input-len", "1000",
        "--output-len", "5000",
        "--seed", str(20260809 + repetition),
        "--temperature", "1",
        "--top-p", "0.95",
        "--top-k", "50",
        "--prompt-vocab-size", "151936",
        "--excluded-prompt-token-id", "151669",
        "--fa3-provider", "sgl_kernel",
    ]
    return args, result, trace


def launch_environment(repetition: int, gpu: int, out_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    cache = out_root / "cache" / f"gpu{gpu}" / ENGINE
    python = Path(required_env("NANO_PYTHON"))
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PATH": f"{python.parent}:{env.get('PATH', '')}",
            "PYTHONPATH": str(NANO_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_DISTRIBUTED_PORT": str(26000 + repetition),
            "NANO_ATTENTION_BACKEND": "fa3",
            "NANO_FA3_PROVIDER": "sgl_kernel",
        }
    )
    return env


def run_repetition(repetition: int, gpu: int, out_root: Path) -> None:
    engine = ENGINE
    args, result, trace = command(repetition, out_root)
    directory = out_root / "commands"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / f"{engine}-r{repetition}.json",
        {"command": args, "gpu": gpu, "result": str(result), "trace": str(trace)},
    )
    log = out_root / "logs" / f"{engine}-r{repetition}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {engine} r{repetition} gpu={gpu}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            args,
            cwd=NANO_ROOT,
            env=launch_environment(repetition, gpu, out_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"{engine} repetition {repetition} failed; see {log}")
    if not result.exists() or read_json(result).get("status") != "complete":
        raise RuntimeError(f"{engine} repetition {repetition} failed; see {log}")
    print(f"END   {engine} r{repetition}", flush=True)


def aggregate(out_root: Path, repetitions: int) -> list[dict[str, str]]:
    rows = []
    for engine, label in ((ENGINE, "Nano-vLLM-UNO"),):
        paths = [
            out_root / "results" / f"{engine}-r{index}.json"
            for index in range(repetitions)
        ]
        results = [read_json(path) for path in paths]
        if any(result.get("status") != "complete" for result in results):
            raise RuntimeError(f"incomplete {engine} repetitions")
        draft = statistics.median(
            float(result["timing"]["draft_forward_ms"]["p50"])
            for result in results
        )
        verify = statistics.median(
            float(result["timing"]["verify_forward_ms"]["p50"])
            for result in results
        )
        total = statistics.median(
            float(result["timing"]["cycle_wall_ms"]["p50"])
            for result in results
        )
        rows.append(
            {
                "engine": engine,
                "engine_label": label,
                "draft_forward_ms": f"{draft:.6f}",
                "verify_forward_ms": f"{verify:.6f}",
                "other_overhead_ms": f"{total - draft - verify:.6f}",
                "total_ms": f"{total:.6f}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value]
    if not gpus or args.repetitions < 1:
        raise ValueError("at least one GPU and one repetition are required")
    args.out_root.mkdir(parents=True, exist_ok=True)
    gpu_queue: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    errors = []
    lock = threading.Lock()

    def worker(repetition: int) -> None:
        gpu = gpu_queue.get()
        try:
            run_repetition(repetition, gpu, args.out_root)
        except BaseException as error:
            with lock:
                errors.append(error)
        finally:
            gpu_queue.put(gpu)

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(args.repetitions)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("Figure 2 GPU run failed") from errors[0]

    rows = aggregate(args.out_root, args.repetitions)
    write_csv(args.out_root / "figure2.csv", rows)
    write_json(
        args.out_root / "summary.json",
        {"rows": rows, "repetitions": args.repetitions},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
