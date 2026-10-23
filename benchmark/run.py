from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import re
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark.adapters import build_launch
from benchmark.core import Point, load_points, read_json, result_throughput, validate_result, write_json
from benchmark.plot import render


CAPACITY = re.compile(
    r"out of memory|resident[._ ]capacity|cannot remain resident|"
    r"kv.?cache[^\n]{0,160}capacity|free_blocks=0",
    re.IGNORECASE,
)


class PeakGpuMemory:
    """Sample total device memory without synchronizing the timed CUDA work."""

    def __init__(self, gpu: int, interval_ms: int = 100) -> None:
        self.gpu = gpu
        self.interval_ms = interval_ms
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.messages: list[str] = []
        self.first_output = threading.Event()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            value = line.strip()
            try:
                self.samples.append(int(value))
            except ValueError:
                if value:
                    self.messages.append(value)
            finally:
                self.first_output.set()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={self.gpu}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if not self.first_output.wait(timeout=10):
            self.stop()
            raise RuntimeError(f"nvidia-smi did not report memory for GPU {self.gpu}")
        if not self.samples:
            message = "; ".join(self.messages[-3:]) or "no numeric samples"
            self.stop()
            raise RuntimeError(
                f"nvidia-smi could not monitor GPU {self.gpu}: {message}"
            )

    def stop(self) -> int:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)
        if not self.samples:
            message = "; ".join(self.messages[-3:]) or "no numeric samples"
            raise RuntimeError(
                f"nvidia-smi produced no memory samples for GPU {self.gpu}: {message}"
            )
        return max(self.samples)


@dataclass
class Task:
    point: Point
    key: str
    directory: Path
    result_path: Path
    launch: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run, reuse, aggregate, and plot paper figure experiments."
    )
    parser.add_argument("--figures", nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--paired-repetitions",
        action="store_true",
        help=(
            "assign one repetition to each shard and one batch size to each GPU; "
            "methods run sequentially on that fixed GPU"
        ),
    )
    parser.add_argument(
        "--balanced-repetitions",
        action="store_true",
        help=(
            "place each point's repetitions on distinct shards and dynamically "
            "balance independent points across GPUs"
        ),
    )
    parser.add_argument("--gpu-cooldown-s", type=float, default=5.0)
    return parser.parse_args()


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def make_tasks(points: list[Point], out_root: Path) -> list[Task]:
    tasks = []
    for point in points:
        staging = out_root / ".planning" / f"{point.safe_name}_c{point.batch_size}_r{point.repetition}"
        staging.mkdir(parents=True, exist_ok=True)
        provisional = build_launch(point, staging, staging / "result.json")
        implementation = provisional.implementation()
        implementation["launch"] = {
            "command": [
                argument.replace(str(staging), "<TASK_DIR>")
                for argument in provisional.command
            ],
            "cwd": str(provisional.cwd),
            "pythonpath": str(provisional.pythonpath),
            "environment": provisional.environment,
        }
        key = point.digest(implementation)
        directory = out_root / "experiments" / key
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        launch = build_launch(point, directory, result_path)
        write_json(directory / "spec.json", point.identity(implementation))
        tasks.append(Task(point, key, directory, result_path, launch))
    return tasks


def plan_tasks(points: list[Point], out_root: Path) -> list[Task]:
    lock_path = out_root / ".planning.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return make_tasks(points, out_root)


def valid(task: Task) -> bool:
    try:
        validate_result(read_json(task.result_path), task.point)
        return True
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return False


def infeasible(task: Task) -> bool:
    try:
        result = read_json(task.result_path)
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return False
    return (
        result.get("status") == "infeasible"
        and result.get("batch_size") == task.point.batch_size
    )


def environment(task: Task, gpu: int, shard: int, cache_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    python = Path(task.launch.command[0])
    executable_dirs = [python.parent]
    shared_runtime_env = {
        "tools-python": "tools",
        "nano-python": "sglang-spec",
        "sglang-spec-python": "sglang-spec",
        "sglang-diffusion-python": "sglang-diffusion",
        "vllm-diffgemma-python": "vllm-diffgemma",
    }.get(python.name)
    if shared_runtime_env is not None:
        runtime_bin = python.parent.parent / "envs" / shared_runtime_env / "bin"
        if runtime_bin.is_dir():
            executable_dirs.insert(0, runtime_bin)
    cache = cache_root / f"shard{shard}" / f"gpu{gpu}" / task.point.adapter.replace("/", "-")
    cache.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PATH": ":".join([*(str(path) for path in executable_dirs), env.get("PATH", "")]),
            "PYTHONPATH": str(task.launch.pythonpath),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_MODULES_CACHE": str(cache / "hf_modules"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_DISTRIBUTED_PORT": str(28000 + gpu + 100 * shard),
            **task.launch.environment,
        }
    )
    return env


def run_one(
    task: Task,
    args: argparse.Namespace,
    shard: int,
    gpu_queue: queue.Queue[int],
    print_lock: threading.Lock,
) -> dict[str, Any]:
    if not args.rerun:
        if valid(task):
            return {"key": task.key, "status": "reused"}
        if infeasible(task):
            return {"key": task.key, "status": "infeasible", "reused": True}
    gpu = gpu_queue.get()
    launched = False
    try:
        record = {
            "key": task.key,
            "point": asdict(task.point),
            "command": task.launch.command,
            "cwd": str(task.launch.cwd),
            "gpu": gpu,
            "hostname": socket.gethostname(),
        }
        write_json(task.directory / "command.json", record)
        if args.dry_run:
            return {**record, "status": "dry-run"}
        launched = True
        log_path = task.directory / "run.log"
        with print_lock:
            print(
                f"START gpu={gpu} {task.point.experiment_id} "
                f"{task.point.adapter} C{task.point.batch_size}",
                flush=True,
            )
        started = time.time()
        memory = PeakGpuMemory(gpu)
        memory.start()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                code = subprocess.call(
                    task.launch.command,
                    cwd=task.launch.cwd,
                    env=environment(task, gpu, shard, args.out_root / "cache"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        finally:
            peak_gpu_memory_mib = memory.stop()
        if task.result_path.exists():
            result = read_json(task.result_path)
            result["peak_gpu_memory_mib"] = peak_gpu_memory_mib
            write_json(task.result_path, result)
        status = "complete" if code == 0 and valid(task) else "failed"
        if status == "failed":
            result = read_json(task.result_path) if task.result_path.exists() else {}
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-20_000:]
            capacity_error = CAPACITY.search(tail)
            if result.get("status") == "infeasible" or capacity_error:
                status = "infeasible"
                if result.get("status") != "infeasible":
                    write_json(
                        task.result_path,
                        {
                            "schema_version": 4,
                            "status": "infeasible",
                            "reason": "resident_capacity",
                            "detected_by": "launcher_log",
                            "engine": task.point.adapter,
                            "algorithm": None,
                            "model": None,
                            "batch_size": task.point.batch_size,
                            "input_len": task.point.input_len,
                            "output_len": task.point.output_len,
                            "temperature": task.point.parameters["temperature"],
                            "top_p": task.point.top_p,
                            "top_k": task.point.top_k,
                            "seed": 20260807 + task.point.repetition,
                            "expected_tokens": (
                                task.point.batch_size * task.point.output_len
                            ),
                            "generated_tokens": 0,
                            "requested_tpf": task.point.engine_forced_tpf,
                            "measured_tpf": None,
                            "end_to_end_elapsed_s": None,
                            "aggregate_tps": None,
                            "per_user_tps": None,
                            "peak_gpu_memory_mib": peak_gpu_memory_mib,
                            "timing_scope": None,
                            "full_batch_resident": False,
                            "num_preemptions": None,
                            "per_request_retractions": None,
                            "per_request_token_counts": None,
                            "cuda_graph_hits": None,
                            "cuda_graph_misses": None,
                        },
                    )
        elapsed = time.time() - started
        with print_lock:
            print(
                f"END   gpu={gpu} {task.point.experiment_id} "
                f"C{task.point.batch_size} {status} {elapsed:.1f}s",
                flush=True,
            )
        return {
            **record,
            "status": status,
            "returncode": code,
            "elapsed_s": elapsed,
            "peak_gpu_memory_mib": peak_gpu_memory_mib,
        }
    finally:
        if launched and args.gpu_cooldown_s:
            time.sleep(args.gpu_cooldown_s)
        gpu_queue.put(gpu)


def rotate_block(tasks: list[Task], repetition: int, repetitions: int) -> list[Task]:
    ordered = sorted(
        tasks,
        key=lambda task: (task.point.experiment_id, task.point.adapter),
    )
    if not ordered:
        return ordered
    offset = repetition * len(ordered) // repetitions
    return ordered[offset:] + ordered[:offset]


def run_paired(
    tasks: list[Task],
    args: argparse.Namespace,
    shard: int,
    shards: int,
    gpus: list[int],
) -> list[dict[str, Any]]:
    repetitions = sorted({task.point.repetition for task in tasks})
    if repetitions != list(range(shards)):
        raise ValueError(
            "paired scheduling requires one shard per repetition: "
            f"repetitions={repetitions}, shards={shards}"
        )
    selected = [task for task in tasks if task.point.repetition == shard]
    batches = sorted({task.point.batch_size for task in selected})
    if len(batches) > len(gpus):
        raise ValueError(
            f"paired scheduling needs at least {len(batches)} GPUs, got {len(gpus)}"
        )
    lock = threading.Lock()

    def run_batch(batch: int, gpu: int) -> list[dict[str, Any]]:
        block = [task for task in selected if task.point.batch_size == batch]
        block = rotate_block(block, shard, shards)
        fixed_gpu: queue.Queue[int] = queue.Queue()
        fixed_gpu.put(gpu)
        return [run_one(task, args, shard, fixed_gpu, lock) for task in block]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batches)) as pool:
        futures = [
            pool.submit(run_batch, batch, gpu)
            for batch, gpu in zip(batches, gpus)
        ]
        return [record for future in futures for record in future.result()]


def balanced_assignments(tasks: list[Task], shards: int) -> dict[str, int]:
    groups: dict[bytes, list[Task]] = {}
    for task in tasks:
        identity = asdict(task.point)
        identity.pop("repetition")
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode()
        groups.setdefault(encoded, []).append(task)

    loads = [0] * shards
    assignments: dict[str, int] = {}
    ordered_groups = sorted(
        groups.items(), key=lambda item: hashlib.sha256(item[0]).digest()
    )
    for encoded, group in ordered_groups:
        if len(group) > shards:
            raise ValueError(
                "balanced scheduling needs at least one shard per repetition: "
                f"repetitions={len(group)}, shards={shards}"
            )
        digest = hashlib.sha256(encoded).digest()
        offset = int.from_bytes(digest[:8], "big") % shards
        candidates = sorted(
            range(shards),
            key=lambda candidate: (
                loads[candidate],
                (candidate - offset) % shards,
            ),
        )
        for task, target in zip(
            sorted(group, key=lambda item: item.point.repetition),
            candidates,
        ):
            assignments[task.key] = target
            loads[target] += 1
    return assignments


def aggregate(figures: list[str], views: dict[str, list[dict[str, str]]], tasks: list[Task], out_root: Path) -> dict[str, Any]:
    indexed: dict[tuple[str, str, int], list[Task]] = {}
    for task in tasks:
        indexed.setdefault(
            (task.point.experiment_id, task.point.adapter, task.point.batch_size), []
        ).append(task)
    summary: dict[str, Any] = {}
    for figure in figures:
        rows = []
        infeasible_points = []
        missing = []
        for view in views[figure]:
            for batch_size in sorted({task.point.batch_size for task in tasks}):
                repetitions = indexed.get((view["id"], view["adapter"], batch_size), [])
                if repetitions and all(infeasible(task) for task in repetitions):
                    infeasible_points.append(
                        f"{view['id']}:{view['adapter']}:C{batch_size}"
                    )
                    continue
                if not repetitions or not all(valid(task) for task in repetitions):
                    missing.append(f"{view['id']}:{view['adapter']}:C{batch_size}")
                    continue
                results = [read_json(task.result_path) for task in repetitions]
                task = repetitions[0]
                tps = statistics.median(map(result_throughput, results))
                peak_gpu_memory_mib = max(
                    int(result["peak_gpu_memory_mib"]) for result in results
                )
                measured_values = [
                    value
                    for result in results
                    if (value := result.get("measured_tpf"))
                    is not None
                ]
                rows.append(
                    {
                        "figure": figure,
                        "curve": view["curve"],
                        "experiment_id": task.point.experiment_id,
                        "model_id": task.point.model_id,
                        "configuration_id": task.point.configuration_id,
                        "adapter": task.point.adapter,
                        "batch_size": batch_size,
                        "input_len": task.point.input_len,
                        "output_len": task.point.output_len,
                        "temperature": task.point.parameters["temperature"],
                        "requested_tpf": task.point.engine_forced_tpf,
                        "measured_tpf": (
                            statistics.median(map(float, measured_values))
                            if measured_values
                            else None
                        ),
                        "aggregate_tps": tps,
                        "per_user_tps": tps / batch_size,
                        "peak_gpu_memory_mib": peak_gpu_memory_mib,
                    }
                )
        output = out_root / "figures" / figure
        frontier = render(figure, rows, output) if rows else []
        summary[figure] = {
            "valid_points": len(rows),
            "frontier_points": len(frontier),
            "infeasible_points": infeasible_points,
            "missing_points": missing,
            "output": str(output),
        }
    write_json(out_root / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    _, views, points = load_points(args.figures)
    args.out_root.mkdir(parents=True, exist_ok=True)
    tasks = plan_tasks(points, args.out_root)
    if args.aggregate_only:
        summary = aggregate(args.figures, views, tasks, args.out_root)
        return int(any(item["missing_points"] for item in summary.values()))

    shard = args.shard_index if args.shard_index is not None else int(os.environ.get("SLURM_PROCID", "0"))
    shards = args.num_shards if args.num_shards is not None else int(os.environ.get("SLURM_NTASKS", "1"))
    if not 0 <= shard < shards:
        raise ValueError(f"invalid shard {shard}/{shards}")
    gpus = comma_ints(args.gpus)
    if not gpus:
        raise ValueError("at least one GPU is required")
    if args.paired_repetitions and args.balanced_repetitions:
        raise ValueError("choose either paired or balanced repetition scheduling")
    if args.paired_repetitions:
        records = run_paired(tasks, args, shard, shards, gpus)
    else:
        if args.balanced_repetitions:
            assignments = balanced_assignments(tasks, shards)
            selected = [
                task for task in tasks if assignments[task.key] == shard
            ]
        else:
            selected = [
                task for index, task in enumerate(tasks) if index % shards == shard
            ]
        selected.sort(
            key=lambda task: (task.point.batch_size, task.point.experiment_id),
            reverse=True,
        )
        gpu_queue: queue.Queue[int] = queue.Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)
        lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            records = list(
                pool.map(
                    lambda task: run_one(task, args, shard, gpu_queue, lock),
                    selected,
                )
            )
    write_json(args.out_root / f"records-shard{shard}.json", records)
    if args.dry_run or shards > 1:
        return int(any(record["status"] == "failed" for record in records))
    summary = aggregate(args.figures, views, tasks, args.out_root)
    return int(
        any(record["status"] == "failed" for record in records)
        or any(item["missing_points"] for item in summary.values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
