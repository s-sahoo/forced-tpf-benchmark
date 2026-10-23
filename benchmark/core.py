from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CALIBRATION = REPO / "configs/forced_tpf_calibration.json"
FIGURE_DIR = REPO / "configs/figures"
TIMING_SCOPE = "end_to_end_including_prefill"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Point:
    experiment_id: str
    model_id: str
    configuration_id: str
    adapter: str
    batch_size: int
    repetition: int
    input_len: int
    output_len: int
    top_p: float
    top_k: int
    parameters: dict[str, Any]
    engine_forced_tpf: float | None

    def identity(self, implementation: dict[str, Any]) -> dict[str, Any]:
        return {**asdict(self), "implementation": implementation}

    def digest(self, implementation: dict[str, Any]) -> str:
        encoded = json.dumps(
            self.identity(implementation), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:20]

    @property
    def safe_name(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.experiment_id)


def resolve_experiment(catalog: dict[str, Any], experiment_id: str) -> tuple[str, str, dict[str, Any]]:
    try:
        model_id, configuration_id = experiment_id.split(".", 1)
        entry = catalog[model_id][configuration_id]
    except (ValueError, KeyError) as error:
        raise ValueError(f"unknown experiment id: {experiment_id}") from error
    parameters = dict(entry.get("parameters", {}))
    if "temperature" not in parameters:
        raise ValueError(f"{experiment_id} omits parameters.temperature")
    return model_id, configuration_id, entry


def load_points(figures: list[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]], list[Point]]:
    catalog = read_json(CALIBRATION)
    workload = catalog["benchmark"]
    views: dict[str, list[dict[str, str]]] = {}
    unique: dict[tuple[Any, ...], Point] = {}
    for figure in figures:
        path = FIGURE_DIR / f"{figure}.json"
        manifest = read_json(path)
        experiments = manifest["experiments"]
        views[figure] = experiments
        for view in experiments:
            model_id, configuration_id, entry = resolve_experiment(catalog, view["id"])
            for repetition in range(int(workload["timed_repetitions"])):
                for batch_size in workload["batch_sizes"]:
                    point = Point(
                        experiment_id=view["id"],
                        model_id=model_id,
                        configuration_id=configuration_id,
                        adapter=view["adapter"],
                        batch_size=int(batch_size),
                        repetition=repetition,
                        input_len=int(workload["input_len"]),
                        output_len=int(workload["output_len"]),
                        top_p=float(workload["top_p"]),
                        top_k=int(workload["top_k"]),
                        parameters=dict(entry["parameters"]),
                        engine_forced_tpf=entry.get("engine_forced_tpf"),
                    )
                    key = (
                        point.experiment_id,
                        point.adapter,
                        point.batch_size,
                        point.repetition,
                    )
                    unique[key] = point
    return catalog, views, list(unique.values())


def result_throughput(result: dict[str, Any]) -> float:
    if result.get("aggregate_tps") is not None:
        return float(result["aggregate_tps"])
    raise ValueError("result omits aggregate throughput")


def validate_result(result: dict[str, Any], point: Point) -> None:
    expected_tokens = point.batch_size * point.output_len
    if int(result.get("schema_version", -1)) != 4:
        raise ValueError("result uses an unsupported schema")
    if result.get("status") != "complete":
        raise ValueError(f"status is {result.get('status')!r}")
    if int(result.get("generated_tokens", -1)) != expected_tokens:
        raise ValueError("generated token count does not match the requested cohort")
    if result.get("timing_scope") != TIMING_SCOPE:
        raise ValueError("timing scope does not include admission, prefill, and decode")
    if result.get("full_batch_resident") is not True:
        raise ValueError("the full request cohort was not resident")
    if int(result.get("num_preemptions", 0)) != 0:
        raise ValueError("request cohort was preempted")
    if any(result.get("per_request_retractions") or []):
        raise ValueError("request cohort was retracted")
    if result.get("cuda_graph_misses") not in (None, 0):
        raise ValueError("timed decode had CUDA-graph misses")
    peak_gpu_memory_mib = result.get("peak_gpu_memory_mib")
    if peak_gpu_memory_mib is None or int(peak_gpu_memory_mib) <= 0:
        raise ValueError("result omits positive device-level peak GPU memory")
    if point.engine_forced_tpf is not None:
        requested = float(result.get("requested_tpf", float("nan")))
        if abs(requested - point.engine_forced_tpf) > 1e-9:
            raise ValueError("forced TPF does not match the catalog")
        measured = result.get("measured_tpf")
        tolerance = float(result.get("forced_tpf_tolerance") or 0.025)
        if measured is None or abs(float(measured) - requested) > tolerance:
            raise ValueError("measured TPF missed the requested forced TPF")
    result_throughput(result)
