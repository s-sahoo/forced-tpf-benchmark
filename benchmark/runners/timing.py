from __future__ import annotations

import json
import math
import os
import statistics
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RESULT_SCHEMA_VERSION = 4
# Kept as local constants so runner code can describe its result before the
# common normalizer below enforces the public schema.
FORCED_TPF_SCHEMA_VERSION = RESULT_SCHEMA_VERSION
AR_SCHEMA_VERSION = RESULT_SCHEMA_VERSION
NANO_NORMALIZATION_DIVISOR = 2
TIMING_SCOPE = "end_to_end_including_prefill"

COMMON_RESULT_FIELDS = (
    "schema_version",
    "status",
    "engine",
    "algorithm",
    "model",
    "batch_size",
    "input_len",
    "output_len",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "requested_tpf",
    "measured_tpf",
    "generated_tokens",
    "expected_tokens",
    "end_to_end_elapsed_s",
    "aggregate_tps",
    "per_user_tps",
    "peak_gpu_memory_mib",
    "timing_scope",
    "full_batch_resident",
    "num_preemptions",
    "per_request_retractions",
    "per_request_token_counts",
    "cuda_graph_hits",
    "cuda_graph_misses",
)


def standardize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize fields shared by every engine while retaining diagnostics."""
    normalized = dict(result)
    normalized["schema_version"] = RESULT_SCHEMA_VERSION
    for field in COMMON_RESULT_FIELDS:
        normalized.setdefault(field, None)
    if normalized["per_user_tps"] is None and normalized["aggregate_tps"] is not None:
        batch_size = int(normalized["batch_size"])
        normalized["per_user_tps"] = float(normalized["aggregate_tps"]) / batch_size
    return normalized


@dataclass(frozen=True)
class ForcedTpfTask:
    method: str
    engine: str
    block_size: int
    batch_size: int
    forced_tpf: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def cap_two_pass_tpf(requested_tpf: float, block_size: int) -> float:
    if requested_tpf < 1.0:
        raise ValueError("forced_tpf must be at least 1")
    return min(requested_tpf, (block_size + 1) / 2.0)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(samples),
        "sum": sum(samples),
        "mean": statistics.fmean(samples),
        "p05": percentile(samples, 0.05),
        "p50": statistics.median(samples),
        "p95": percentile(samples, 0.95),
        "min": min(samples),
        "max": max(samples),
    }


def _component_summary(
    wall: list[float],
    model_gpu: list[float],
    exposed_non_model: list[float],
) -> dict[str, Any]:
    # Under overlap, launch timestamps are a CPU-side cadence while the exact
    # graph events live on the GPU stream. Rare adjacent cadence samples can
    # borrow time from one another (one negative residual followed by a larger
    # positive residual), so only the aggregate additive invariant is valid.
    if sum(exposed_non_model) < -0.001 * sum(wall):
        raise RuntimeError(
            "aggregate model GPU time materially exceeded decode cadence"
        )
    return {
        "step_wall_ms": distribution(wall),
        "model_gpu_ms": distribution(model_gpu),
        "exposed_non_model_ms": distribution(exposed_non_model),
        "model_fraction": sum(model_gpu) / sum(wall),
        "exposed_non_model_fraction": sum(exposed_non_model) / sum(wall),
    }


def _trace_components(
    trace: dict[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    """Read Nano's canonical names while retaining old trace compatibility."""
    model_key = "model_gpu_ms" if "model_gpu_ms" in trace else "forward_gpu_ms"
    exposed_key = (
        "exposed_non_model_ms"
        if "exposed_non_model_ms" in trace
        else "non_forward_ms"
    )
    return (
        [float(value) for value in trace["step_wall_ms"]],
        [float(value) for value in trace[model_key]],
        [float(value) for value in trace[exposed_key]],
    )


def summarize_uno_cycle_trace(
    trace: dict[str, Any],
    *,
    engine: str,
    batch_size: int,
) -> dict[str, Any]:
    """Summarize stationary two-pass UNO draft/verify cycles."""
    wall = [float(value) for value in trace["step_wall_ms"]]
    records = trace["forward_records"]
    cycles: list[tuple[float, dict[str, Any], dict[str, Any]]] = []

    if engine == "nano":
        by_step: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_step.setdefault(int(record["step_index"]), []).append(record)
        for index, cycle_wall in enumerate(wall, start=1):
            pair = by_step.get(index, [])
            if len(pair) == 2 and all(
                int(record["batch_size"]) == batch_size for record in pair
            ):
                cycles.append((cycle_wall, pair[0], pair[1]))
        # Drop the first cycle, which still includes the outer cold-start step.
        cycles = cycles[1:]
    else:
        raise ValueError(f"unknown UNO timing engine: {engine}")

    if not cycles:
        raise RuntimeError("trace has no stationary full-cohort UNO cycles")
    draft_ms = [float(draft["gpu_ms"]) for _, draft, _ in cycles]
    verify_ms = [float(verify["gpu_ms"]) for _, _, verify in cycles]
    cycle_ms = [cycle_wall for cycle_wall, _, _ in cycles]
    overhead_ms = [
        total - draft - verify
        for total, draft, verify in zip(cycle_ms, draft_ms, verify_ms)
    ]
    if sum(overhead_ms) < -0.001 * sum(cycle_ms):
        raise RuntimeError("forward GPU time materially exceeded cycle wall time")
    selected_records = [
        record
        for _, draft, verify in cycles
        for record in (draft, verify)
    ]
    return {
        "scope": "stationary_full_cohort_uno_cycles",
        "cycle_count": len(cycles),
        "all_cuda_graph_hits": all(
            bool(record["cuda_graph_hit"]) for record in selected_records
        ),
        "lm_head_in_cuda_graph": bool(trace["lm_head_in_cuda_graph"]),
        "cycle_wall_ms": distribution(cycle_ms),
        "draft_forward_ms": distribution(draft_ms),
        "verify_forward_ms": distribution(verify_ms),
        "other_overhead_ms": distribution(overhead_ms),
    }
def summarize_nano_ar_trace(
    trace: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    wall, model_gpu, exposed_non_model = _trace_components(trace)
    if not wall or not (len(wall) == len(model_gpu) == len(exposed_non_model)):
        raise RuntimeError("nano timing arrays are empty or misaligned")

    records_by_step: dict[int, list[dict[str, Any]]] = {}
    for record in trace["forward_records"]:
        records_by_step.setdefault(int(record["step_index"]), []).append(record)

    eligible: list[int] = []
    for index in range(len(wall)):
        records = records_by_step.get(index + 1, [])
        if (
            len(records) == NANO_NORMALIZATION_DIVISOR
            and all(
                int(record["batch_size"]) == batch_size
                and int(record["block_len"]) == 1
                for record in records
            )
        ):
            eligible.append(index)

    # Exclude pipeline/cold-start behavior from both engines. The first nano
    # cycle is complete, but it is not representative of stationary decode.
    selected = eligible[1:]
    if not selected:
        raise RuntimeError("nano trace has no stationary full-cohort AR cycles")

    selected_records = [
        record
        for index in selected
        for record in records_by_step[index + 1]
    ]
    all_graph_hits = all(
        bool(record["cuda_graph_hit"]) for record in selected_records
    )
    raw_wall = [wall[index] for index in selected]
    raw_model_gpu = [model_gpu[index] for index in selected]
    raw_exposed_non_model = [exposed_non_model[index] for index in selected]
    divisor = float(NANO_NORMALIZATION_DIVISOR)
    unit_wall = [value / divisor for value in raw_wall]
    unit_model_gpu = [value / divisor for value in raw_model_gpu]
    unit_exposed_non_model = [value / divisor for value in raw_exposed_non_model]

    return {
        "scope": "stationary_full_cohort_ar_decode",
        "unit": "one_forward_and_one_generated_token",
        "normalization": {
            "description": (
                "one normal nano block-1 scheduler cycle contains two "
                "forwards and emits two tokens; every component is divided by 2"
            ),
            "divisor": NANO_NORMALIZATION_DIVISOR,
            "raw_forwards_per_cycle": 2,
            "raw_tokens_per_cycle": 2,
        },
        "full_batch_step_count": len(selected),
        "excluded_initial_step_count": 1,
        "unaligned_step_count": len(wall) - len(eligible),
        "all_cuda_graph_hits": all_graph_hits,
        **_component_summary(unit_wall, unit_model_gpu, unit_exposed_non_model),
        "raw_cycle": _component_summary(
            raw_wall,
            raw_model_gpu,
            raw_exposed_non_model,
        ),
    }


def summarize_sglang_ar_trace(
    trace: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    records = [
        record
        for record in trace["forward_records"]
        if int(record["batch_size"]) == batch_size
    ]
    # Record zero is the overlap-pipeline fill interval between decode launches
    # one and two. Subsequent launch intervals are the stationary output cadence.
    selected = records[1:]
    if not selected:
        raise RuntimeError("SGLang trace has no stationary full-cohort AR steps")
    wall = [float(record["step_wall_ms"]) for record in selected]
    model_gpu = [
        float(record["gpu_ms"] if "gpu_ms" in record else record["forward_gpu_ms"])
        for record in selected
    ]
    exposed_non_model = [
        float(
            record["exposed_non_model_ms"]
            if "exposed_non_model_ms" in record
            else record["non_forward_ms"]
        )
        for record in selected
    ]
    return {
        "scope": "stationary_full_cohort_ar_decode",
        "unit": "one_forward_and_one_generated_token",
        "normalization": {
            "description": "one SGLang AR decode step is already one forward",
            "divisor": 1,
            "raw_forwards_per_cycle": 1,
            "raw_tokens_per_cycle": 1,
        },
        "full_batch_step_count": len(selected),
        "excluded_initial_step_count": 1,
        "unaligned_step_count": len(trace["forward_records"]) - len(records),
        "all_cuda_graph_hits": all(
            bool(record["cuda_graph_hit"]) for record in selected
        ),
        **_component_summary(wall, model_gpu, exposed_non_model),
    }
