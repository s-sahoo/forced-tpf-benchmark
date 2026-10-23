#!/usr/bin/env python3
"""Promote selected reviewed run outputs into the paper and render them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path


PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
THROUGHPUT_FIELDS = (
    "figure",
    "curve",
    "experiment_id",
    "model_id",
    "configuration_id",
    "adapter",
    "batch_size",
    "input_len",
    "output_len",
    "temperature",
    "requested_tpf",
    "measured_tpf",
    "aggregate_tps",
    "per_user_tps",
    "peak_gpu_memory_mib",
)
FIGURE2_FIELDS = (
    "engine",
    "engine_label",
    "draft_forward_ms",
    "verify_forward_ms",
    "other_overhead_ms",
    "total_ms",
)
SPECS = {
    "figure1": (
        Path("figures/figure1/all_valid_points.csv"),
        PAPER / "data/figure1.csv",
        THROUGHPUT_FIELDS,
        (PAPER / "figure1/plot.py",),
    ),
    "figure2": (
        Path("figure2.csv"),
        PAPER / "data/figure2.csv",
        FIGURE2_FIELDS,
        (PAPER / "figure2/plot.py",),
    ),
    "figure3_t1": (
        Path("figures/figure3_t1/all_valid_points.csv"),
        PAPER / "data/figure3_t1.csv",
        THROUGHPUT_FIELDS,
        (PAPER / "figure3/plot.py", "--temperature", "1"),
    ),
    "figure4": (
        Path("figures/figure4/all_valid_points.csv"),
        PAPER / "data/figure4.csv",
        THROUGHPUT_FIELDS,
        (PAPER / "figure4/plot.py",),
    ),
}


def load_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No valid measurements: {path}")
    missing = set(fields) - set(reader.fieldnames or ())
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
    return [{field: row[field] for field in fields} for row in rows]


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_checksums() -> None:
    destinations = sorted({spec[1] for spec in SPECS.values()})
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in destinations
    ]
    (PAPER / "data/SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=tuple(SPECS),
        required=True,
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()

    selected = {}
    for figure in args.figures:
        relative_input, _, fields, _ = SPECS[figure]
        selected[figure] = load_rows(run_root / relative_input, fields)

    for figure, rows in selected.items():
        _, destination, fields, _ = SPECS[figure]
        write_csv(destination, fields, rows)
    write_checksums()
    for figure in selected:
        command = SPECS[figure][3]
        subprocess.run(
            [sys.executable, *(str(item) for item in command)],
            cwd=REPO,
            check=True,
        )

    print(", ".join(f"{figure}: {len(rows)}" for figure, rows in selected.items()))


if __name__ == "__main__":
    main()
