#!/usr/bin/env python3
"""Validate the checked-in measurement tables used by the paper figures."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
EXPECTED_FILES = {
    "figure1.csv",
    "figure2.csv",
    "figure3_t1.csv",
    "figure4.csv",
}
FRONTIER_FIELDS = {
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
}
FIGURE2_FIELDS = {
    "engine",
    "engine_label",
    "draft_forward_ms",
    "verify_forward_ms",
    "other_overhead_ms",
    "total_ms",
}


def checksums() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (DATA / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    return entries


def main() -> None:
    expected_hashes = checksums()
    if set(expected_hashes) != EXPECTED_FILES:
        raise RuntimeError("SHA256SUMS does not list exactly Figures 1–4")

    for name in sorted(EXPECTED_FILES):
        path = DATA / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hashes[name]:
            raise RuntimeError(f"Checksum mismatch: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            raise RuntimeError(f"Empty canonical data table: {path}")
        if name == "figure2.csv" and len(rows) != 1:
            raise RuntimeError(f"Figure 2 must contain exactly one engine: {path}")
        expected_fields = FIGURE2_FIELDS if name == "figure2.csv" else FRONTIER_FIELDS
        if set(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(f"Non-canonical schema: {path}")

    print("Validated canonical data for Figures 1–4.")


if __name__ == "__main__":
    main()
