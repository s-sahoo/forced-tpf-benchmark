#!/usr/bin/env python3
"""Validate the shared paper workload, calibration catalog, and figure views."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "configs/forced_tpf_calibration.json"
FIGURES = REPO / "configs/figures"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    catalog = read(CATALOG)
    if catalog.get("schema_version") != 3:
        raise ValueError("forced-TPF calibration must use schema version 3")
    workload = catalog["benchmark"]
    required = {
        "input_len", "output_len", "batch_sizes", "top_p", "top_k", "timed_repetitions"
    }
    if set(workload) != required:
        raise ValueError(f"benchmark keys must be exactly {sorted(required)}")
    if workload["batch_sizes"] != [1, 2, 4, 8, 16, 32, 64, 128]:
        raise ValueError("the paper benchmark must attempt concurrency through 128")

    count = 0
    for path in sorted(FIGURES.glob("*.json")):
        manifest = read(path)
        if set(manifest) != {"schema_version", "experiments"}:
            raise ValueError(f"unexpected keys in {path}")
        for experiment in manifest["experiments"]:
            if set(experiment) != {"id", "adapter", "curve"}:
                raise ValueError(f"invalid experiment view in {path}: {experiment}")
            model, configuration = experiment["id"].split(".", 1)
            entry = catalog[model][configuration]
            if "temperature" not in entry.get("parameters", {}):
                raise ValueError(f"{experiment['id']} omits temperature")
            count += 1
    if count != 41:
        raise ValueError(f"expected 41 figure experiment views, found {count}")
    print(f"Validated {count} paper experiment views and the shared workload.")


if __name__ == "__main__":
    main()
