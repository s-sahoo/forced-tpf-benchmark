#!/usr/bin/env python3
"""Compare normalized rerun measurements with the reviewed paper CSVs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


PAPER = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["curve"], row["configuration_id"], int(row["batch_size"])


def canonical_tps(row: dict[str, str]) -> float:
    return float(row["aggregate_tps"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("1", "3_t1", "4"),
        default=("1", "3_t1", "4"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    comparison = []
    missing = []
    for figure in args.figures:
        rerun_name = f"figure{figure}"
        rerun_path = args.run_root / "figures" / rerun_name / "all_valid_points.csv"
        rerun = {
            (row["curve"], row["configuration_id"], int(row["batch_size"])): row
            for row in rows(rerun_path)
        }
        for expected in rows(PAPER / "data" / f"figure{figure}.csv"):
            key = canonical_key(expected)
            actual = rerun.get(key)
            if actual is None:
                missing.append(f"figure{figure}:{key}")
                continue
            before = canonical_tps(expected)
            after = float(actual["aggregate_tps"])
            comparison.append(
                {
                    "figure": figure,
                    "curve": key[0],
                    "configuration_id": key[1],
                    "batch_size": key[2],
                    "current_tps": before,
                    "rerun_tps": after,
                    "delta_tps": after - before,
                    "delta_percent": 100.0 * (after / before - 1.0),
                }
            )

    output = args.output or args.run_root / "paper_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(comparison[0]) if comparison else ["figure"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)

    for figure in args.figures:
        deltas = [row["delta_percent"] for row in comparison if row["figure"] == figure]
        if deltas:
            print(
                f"Figure {figure}: {len(deltas)} matched points, "
                f"median {statistics.median(deltas):+.2f}%, "
                f"range [{min(deltas):+.2f}%, {max(deltas):+.2f}%]"
            )
    if missing:
        print(f"Missing {len(missing)} reviewed points:")
        for item in missing:
            print(f"- {item}")
    print(f"Point comparison: {output}")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
