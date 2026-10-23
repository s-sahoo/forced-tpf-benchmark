#!/usr/bin/env python3
"""Regenerate the temperature-1 variant of the official Figure 3."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "forced-tpf-benchmark"
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from benchmark.plot import pareto, point_label, write_csv


CURVES = {
    "uno": ("Uno", "#1565c0", "o", "-"),
    "eagle3": ("EAGLE-3", "#ef6c00", "s", "-"),
    "dflash": ("DFlash", "#00897b", "D", "-"),
    "ar": ("AR", "#37474f", "P", "--"),
}
def load_points(temperature: int) -> list[dict[str, object]]:
    path = REPO / "paper/data" / f"figure3_t{temperature}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        if row["curve"] not in CURVES:
            raise ValueError(f"unknown Figure 3 curve: {row['curve']}")
        points.append(
            {
                **row,
                "batch_size": int(row["batch_size"]),
                "aggregate_tps": float(row["aggregate_tps"]),
                "per_user_tps": float(row["per_user_tps"]),
            }
        )
    if {point["curve"] for point in points} != set(CURVES):
        raise RuntimeError(f"Figure 3 t={temperature} omits a method")
    return points


def annotation_offset(point: dict[str, object]) -> tuple[int, int]:
    curve = str(point["curve"])
    batch = int(point["batch_size"])
    if batch == 64:
        return {
            "uno": (6, -18),
            "dflash": (6, 6),
            "eagle3": (6, 16),
        }.get(curve, (6, 7))
    if curve == "uno" and batch in {4, 8, 16}:
        return (6, -18)
    if curve == "dflash" and batch in {1, 2}:
        return (6, -18)
    return (6, 7)


def plot(temperature: int) -> None:
    points = load_points(temperature)
    frontier = [
        point
        for curve in CURVES
        for point in pareto([row for row in points if row["curve"] == curve])
    ]
    write_csv(HERE / f"figure3_t{temperature}_frontier.csv", frontier)

    plt.rcParams.update(
        {"font.size": 15, "axes.labelsize": 19, "legend.fontsize": 14}
    )
    figure, axis = plt.subplots(figsize=(12, 8), dpi=180)
    for curve, (name, color, marker, linestyle) in CURVES.items():
        selected = [point for point in frontier if point["curve"] == curve]
        axis.plot(
            [point["per_user_tps"] for point in selected],
            [point["aggregate_tps"] for point in selected],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.8,
            markersize=8,
            label=name,
        )
        for point in selected:
            axis.annotate(
                point_label(point),
                (point["per_user_tps"], point["aggregate_tps"]),
                xytext=annotation_offset(point),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
    axis.set_xlabel("tok/s/stream")
    axis.set_ylabel("tok/s")
    axis.grid(True, linestyle="--", alpha=0.28)
    axis.legend()
    figure.tight_layout()
    stem = HERE / f"figure3_t{temperature}"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(figure)
    svg = stem.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=int, choices=(1,), action="append")
    args = parser.parse_args()
    for temperature in args.temperature or (1,):
        plot(temperature)


if __name__ == "__main__":
    main()
