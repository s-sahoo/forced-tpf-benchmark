#!/usr/bin/env python3
"""Regenerate the official Figure 1 throughput frontiers."""

from __future__ import annotations

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


DATA = REPO / "paper/data/figure1.csv"
CURVES = {
    "nano_uno": ("Nano-vLLM Uno", "#1565c0", "o", "-"),
    "nano_ar": ("Nano-vLLM Qwen AR", "#37474f", "P", "--"),
    "sglang_ar": ("SGLang Qwen AR", "#ef6c00", "s", "--"),
}


def load_points() -> list[dict[str, object]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        if row["curve"] not in CURVES:
            raise ValueError(f"unknown Figure 1 curve: {row['curve']}")
        points.append(
            {
                **row,
                "batch_size": int(row["batch_size"]),
                "aggregate_tps": float(row["aggregate_tps"]),
                "per_user_tps": float(row["per_user_tps"]),
            }
        )
    if {point["curve"] for point in points} != set(CURVES):
        raise RuntimeError("Figure 1 does not contain all four curves")
    return points


def main() -> None:
    points = load_points()
    frontier = [
        point
        for curve in CURVES
        for point in pareto([row for row in points if row["curve"] == curve])
    ]
    write_csv(HERE / "figure1_frontier.csv", frontier)

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
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
    axis.set_xlabel("tok/s/stream")
    axis.set_ylabel("tok/s")
    axis.grid(True, linestyle="--", alpha=0.28)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(HERE / f"figure1.{suffix}", bbox_inches="tight")
    plt.close(figure)
    svg = HERE / "figure1.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
