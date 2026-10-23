#!/usr/bin/env python3
"""Regenerate the official Figure 4 throughput frontiers."""

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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from benchmark.plot import pareto, point_label, write_csv


DATA = REPO / "paper/data/figure4.csv"
INPUT_LEN = 1024
OUTPUT_LEN = 8192
CURVES = {
    "uno": ("Nano-Uno BBQ-8B", "#1565c0", "o", "-"),
    "ar": ("Nano-Uno BBQ-8B AR", "#7b1fa2", "^", "--"),
    "nemotron": (
        "Nemotron Diffusion-14B LinearSpec-LoRA",
        "#ef6c00",
        "D",
        "-",
    ),
    "diffgemma": (
        "DiffusionGemma-26B-A4B vLLM",
        "#00897b",
        "s",
        "-",
    ),
}


def load_points(data: Path) -> list[dict[str, object]]:
    with data.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        if row["curve"] not in CURVES:
            raise ValueError(f"unknown Figure 4 curve: {row['curve']}")
        if (
            int(row["input_len"]) != INPUT_LEN
            or int(row["output_len"]) != OUTPUT_LEN
        ):
            raise ValueError("Figure 4 contains a non-1K/8K workload")
        points.append(
            {
                **row,
                "batch_size": int(row["batch_size"]),
                "aggregate_tps": float(row["aggregate_tps"]),
                "per_user_tps": float(row["per_user_tps"]),
            }
        )
    if {point["curve"] for point in points} != set(CURVES):
        raise RuntimeError("Figure 4 does not contain all four methods")
    return points


def annotate(
    axis: plt.Axes,
    points: list[dict[str, object]],
    curve: str,
    color: str,
) -> None:
    for index, point in enumerate(points):
        batch = int(point["batch_size"])
        if curve != "uno" and batch not in {1, 4, 16, 32, 64}:
            continue
        label = point_label(point) if curve == "uno" else f"C{batch}"
        offset = (6, 7 if index % 2 == 0 else -15)
        if curve == "nemotron":
            offset = (6, -15)
        axis.annotate(
            label,
            (point["per_user_tps"], point["aggregate_tps"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=11,
            color=color,
        )


def plot(points: list[dict[str, object]], output_dir: Path) -> None:
    frontier = [
        point
        for curve in CURVES
        for point in pareto([row for row in points if row["curve"] == curve])
    ]
    write_csv(output_dir / "figure4_frontier.csv", frontier)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    figure, axis = plt.subplots(figsize=(9.0, 7.4), dpi=180)
    for curve, (name, color, marker, linestyle) in CURVES.items():
        selected = [point for point in frontier if point["curve"] == curve]
        axis.plot(
            [point["per_user_tps"] for point in selected],
            [point["aggregate_tps"] for point in selected],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.8,
            markersize=7.5,
            label=name,
            zorder=3,
        )
        annotate(axis, selected, curve, color)

    axis.set_xscale("log")
    axis.set_xlim(13, 900)
    axis.set_ylim(0, 5600)
    axis.xaxis.set_major_locator(FixedLocator([16, 25, 50, 100, 200, 400, 800]))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="major", linestyle="--", alpha=0.26)
    axis.grid(True, which="minor", linestyle=":", alpha=0.10)
    axis.set_axisbelow(True)
    axis.set_xlabel("tok/s/stream", weight="bold")
    axis.set_ylabel("tok/s", weight="bold")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncols=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.8)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"figure4.{suffix}", bbox_inches="tight")
    plt.close(figure)
    svg = output_dir / "figure4.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(load_points(args.data), args.output_dir)


if __name__ == "__main__":
    main()
