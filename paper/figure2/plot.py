#!/usr/bin/env python3
"""Render the historical diffusion-cycle latency breakdown."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "forced-tpf-benchmark"
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data/figure2.csv"
COMPONENTS = (
    ("draft_forward_ms", "Draft forward", "#1f77b4"),
    ("verify_forward_ms", "Verify forward", "#ff7f0e"),
    ("other_overhead_ms", "Other overhead", "#2ca02c"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def load_rows(data: Path) -> list[dict[str, str]]:
    with data.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["engine"] for row in rows] != ["nano"]:
        raise RuntimeError("Figure 2 requires exactly one Nano row")
    for row in rows:
        components = sum(float(row[key]) for key, _, _ in COMPONENTS)
        if abs(components - float(row["total_ms"])) > 0.011:
            raise RuntimeError(f"Figure 2 components do not sum for {row['engine']}")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
        }
    )
    fig, axis = plt.subplots(figsize=(12.5, 2.6), dpi=180)
    positions = [0]
    left = [0.0]
    for key, label, color in COMPONENTS:
        values = [float(row[key]) for row in rows]
        bars = axis.barh(
            positions,
            values,
            left=left,
            height=0.52,
            color=color,
            label=label,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )
        left = [start + value for start, value in zip(left, values)]

    totals = [float(row["total_ms"]) for row in rows]
    for position, total in zip(positions, totals):
        axis.text(
            total + 0.35,
            position,
            f"{total:.2f} ms",
            va="center",
            weight="bold",
        )
    axis.set_yticks(positions, [row["engine_label"] for row in rows])
    axis.set_xlabel("Full draft+verify cycle latency (ms) — lower is better")
    axis.set_xlim(0, 14.5)
    axis.xaxis.grid(True, color="#dddddd", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(-0.4, 0.75)
    axis.legend(loc="upper center", ncol=3, frameon=True)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"figure2.{suffix}", bbox_inches="tight")
    plt.close(fig)
    svg = args.output_dir / "figure2.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
