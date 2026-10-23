from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "forced-tpf-benchmark"
import matplotlib.pyplot as plt


STYLES = {
    "figure1": {
        "nano_uno": ("Nano-vLLM Uno", "#1565c0", "o", "-"),
        "nano_ar": ("Nano-vLLM Qwen AR", "#37474f", "P", "--"),
        "sglang_ar": ("SGLang Qwen AR", "#ef6c00", "s", "--"),
    },
    "figure3_t1": {
        "uno": ("Uno", "#1565c0", "o", "-"),
        "eagle3": ("EAGLE-3", "#ef6c00", "s", "-"),
        "dflash": ("DFlash", "#00897b", "D", "-"),
        "ar": ("AR", "#37474f", "P", "--"),
    },
    "figure4": {
        "uno": ("Nano-Uno BBQ-8B", "#1565c0", "o", "-"),
        "ar": ("Nano-Uno BBQ-8B AR", "#7b1fa2", "^", "--"),
        "nemotron": ("Nemotron Diffusion-14B LinearSpec-LoRA", "#ef6c00", "D", "-"),
        "diffgemma": ("DiffusionGemma-26B-A4B vLLM", "#00897b", "s", "-"),
    },
}


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        x, y = row["per_user_tps"], row["aggregate_tps"]
        if not any(
            other is not row
            and other["per_user_tps"] >= x
            and other["aggregate_tps"] >= y
            and (other["per_user_tps"] > x or other["aggregate_tps"] > y)
            for other in rows
        ):
            frontier.append(row)
    return sorted(frontier, key=lambda item: item["per_user_tps"])


def point_label(row: dict[str, Any]) -> str:
    configuration = row["configuration_id"]
    batch = row["batch_size"]
    if configuration.startswith("ar_"):
        return f"C{batch}"
    name = configuration.split("_t", 1)[0]
    if name.startswith("uno("):
        b, k, v = name[4:-1].split(",")
        prefix = f"B{b}/K{k}/V{v}"
    elif name.startswith("eagle3("):
        _, _, v = name[7:-1].split(",")
        prefix = f"B{v}"
    elif name.startswith("dflash("):
        b = name[7:-1].split(",", 1)[0]
        prefix = f"B{b}"
    elif name.startswith("linearspec("):
        prefix = "LinearSpec"
    elif name.startswith("diffusion("):
        prefix = "DiffGemma"
    else:
        prefix = name
    return f"{prefix}/C{batch}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["figure", "curve"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(figure: str, rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    styles = STYLES[figure]
    frontier = []
    for curve in styles:
        frontier.extend(pareto([row for row in rows if row["curve"] == curve]))
    write_csv(output_dir / "all_valid_points.csv", rows)
    write_csv(output_dir / "frontier.csv", frontier)

    plt.rcParams.update({"font.size": 15, "axes.labelsize": 19, "legend.fontsize": 14})
    canvas, axis = plt.subplots(figsize=(12, 8), dpi=180)
    for curve, (label, color, marker, linestyle) in styles.items():
        selected = [row for row in frontier if row["curve"] == curve]
        if not selected:
            continue
        axis.plot(
            [row["per_user_tps"] for row in selected],
            [row["aggregate_tps"] for row in selected],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.8,
            markersize=8,
        )
        for row in selected:
            axis.annotate(
                point_label(row),
                (row["per_user_tps"], row["aggregate_tps"]),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
    axis.set_xlabel("tok/s/stream")
    axis.set_ylabel("tok/s")
    axis.grid(True, linestyle="--", alpha=0.28)
    axis.legend()
    canvas.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        canvas.savefig(output_dir / f"{figure}.{suffix}", bbox_inches="tight")
    plt.close(canvas)
    return frontier
