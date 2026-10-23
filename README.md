# Forced-TPF benchmark

This repository reproduces paper Figures 1–4. It contains the exact
engine source snapshots, forced-TPF calibration, GPU launchers, reviewed CSV
measurements, and plotting code used by those figures.

## Quick start

Regenerate all four checked-in figures without GPUs:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-plot.txt
make figures PYTHON=.venv/bin/python
```

Install every GPU runtime:

```bash
scripts/install.sh
cp .venvs/forced-tpf/assets.env.example .venvs/forced-tpf/assets.env
# Edit assets.env with model and LoRA paths.
```

Installation is CPU-only and uses pinned prebuilt wheels; it does not require
a host Python, CUDA toolkit, GPU allocation, or access to K2M. Validate the
installed CUDA runtimes later on one H200 with
`srun --gres=gpu:1 scripts/doctor.sh --gpu`.

Then submit one experiment-to-plot workflow. Identical points requested by
multiple figures are run once and shared:

```bash
sbatch --nodes=8 paper_scripts/run_figures.slurm \
  figure1 figure3_t0 figure3_t1 figure4
```

Figure 2 has a dedicated paired timing workflow:

```bash
sbatch paper_scripts/run_figure2.slurm
```

GPU runs write candidate CSVs and plots under their chosen run directory.
After reviewing them, promote only the figures you want to publish; the same
command replaces their canonical CSVs, refreshes checksums, and regenerates
their official plots:

```bash
source .venvs/forced-tpf/benchmark.env
"$PLOT_PYTHON" paper/promote_run.py /path/to/run \
  --figures figure1 figure3_t0 figure3_t1 figure4
"$PLOT_PYTHON" paper/promote_run.py /path/to/figure2-run --figures figure2
```

`PLOT_PYTHON` is defined by `benchmark.env`; a local Python environment with
`requirements-plot.txt` installed works too. Run directories may live anywhere
on local or shared storage and are not referenced by the checked-in tables.

See `REPRODUCING.md` for prerequisites and `BENCHMARKING.md` for experiment
details.

## Layout

- `paper/data/`: reviewed machine-readable inputs for Figures 1–4.
- `paper/figure{1,2,3,4}/`: plot code and rendered figures; Figures 1, 3, and 4
  also contain generated frontier tables.
- `benchmark/`: planner, exact result cache, launch adapters, and plotting.
- `configs/forced_tpf_calibration.json`: shared workload and experiment catalog.
- `configs/figures/`: minimal declarative figure views.
- `scripts/`: user-facing installation and experiment entrypoints.
- `benchmark/adapters/`: one launch-policy module per runtime family.
- `benchmark/runners/`: GPU point programs invoked inside engine environments.
- `engines/`: five pinned engine source trees; see `ENGINES.md`.

Raw runs, checkpoints, caches, and environments are intentionally ignored.
The pre-cleanup repository is preserved on branch
`archive/pre-paper-reproduction-cleanup`.
