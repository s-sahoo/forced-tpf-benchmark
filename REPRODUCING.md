# Reproduction guide

## Plot the reviewed measurements

This CPU-only path validates checksums and regenerates PNG, PDF, and SVG files
for Figures 1–4 plus Pareto-frontier CSVs for Figures 1, 3, and 4:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-plot.txt
make figures PYTHON=.venv/bin/python
```

The exact plot inputs are under `paper/data/`. They contain only normalized
measurements; the original raw run directories are not required.

## Re-run the GPU measurements

Installation requires Linux x86_64 with glibc 2.31+, `curl`, network access,
and enough local/shared disk space. It does **not** require a host Python, GPU,
CUDA toolkit, Slurm allocation, or model checkpoint. Install every isolated
runtime with pinned prebuilt wheels:

```bash
scripts/install.sh
```

The installer bootstraps pinned `uv` and Python 3.12, creates four isolated
environments under `.venvs/forced-tpf/`, shares one download cache, forbids
source builds, overlays the vendored forced-TPF vLLM delta, validates package
metadata, and writes `.venvs/forced-tpf/benchmark.env`. It does not load a
model, reserve a GPU, run an experiment, or make a plot.

On the K2M cluster, the launchers retain a final fallback to the existing
shared runtime at `/mnt/weka/shrd/k2m/duc.pham/forced-tpf-runtime`. Therefore
the established Slurm commands continue to work unchanged for users who can
read that directory. A newly installed local or explicitly selected runtime
takes precedence.

Configure model and LoRA paths separately:

```bash
cp .venvs/forced-tpf/assets.env.example .venvs/forced-tpf/assets.env
$EDITOR .venvs/forced-tpf/assets.env
```

Paths can be local checkpoint directories or Hugging Face model IDs. To keep
assets elsewhere, set `FORCED_TPF_ASSET_ENV=/absolute/path/assets.env` before
sourcing or using `benchmark.env`.

On a CUDA-12.9-driver-compatible H200 node, validate imports and CUDA without
loading checkpoints:

```bash
srun --gres=gpu:1 scripts/doctor.sh --gpu
```

For a machine without internet access, prepare and copy a wheelhouse from an
internet-connected Linux x86_64 host:

```bash
scripts/build_wheelhouse.sh /path/to/wheelhouse
scripts/install.sh --wheelhouse /path/to/wheelhouse
```

After installation, submit the figures you want in one command:

```bash
sbatch --nodes=8 paper_scripts/run_figures.slurm \
  figure1 figure3_t0 figure3_t1 figure4
```

Figure 2 uses detailed, opt-in engine instrumentation and three paired
same-GPU repetitions:

```bash
sbatch paper_scripts/run_figure2.slurm
```

The workflow runs one engine process per GPU, resumes only exact matching
result JSONs, rejects incomplete or non-resident cohorts, and plots each
requested figure. Use `--nodes=N` at submission time to shard independent
points over multiple eight-GPU nodes. See `BENCHMARKING.md` for details.
The checked-in calibration fixes the repetition count at three; increasing
`--nodes` only supplies more dynamically allocated GPU slots. Each result JSON
and normalized figure CSV records the total device-level peak GPU memory.

The Slurm scripts print the run directory when they finish. Review its
candidate CSVs and plots, then replace selected official artifacts with:

```bash
source .venvs/forced-tpf/benchmark.env
"$PLOT_PYTHON" paper/promote_run.py /path/to/run \
  --figures figure1 figure3_t0 figure3_t1 figure4
"$PLOT_PYTHON" paper/promote_run.py /path/to/figure2-run --figures figure2
```

On K2M, source the shared `benchmark.env` instead if no local runtime was
installed. A run directory can be anywhere; promotion does not store its path
in the canonical data.
