# Paper experiment entrypoints

From the repository root:

```bash
sbatch --nodes=8 paper_scripts/run_figures.slurm \
  figure1 figure3_t0 figure3_t1 figure4
```

The entrypoint loads `BENCHMARK_ENV`, shards exact experiment points across
nodes and GPUs, resumes matching results, and creates every requested plot. If
`BENCHMARK_ENV` is unset, it uses the repository-local runtime installed at
`.venvs/forced-tpf` (or the root named by `FORCED_TPF_ENV_ROOT`), then the
existing shared K2M runtime when that path is available.
The checked-in calibration runs three repetitions per point; the node count
only controls the number of dynamic GPU workers.

See `BENCHMARKING.md` for required assets and output overrides.

Figure 2 uses a separate single-node timing job:

```bash
sbatch paper_scripts/run_figure2.slurm
```
