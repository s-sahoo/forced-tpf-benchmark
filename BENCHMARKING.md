# GPU benchmark workflow

Install the pinned binary runtimes and configure checkpoint paths as described
in `REPRODUCING.md`. The launch scripts prefer an explicit `BENCHMARK_ENV`,
then `${FORCED_TPF_ENV_ROOT:-.venvs/forced-tpf}/benchmark.env`. On K2M they
fall back to the existing shared paper runtime when neither is configured.

The complete paper workload and all engine-ready forced TPF values live in
`configs/forced_tpf_calibration.json`. Figure manifests in `configs/figures/`
only select catalog IDs, launch adapters, and curve labels.

Every point uses one GPU, TP=DP=1, token-only I/O, deterministic distinct
warmup/timed prompts, and end-to-end `generate()` timing that includes request
admission, prefill, and decode. Engine initialization and warmup generation are
excluded. The planner attempts C1 through C128 independently and plots only
complete, resident, non-preempted points.

Run one or more figures on an allocated GPU node:

```bash
scripts/run_figures.sh \
  --figures figure1 figure3_t0 figure3_t1 figure4 \
  --out-root /path/to/results \
  --gpus 0,1,2,3,4,5,6,7
```

Or allocate and run through Slurm:

```bash
sbatch --nodes=8 paper_scripts/run_figures.slurm \
  figure1 figure3_t0 figure3_t1 figure4
```

Set `PAPER_RUN_ROOT=/shared/path` on submission to choose the standard-suite
run directory. Figure 2 uses `FIGURE2_RUN_ROOT` in the same way.

The calibration requests three repetitions per point. Eight nodes provide 64
dynamic GPU slots; they do not change the repetition count. Repetitions of the
same point are assigned to distinct nodes when enough nodes are available.

The cache key includes the resolved workload, calibration parameters, adapter,
batch, repetition, point-runner source, and vendored-engine provenance. Thus
unchanged points are shared across figures and resumed; changed parameters or
code produce a new key. Outputs are under `OUT_ROOT/experiments/`, with plots
and normalized candidate CSVs under `OUT_ROOT/figures/`. `OUT_ROOT` may be any
writable local or shared path.

Every engine writes the same result fields used by the planner:
`requested_tpf`, `measured_tpf`, `aggregate_tps`, `per_user_tps`,
`end_to_end_elapsed_s`, `peak_gpu_memory_mib`, token counts, and
cohort/CUDA-graph validity.
Engine-specific diagnostics remain alongside that common contract. The
normalized figure CSVs share one schema and omit internal cache keys.
`measured_tpf` always means useful generated tokens per decode forward;
speculative engines may additionally report `acceptance_tpf`.

`peak_gpu_memory_mib` is the maximum total memory reported for the physical
GPU at 100 ms intervals while the engine point is running. Because each node
is exclusive and each point owns one GPU, this includes model weights, CUDA
graphs, runtime workspaces, and—for speculative engines—the draft model and
its KV cache. Aggregation uses median throughput and the maximum peak memory
across the three repetitions. This is the configured runtime footprint: an
engine that preallocates a KV pool reports that reservation, so the value is
not a separately tuned theoretical minimum for the point.

Figure 2 measures one natural-acceptance UNO draft-and-verify cycle rather
than a forced-TPF sweep. Run its three paired same-GPU repetitions with:

```bash
scripts/reproduce_figure2.sh /path/to/figure2-run --gpus 0,1,2
```

This writes the candidate table and plot under `/path/to/figure2-run`.

After reviewing any run, promote selected figures with the plotting Python
defined by `benchmark.env`:

```bash
"$PLOT_PYTHON" paper/promote_run.py /path/to/results \
  --figures figure1 figure3_t0 figure3_t1 figure4
"$PLOT_PYTHON" paper/promote_run.py /path/to/figure2-run --figures figure2
```

Promotion copies only normalized measurements into `paper/data/`, refreshes
their checksums, and regenerates the selected official plots. Raw result paths
remain solely in the run directory and are deliberately not embedded in the
paper CSVs.
