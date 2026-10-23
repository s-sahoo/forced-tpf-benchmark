# Active helper scripts

- `install.sh`: install all pinned binary runtimes for Figures 1–4; no GPU or
  host Python is required.
- `doctor.sh`: validate isolation and package versions on a login node, or
  imports and CUDA with `--gpu` inside a one-GPU allocation.
- `build_wheelhouse.sh`: download the complete pinned wheel set for an offline
  installation.
- `run_figures.sh`: shared local experiment-to-plot entrypoint.
- `reproduce_figure2.sh`: paired Figure 2 timing experiment and plot.
- Engine-family launch policies live in `benchmark/adapters/`; the GPU point
  programs they invoke live in `benchmark/runners/`.
- `prepare_vllm_diffgemma_cu129_runtime.sh`: overlay the vendored vLLM delta
  on the pinned prebuilt wheel.

Use `paper_scripts/run_figures.slurm` to shard the same workflow with Slurm.
Use `paper_scripts/run_figure2.slurm` for the dedicated Figure 2 measurement.
The ordinary forced-TPF suite is Figures 1, 3 (t=0 and t=1), and 4.
