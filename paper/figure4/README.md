# Figure 4: BBQ and diffusion-baseline frontiers

The 1K-input/8K-output plot compares the Pareto frontier over six Nano BBQ Uno
sampler configurations with Nano BBQ AR, Nemotron Diffusion-14B
LinearSpec-LoRA, and vLLM DiffusionGemma-26B-A4B. Every point is one H200 and
includes prefill.

```bash
sbatch --nodes=8 paper_scripts/run_figures.slurm figure4
```

The workflow attempts C1–C128 independently, validates every completed cohort,
and plots the valid Pareto frontier. The current canonical data contains every
feasible C1–C64 point; DiffusionGemma stops at C32 because C64 exceeds its
validated KV-cache capacity.
