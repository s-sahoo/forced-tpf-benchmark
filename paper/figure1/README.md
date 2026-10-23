# Figure 1: serving-engine frontiers

The reviewed 1K-input/8K-output, temperature-1 figure compares Nano Uno
(Pareto frontier of Linear-4 and Tree-V60/N16), Nano Qwen AR, and SGLang Qwen
AR.

The checked-in AR curves use end-to-end measurements with detailed tracing
disabled. Nano and speculative SGLang both use SGL-Kernel FA3; SGLang keeps
its production overlap scheduler enabled.

`plot.py` reads `paper/data/figure1.csv`, derives the frontier table, and
writes all three image formats. To rerun the GPU matrix and produce a plot
from fresh JSON results:

```bash
sbatch --nodes=8 paper_scripts/run_figures.slurm figure1
```

The current experiment configuration resolves Tree-V60/N16/K32 and every TPF
from `configs/forced_tpf_calibration.json`.
