# Figure 3: lossless-method frontiers

Figure 3 compares the 1K-input/8K-output throughput frontiers for Uno,
EAGLE-3, DFlash, and AR at temperatures 0 and 1. The checked-in data uses
SGL-Kernel FA3 for the three speculative methods and untraced Nano AR.

`plot.py` reads every valid candidate in `paper/data/figure3_t0.csv` and
`figure3_t1.csv`, derives each method's Pareto frontier, and writes the
temperature-qualified frontier tables and image exports.

```bash
sbatch --nodes=8 paper_scripts/run_figures.slurm figure3_t0 figure3_t1
```

Matching Nano and AR experiments are automatically shared with Figure 1 when
both figures use the same output root.
